"""
autiz_model.py — single source of truth for the Autiz v2 model classes.

notebooks/autiz_v2.ipynb contains an identical copy of everything between the
"BEGIN SHARED MODEL CODE" and "END SHARED MODEL CODE" markers, in a cell marked
"KEEP IN SYNC with src/api/autiz_model.py". Change both together.

Design (v2):
    RoBERTa-large CLS [1024] --ContentBranch--> C [256] --content_head--> content score [1]
    GeMAPS (62, z-scored)    --ProsodyBranch--> D [256] --recon_head----> reconstruction of the 62 features
C is trained on the content-scoring task (ChaLearn First Impressions V2 transcripts) and then frozen.
D is trained to reconstruct the GeMAPS features while being statistically independent of C.
recon_head is for training only and is not needed for serving.

Checkpoint models/checkpoints/autiz_v2.pt is a dict with keys
    "content_branch", "content_head", "prosody_branch", "recon_head"  (state dicts)
    "config"  (plain python values: dims, seed, content_label_source, date, ...)
"""

import json

import numpy as np
import torch
import torch.nn as nn

# ===== BEGIN SHARED MODEL CODE =====

CONTENT_DIM = 1024
GEMAPS_DIM = 62
HIDDEN_DIM = 256


class ContentBranch(nn.Module):
    """RoBERTa CLS embedding -> C. Never sees audio."""

    def __init__(self, input_dim=CONTENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ProsodyBranch(nn.Module):
    """z-scored GeMAPS features -> D. Never sees the transcript."""

    def __init__(self, input_dim=GEMAPS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ReconHead(nn.Module):
    """D -> reconstruction of the z-scored GeMAPS input. Training only."""

    def __init__(self, hidden_dim=HIDDEN_DIM, output_dim=GEMAPS_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, d):
        return self.layers(d)


def build_models(hidden_dim=HIDDEN_DIM):
    """Fresh (randomly initialised) modules for every part of the v2 model, keyed like the checkpoint."""
    return {
        "content_branch": ContentBranch(CONTENT_DIM, hidden_dim),
        "content_head": nn.Linear(hidden_dim, 1),
        "prosody_branch": ProsodyBranch(GEMAPS_DIM, hidden_dim),
        "recon_head": ReconHead(hidden_dim, GEMAPS_DIM),
    }


# ===== END SHARED MODEL CODE =====


REQUIRED_CHECKPOINT_KEYS = ("content_branch", "content_head", "prosody_branch", "recon_head", "config")


def load_autiz_v2(path, device="cpu"):
    """Load autiz_v2.pt for serving.

    Returns (modules, config). `modules` holds content_branch, content_head and prosody_branch in eval
    mode with frozen parameters; recon_head is training-only and is not loaded.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # very old torch without the weights_only argument
        ckpt = torch.load(path, map_location=device)
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise KeyError(f"{path} is missing checkpoint keys: {missing}")
    config = ckpt["config"]
    modules = build_models(int(config.get("dims", {}).get("hidden", HIDDEN_DIM)))
    modules.pop("recon_head")
    for name, module in modules.items():
        module.load_state_dict(ckpt[name])
        module.to(device).eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return modules, config


def load_scaler(path):
    """Load gemaps_scaler.json -> dict with numpy 'mean' and 'std' and float 'z_clip'."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "feature_names": raw["feature_names"],
        "mean": np.asarray(raw["mean"], dtype=np.float64),
        "std": np.asarray(raw["std"], dtype=np.float64),
        "z_clip": float(raw["z_clip"]),
    }


def apply_scaler(features, scaler):
    """z-score raw GeMAPS ([..., 62]) with the saved scaler and clip to +-z_clip (same as training)."""
    z = (np.asarray(features, dtype=np.float64) - scaler["mean"]) / scaler["std"]
    return np.clip(z, -scaler["z_clip"], scaler["z_clip"]).astype(np.float32)
