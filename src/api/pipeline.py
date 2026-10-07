"""
pipeline.py: Autiz v2 serving pipeline.

    audio --ffmpeg--> 16 kHz mono wav --openSMILE--> 62 raw GeMAPS --apply_scaler--> z-scored features
    audio --Whisper--> transcript --RoBERTa-large--> CLS [1024] --ContentBranch--> C --content_head--> content score

Artifacts (folder AUTIZ_MODELS_DIR): autiz_v2.pt, gemaps_scaler.json, prosody_baseline.joblib.
They are loaded lazily and once. Files stamped "SMOKE RUN, NOT RESULTS" are refused unless
ALLOW_SMOKE_ARTIFACTS=1. Model classes and scaler helpers come from autiz_model.py; scaling is never
reimplemented here (it applies a z-score clip and a std floor). torch is imported lazily so that the
module imports (and the tests that stub `run`) work without it.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SMOKE_STAMP = "SMOKE RUN, NOT RESULTS"
PROXY_STAMP = "PROXY, not a trained scorer"
BASELINE_LABEL = "illustrative comparison model, not an evaluator or vendor tool"

CHECKPOINT_FILE = "autiz_v2.pt"
SCALER_FILE = "gemaps_scaler.json"
BASELINE_FILE = "prosody_baseline.joblib"
INDEPENDENCE_FILE = "eval_results_independence.json"

PITCH_FEATURE = "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"
UNVOICED_FEATURE = "MeanUnvoicedSegmentLength"
VOICED_RATE_FEATURE = "VoicedSegmentsPerSec"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Errors the routers turn into HTTP responses

class PipelineError(Exception):
    status_code = 500


class ArtifactError(PipelineError):
    status_code = 503


class MissingArtifactError(ArtifactError):
    pass


class SmokeArtifactError(ArtifactError):
    pass


class AudioDecodeError(PipelineError):
    status_code = 422


class NoSpeechError(PipelineError):
    status_code = 422


# Configuration (read from the environment at call time)

def _resolve_dir(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute() and not p.exists() and (_REPO_ROOT / p).exists():
        return _REPO_ROOT / p
    return p


def models_dir() -> Path:
    raw = os.environ.get("AUTIZ_MODELS_DIR")
    if raw:
        return _resolve_dir(raw)
    modal_path = Path("/root/models/checkpoints")
    if modal_path.is_dir():
        return modal_path
    return _REPO_ROOT / "models" / "checkpoints"


def results_dir() -> Path:
    raw = os.environ.get("AUTIZ_RESULTS_DIR")
    return _resolve_dir(raw) if raw else _REPO_ROOT / "src" / "eval" / "results"


def allow_smoke() -> bool:
    return os.environ.get("ALLOW_SMOKE_ARTIFACTS") == "1"


# Smoke-stamp handling. The notebook writes the stamp as: top-level "run_stamp" in every JSON (including
# gemaps_scaler.json), config["run_stamp"] in the checkpoint, a "run_stamp" key in the joblib dict, and a
# "# SMOKE RUN, NOT RESULTS" first line in CSVs (the pipeline loads no CSV). Any non-empty stamp counts.

def _json_stamp(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("run_stamp") if isinstance(data, dict) else None


def _enforce_smoke_policy(stamps: dict) -> bool:
    """Raise if any file is stamped and smoke is not allowed; otherwise return whether the set is smoke."""
    stamped = sorted(name for name, stamp in stamps.items() if stamp)
    if stamped and not allow_smoke():
        raise SmokeArtifactError(
            f"Refusing to load smoke artifact(s) {stamped} (stamped '{SMOKE_STAMP}'). Their numbers are not results. "
            "Set ALLOW_SMOKE_ARTIFACTS=1 to load them for testing."
        )
    return bool(stamped)


def smoke_artifacts_flag() -> bool:
    """Cheap (no torch) value of the response field smoke_artifacts: True only if allowed AND the set is smoke."""
    if not allow_smoke():
        return False
    if _artifacts is not None:
        return _artifacts["smoke"]
    try:
        return bool(_json_stamp(models_dir() / SCALER_FILE))
    except (OSError, ValueError):
        return False


# Artifact loading (lazy, once)

_artifacts = None
_encoders = None


def reset_cache():
    global _artifacts, _encoders
    _artifacts = None
    _encoders = None


def load_artifacts() -> dict:
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    d = models_dir()
    paths = {name: d / name for name in (CHECKPOINT_FILE, SCALER_FILE, BASELINE_FILE)}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise MissingArtifactError(
            f"Required model file(s) not found: {missing}. Run notebooks/autiz_v2.ipynb on Kaggle and copy the "
            "outputs into the models folder (or point AUTIZ_MODELS_DIR at them)."
        )

    try:
        stamps = {SCALER_FILE: _json_stamp(paths[SCALER_FILE])}
        _enforce_smoke_policy(stamps)  # refuse before the heavy imports and loads

        from autiz_model import GEMAPS_DIM, load_autiz_v2, load_scaler
        import joblib

        modules, config = load_autiz_v2(paths[CHECKPOINT_FILE])
        stamps[CHECKPOINT_FILE] = config.get("run_stamp")
        _enforce_smoke_policy(stamps)

        baseline = joblib.load(paths[BASELINE_FILE])
        if not isinstance(baseline, dict) or "ridge" not in baseline:
            raise ValueError(f"{BASELINE_FILE} is not the expected dict with a 'ridge' entry")
        stamps[BASELINE_FILE] = baseline.get("run_stamp")
        smoke = _enforce_smoke_policy(stamps)

        scaler = load_scaler(paths[SCALER_FILE])
        if len(scaler["mean"]) != GEMAPS_DIM or list(baseline["feature_names"]) != list(scaler["feature_names"]):
            raise ValueError("scaler and baseline do not describe the same 62 GeMAPS features")
    except ArtifactError:
        raise
    except (ImportError, KeyError, ValueError, RuntimeError, OSError, EOFError) as e:
        raise ArtifactError(f"Could not load the model artifacts from {d}: {type(e).__name__}: {e}") from e

    _artifacts = {
        "modules": modules,
        "config": config,
        "scaler": scaler,
        "baseline": baseline,
        "smoke": smoke,
        "proxy_stamp": config.get("content_scorer_stamp"),
    }
    print(f"[pipeline] Loaded model artifacts from {d} (smoke={smoke})")
    return _artifacts


def _load_encoders() -> dict:
    global _encoders
    if _encoders is not None:
        return _encoders

    import opensmile
    import torch
    from faster_whisper import WhisperModel
    from transformers import RobertaModel, RobertaTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[pipeline] Loading openSMILE (GeMAPSv01b), Whisper base, RoBERTa-large...")
    _encoders = {
        "smile": opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        ),
        "whisper": WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8"),
        "tokenizer": RobertaTokenizer.from_pretrained("roberta-large"),
        "roberta": RobertaModel.from_pretrained("roberta-large").to(device).eval(),
        "device": device,
    }
    return _encoders


# Audio

def _to_wav(audio_path: str) -> str:
    """Convert any input to 16 kHz mono wav, exactly as the notebook did for training. Caller deletes the result."""
    import imageio_ffmpeg

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        res = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path, "-loglevel", "error"],
            capture_output=True, text=True,
        )
    except BaseException:
        _safe_unlink(wav_path)
        raise
    if res.returncode != 0:
        _safe_unlink(wav_path)
        raise AudioDecodeError(f"Could not decode the audio file: {res.stderr.strip()[:200]}")
    return wav_path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# Scoring

def score_features(cls_embedding, gemaps_raw, art: dict) -> dict:
    """Content score from a [1024] CLS embedding; baseline score and explanation from 62 raw GeMAPS values."""
    import torch
    from autiz_model import apply_scaler

    z = apply_scaler(gemaps_raw, art["scaler"])  # z-score + clip + std floor, same as training
    with torch.no_grad():
        c = art["modules"]["content_branch"](torch.as_tensor(np.asarray(cls_embedding, dtype=np.float32)).reshape(1, -1))
        raw = float(art["modules"]["content_head"](c).reshape(-1)[0].item())

    ridge = art["baseline"]["ridge"]
    baseline_score = float(ridge.predict(z.reshape(1, -1))[0])
    contributions = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1) * z.astype(np.float64)
    names = art["scaler"]["feature_names"]
    top = np.argsort(-np.abs(contributions))[:5]
    return {
        "content_score_raw": raw,
        "content_score": min(1.0, max(0.0, raw)),
        "baseline_score": baseline_score,
        "top_features": [{"feature": names[i], "contribution": float(contributions[i])} for i in top],
    }


def _observations(feature_names, gemaps_raw) -> dict:
    by_name = {n: float(v) for n, v in zip(feature_names, gemaps_raw)}
    return {
        "pitch_variation_stddev_norm": by_name.get(PITCH_FEATURE),
        "mean_unvoiced_segment_sec": by_name.get(UNVOICED_FEATURE),
        "voiced_segments_per_sec": by_name.get(VOICED_RATE_FEATURE),
    }


def _delivery_pattern(obs: dict) -> list:
    lines = []
    if obs["pitch_variation_stddev_norm"] is not None:
        lines.append(f"Pitch variation (normalised std dev, unitless): {obs['pitch_variation_stddev_norm']:.2f}")
    if obs["voiced_segments_per_sec"] is not None:
        lines.append(f"Voiced segments per second: {obs['voiced_segments_per_sec']:.2f}")
    if obs["mean_unvoiced_segment_sec"] is not None:
        lines.append(f"Mean unvoiced segment length: {obs['mean_unvoiced_segment_sec']:.2f} s")
    return lines


def _interpretation(content_score: float, proxy_stamp, transcript: str) -> str:
    excerpt = transcript[:80].rstrip() + ("..." if len(transcript) > 80 else "")
    parts = []
    if proxy_stamp:
        parts.append(f"{proxy_stamp}.")
    parts.append(f"Content score: {content_score:.0%}, computed from the transcript only; the audio is not an input to it.")
    parts.append(f'Transcript excerpt: "{excerpt}".')
    parts.append("The delivery measurements are reported separately and do not change the content score.")
    return " ".join(parts)


def run(audio_path: str, mode: str) -> dict:
    """Full inference. `mode` is accepted for the API but has no effect in this milestone."""
    art = load_artifacts()
    enc = _load_encoders()
    import torch

    wav_path = _to_wav(audio_path)
    try:
        df = enc["smile"].process_file(wav_path)
        feature_names = list(df.columns)
        if feature_names != list(art["scaler"]["feature_names"]):
            raise PipelineError("openSMILE feature names differ from the scaler's; the artifacts do not match this openSMILE setup")
        gemaps_raw = df.values[0].astype(np.float64)

        segments, _ = enc["whisper"].transcribe(wav_path, beam_size=3)
        transcript = " ".join(s.text.strip() for s in segments).strip()
    finally:
        _safe_unlink(wav_path)
    if not transcript:
        raise NoSpeechError("No speech was detected in the audio, so nothing was scored.")

    tok = enc["tokenizer"](transcript, return_tensors="pt", max_length=512, truncation=True).to(enc["device"])
    with torch.no_grad():
        cls = enc["roberta"](**tok).last_hidden_state[:, 0, :].float().cpu().numpy()[0]

    scored = score_features(cls, gemaps_raw, art)
    obs = _observations(feature_names, gemaps_raw)
    proxy_stamp = art["proxy_stamp"]
    return {
        "content_score": scored["content_score"],
        "content_score_raw": scored["content_score_raw"],
        "content_score_source": "proxy" if proxy_stamp else "trained_head",
        "content_scorer_stamp": proxy_stamp,
        "content_label_source": art["config"].get("content_label_source"),
        "intent_label": None,
        "intent_status": "not_trained",
        "prosody_only_baseline_score": scored["baseline_score"],
        "prosody_only_baseline_label": BASELINE_LABEL,
        "transcript": transcript,
        "acoustic_observations": obs,
        "delivery_pattern": _delivery_pattern(obs),
        "explanation": {
            "method": "proxy",
            "explains": "prosody_only_baseline_score",
            "top_features": scored["top_features"],
            "note": (
                "Coefficient x z-score of the prosody-only baseline's 62 inputs. For a linear model this equals the "
                "linear SHAP value relative to the training mean, but SHAP itself is not computed per request. "
                "It explains the baseline only; the content score has no explanation."
            ),
        },
        "interpretation": _interpretation(scored["content_score"], proxy_stamp, transcript),
        "interpretation_source": "template",
        "smoke_artifacts": art["smoke"],
    }


# GET /about

LIMITATIONS = [
    "Only 8 real speakers, all from YouTube, were used for the real-speaker evaluation; results are anecdotal-scale.",
    "The content scorer was trained on ChaLearn First Impressions labels, which are human impressions of video, not ratings of answer quality.",
    "The content score is computed from the transcript only, so it cannot depend on delivery by construction; that invariance is not evidence of training success.",
    "Independence between the content and delivery representations is tested on held-out LibriSpeech speakers; failing to detect a relationship is not proof of independence.",
    "No intent model exists; intent_label is always null.",
    "Transcription uses Whisper base with no fine-tuning, so transcript errors feed into the content score.",
    "This is a research prototype; it makes no claim about detecting or diagnosing autism or any other condition.",
]
CONTENT_LABEL_NOTE = (
    "The content scorer is trained against the ChaLearn First Impressions V2 'interview' label, a human impression "
    "of a video clip. It is a weak stand-in for the quality of the words alone."
)


def get_about() -> dict:
    path = results_dir() / INDEPENDENCE_FILE
    results, note, label_source, results_smoke = None, None, None, False
    if not path.is_file():
        note = f"{INDEPENDENCE_FILE} was not found in {results_dir()}; the training run has not been ingested yet."
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("top level is not a JSON object")
        except (OSError, ValueError) as e:
            note = f"{INDEPENDENCE_FILE} could not be read: {e}"
        else:
            stamp = data.get("run_stamp")
            if stamp and not allow_smoke():
                note = f"{INDEPENDENCE_FILE} is stamped '{stamp}' and was refused. Set ALLOW_SMOKE_ARTIFACTS=1 to show it."
            else:
                results, results_smoke, label_source = data, bool(stamp), data.get("content_label_source")
    return {
        "name": "Autiz (NeuroIntent) research prototype",
        "system_stage": "research_pilot",
        "stores_nothing": True,
        "content_label_source": label_source,
        "content_label_note": CONTENT_LABEL_NOTE,
        "results_available": results is not None,
        "independence_results": results,
        "results_note": note,
        "limitations": LIMITATIONS,
        "smoke_artifacts": smoke_artifacts_flag() or results_smoke,
    }
