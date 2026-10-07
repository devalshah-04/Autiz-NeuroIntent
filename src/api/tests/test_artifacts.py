"""Artifact loading and scoring tests. Small fixtures are built inside the tests; no real checkpoint is needed.
Skipped as a module when torch, scikit-learn or joblib is missing."""

import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

import pipeline  # noqa: E402
from autiz_model import REQUIRED_CHECKPOINT_KEYS, apply_scaler, build_models, load_autiz_v2  # noqa: E402
from main import app  # noqa: E402
from schemas import AnalyzeResponse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_DIR = REPO_ROOT / "models" / "smoke"
FEATURE_NAMES = [pipeline.PITCH_FEATURE, pipeline.UNVOICED_FEATURE, pipeline.VOICED_RATE_FEATURE] + [
    f"feat_{i}" for i in range(3, 62)
]
FILE_OF = {
    "checkpoint": pipeline.CHECKPOINT_FILE,
    "scaler": pipeline.SCALER_FILE,
    "baseline": pipeline.BASELINE_FILE,
}


def make_artifacts(folder, stamped=(), proxy=False, head_bias=0.3, zero_head=False, drop_key=None, baseline_names=None):
    """Write a tiny but real autiz_v2.pt, gemaps_scaler.json and prosody_baseline.joblib into `folder`."""
    folder = Path(folder)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mods = build_models()
    with torch.no_grad():
        if zero_head:
            mods["content_head"].weight.zero_()
        mods["content_head"].bias.fill_(head_bias)

    config = {
        "dims": {"content_in": 1024, "hidden": 256, "gemaps": 62},
        "seed": 0,
        "content_label_source": "proxy" if proxy else "chalearn",
        "content_scorer_stamp": pipeline.PROXY_STAMP if proxy else None,
        "date": "2026-10-07",
        "z_clip": 10.0,
        "roberta": "roberta-large",
        "roberta_frozen": True,
    }
    if "checkpoint" in stamped:
        config["run_stamp"] = pipeline.SMOKE_STAMP
    ckpt = {name: mods[name].state_dict() for name in ("content_branch", "content_head", "prosody_branch", "recon_head")}
    ckpt["config"] = config
    if drop_key:
        del ckpt[drop_key]
    torch.save(ckpt, folder / pipeline.CHECKPOINT_FILE)

    mean = rng.normal(size=62)
    std = rng.uniform(0.5, 2.0, size=62)
    scaler = {"feature_names": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist(), "z_clip": 10.0}
    if "scaler" in stamped:
        scaler = {"run_stamp": pipeline.SMOKE_STAMP, **scaler}
    (folder / pipeline.SCALER_FILE).write_text(json.dumps(scaler), encoding="utf-8")

    ridge = Ridge().fit(rng.normal(size=(50, 62)), rng.normal(size=50))
    ridge.coef_, ridge.intercept_ = np.ones(62), 0.0  # baseline score == sum of the z-scored inputs
    payload = {"ridge": ridge, "feature_names": baseline_names or FEATURE_NAMES, "label": pipeline.BASELINE_LABEL}
    if "baseline" in stamped:
        payload["run_stamp"] = pipeline.SMOKE_STAMP
    joblib.dump(payload, folder / pipeline.BASELINE_FILE)
    return {"mods": mods, "mean": mean, "std": std}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(tmp_path))
    return tmp_path


# Checkpoint format (decision 1: C is the ContentBranch output; checkpoint holds these exact keys)

def test_checkpoint_has_exactly_the_documented_keys_and_shapes(models_dir):
    make_artifacts(models_dir)
    ck = torch.load(models_dir / pipeline.CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    assert set(ck) == {"content_branch", "content_head", "prosody_branch", "recon_head", "config"}
    assert set(ck) == set(REQUIRED_CHECKPOINT_KEYS)
    fresh = build_models()
    for name, module in fresh.items():
        assert set(ck[name]) == set(module.state_dict()), name
    assert tuple(ck["content_branch"]["layers.0.weight"].shape) == (256, 1024)
    assert tuple(ck["prosody_branch"]["layers.0.weight"].shape) == (256, 62)
    assert tuple(ck["content_head"]["weight"].shape) == (1, 256)
    assert tuple(ck["recon_head"]["layers.2.weight"].shape) == (62, 256)


def test_load_autiz_v2_rejects_a_checkpoint_with_a_missing_key(models_dir):
    make_artifacts(models_dir, drop_key="content_head")
    with pytest.raises(KeyError, match="content_head"):
        load_autiz_v2(models_dir / pipeline.CHECKPOINT_FILE)
    with pytest.raises(pipeline.ArtifactError, match="missing checkpoint keys"):
        pipeline.load_artifacts()


# Loading

def test_load_artifacts_gives_frozen_eval_modules_without_recon_head(models_dir):
    make_artifacts(models_dir)
    art = pipeline.load_artifacts()
    assert set(art["modules"]) == {"content_branch", "content_head", "prosody_branch"}
    for module in art["modules"].values():
        assert not module.training
        assert not any(p.requires_grad for p in module.parameters())
    assert art["smoke"] is False and art["proxy_stamp"] is None
    assert pipeline.load_artifacts() is art
    pipeline.reset_cache()
    assert pipeline.load_artifacts() is not art


@pytest.mark.parametrize("missing", ["checkpoint", "scaler", "baseline"])
def test_a_missing_file_raises_a_clear_error(models_dir, missing):
    make_artifacts(models_dir)
    (models_dir / FILE_OF[missing]).unlink()
    with pytest.raises(pipeline.MissingArtifactError, match=FILE_OF[missing]):
        pipeline.load_artifacts()


def test_scaler_and_baseline_must_describe_the_same_features(models_dir):
    make_artifacts(models_dir, baseline_names=list(reversed(FEATURE_NAMES)))
    with pytest.raises(pipeline.ArtifactError, match="same 62 GeMAPS features"):
        pipeline.load_artifacts()


def test_proxy_checkpoint_is_stamped(models_dir):
    make_artifacts(models_dir, proxy=True)
    assert pipeline.load_artifacts()["proxy_stamp"] == "PROXY, not a trained scorer"


# Smoke-stamp policy

@pytest.mark.parametrize("stamped_file", ["scaler", "checkpoint", "baseline"])
def test_any_stamped_file_makes_the_set_smoke_and_is_refused_by_default(models_dir, monkeypatch, stamped_file):
    make_artifacts(models_dir, stamped=(stamped_file,))
    with pytest.raises(pipeline.SmokeArtifactError, match=FILE_OF[stamped_file]):
        pipeline.load_artifacts()
    pipeline.reset_cache()
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    assert pipeline.load_artifacts()["smoke"] is True


def test_allow_flag_alone_does_not_mark_real_artifacts_as_smoke(models_dir, monkeypatch):
    make_artifacts(models_dir)
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    assert pipeline.load_artifacts()["smoke"] is False


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_only_the_value_1_allows_smoke(models_dir, monkeypatch, value):
    make_artifacts(models_dir, stamped=("checkpoint",))
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", value)
    with pytest.raises(pipeline.SmokeArtifactError):
        pipeline.load_artifacts()


# Scoring

def test_content_and_baseline_scores_match_an_independent_reference(models_dir):
    made = make_artifacts(models_dir)
    art = pipeline.load_artifacts()
    cls = np.random.default_rng(1).normal(size=1024).astype(np.float32)
    raw = made["mean"] + 0.5 * made["std"]
    out = pipeline.score_features(cls, raw, art)
    with torch.no_grad():
        expected = made["mods"]["content_head"](made["mods"]["content_branch"](torch.as_tensor(cls).reshape(1, -1))).item()
    assert out["content_score_raw"] == pytest.approx(expected, abs=1e-5)
    assert out["content_score"] == pytest.approx(min(1.0, max(0.0, expected)), abs=1e-5)
    assert out["baseline_score"] == pytest.approx(0.5 * 62, abs=1e-4)


@pytest.mark.parametrize("bias,clipped", [(1.7, 1.0), (-0.4, 0.0), (0.3, 0.3)])
def test_content_score_is_clipped_but_raw_is_kept(models_dir, bias, clipped):
    made = make_artifacts(models_dir, zero_head=True, head_bias=bias)
    out = pipeline.score_features(np.ones(1024), made["mean"], pipeline.load_artifacts())
    assert out["content_score_raw"] == pytest.approx(bias, abs=1e-6)
    assert out["content_score"] == pytest.approx(clipped, abs=1e-6)


def test_serving_applies_the_scaler_z_clip(models_dir):
    made = make_artifacts(models_dir)
    art = pipeline.load_artifacts()
    raw = made["mean"].copy()
    raw[5] += 1000 * made["std"][5]  # z = 1000, must be clipped to 10
    raw[7] -= 3 * made["std"][7]  # z = -3
    out = pipeline.score_features(np.zeros(1024), raw, art)
    assert out["baseline_score"] == pytest.approx(10.0 - 3.0, abs=1e-4)
    top = out["top_features"]
    assert len(top) == 5
    assert (top[0]["feature"], top[0]["contribution"]) == ("feat_5", pytest.approx(10.0))
    assert (top[1]["feature"], top[1]["contribution"]) == ("feat_7", pytest.approx(-3.0))
    # the helper the pipeline uses is the shared one, and it clips
    assert apply_scaler(raw, art["scaler"])[5] == pytest.approx(10.0)


# Full run() through the HTTP layer with fake encoders (no RoBERTa, Whisper, openSMILE or ffmpeg)

class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeWhisper:
    def __init__(self, text, wav_log):
        self.text, self.wav_log = text, wav_log

    def transcribe(self, path, beam_size):
        assert os.path.exists(path)
        segments = [types.SimpleNamespace(text=f"  {self.text}  ")] if self.text else []
        return segments, None


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the heavy parts of pipeline.run; returns a dict to tweak (transcript, feature names) and the temp wavs made."""
    state = {"text": "hello there", "names": FEATURE_NAMES, "wavs": []}

    def to_wav(audio_path):
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        state["wavs"].append(path)
        return path

    def process_file(path):
        raw = np.zeros(62)
        raw[0], raw[1], raw[2] = 0.21, 0.18, 2.9
        return types.SimpleNamespace(columns=state["names"], values=np.array([raw]))

    encoders = {
        "smile": types.SimpleNamespace(process_file=process_file),
        "whisper": types.SimpleNamespace(transcribe=lambda path, beam_size: FakeWhisper(state["text"], None).transcribe(path, beam_size)),
        "tokenizer": lambda text, **kw: FakeEncoding(input_ids=torch.zeros(1, 3, dtype=torch.long)),
        "roberta": lambda **kw: types.SimpleNamespace(last_hidden_state=torch.ones(1, 3, 1024)),
        "device": "cpu",
    }
    monkeypatch.setattr(pipeline, "_to_wav", to_wav)
    monkeypatch.setattr(pipeline, "_load_encoders", lambda: encoders)
    return state


def post_audio(client, mode="universal_fairness", path="/analyze"):
    return client.post(path, files={"audio": ("a.wav", b"RIFF-not-read", "audio/wav")}, data={"mode": mode})


def test_analyze_end_to_end_with_real_artifacts_and_fake_encoders(models_dir, fake_run):
    made = make_artifacts(models_dir)
    r = post_audio(TestClient(app))
    assert r.status_code == 200, r.text
    data = r.json()
    AnalyzeResponse.model_validate(data)
    with torch.no_grad():
        expected_raw = made["mods"]["content_head"](made["mods"]["content_branch"](torch.ones(1, 1024))).item()
    assert data["transcript"] == "hello there"
    assert data["content_score_raw"] == pytest.approx(expected_raw, abs=1e-5)
    assert data["content_score"] == pytest.approx(min(1.0, max(0.0, expected_raw)), abs=1e-5)
    assert data["content_score_source"] == "trained_head" and data["content_scorer_stamp"] is None
    assert data["content_label_source"] == "chalearn"
    assert data["intent_label"] is None and data["intent_status"] == "not_trained"
    assert data["acoustic_observations"] == {
        "pitch_variation_stddev_norm": 0.21, "mean_unvoiced_segment_sec": 0.18, "voiced_segments_per_sec": 2.9,
    }
    assert data["delivery_pattern"][0] == "Pitch variation (normalised std dev, unitless): 0.21"
    assert data["explanation"]["method"] == "proxy" and len(data["explanation"]["top_features"]) == 5
    assert data["interpretation_source"] == "template"
    assert data["smoke_artifacts"] is False
    assert len(fake_run["wavs"]) == 1 and not os.path.exists(fake_run["wavs"][0])


def test_score_end_to_end_reports_proxy_scorer(models_dir, fake_run):
    make_artifacts(models_dir, proxy=True)
    data = post_audio(TestClient(app), path="/score").json()
    assert data["content_score_source"] == "proxy"
    assert data["content_scorer_stamp"] == "PROXY, not a trained scorer"


def test_smoke_artifacts_refused_then_allowed_end_to_end(models_dir, fake_run, monkeypatch):
    make_artifacts(models_dir, stamped=("checkpoint", "scaler", "baseline"))
    client = TestClient(app)
    r = post_audio(client)
    assert r.status_code == 503 and "ALLOW_SMOKE_ARTIFACTS=1" in r.json()["detail"]
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    pipeline.reset_cache()
    for path in ("/analyze", "/score"):
        data = post_audio(client, path=path).json()
        assert data["smoke_artifacts"] is True


def test_no_speech_is_422_and_temp_wav_is_deleted(models_dir, fake_run):
    make_artifacts(models_dir)
    fake_run["text"] = ""
    r = post_audio(TestClient(app))
    assert r.status_code == 422 and "No speech" in r.json()["detail"]
    assert len(fake_run["wavs"]) == 1 and not os.path.exists(fake_run["wavs"][0])


def test_feature_name_mismatch_is_reported_and_temp_wav_is_deleted(models_dir, fake_run):
    make_artifacts(models_dir)
    fake_run["names"] = [f"other_{i}" for i in range(62)]
    r = post_audio(TestClient(app))
    assert r.status_code == 500 and "feature names differ" in r.json()["detail"]
    assert not os.path.exists(fake_run["wavs"][0])


# Real smoke artifact set from Kaggle (skipped until files are placed in models/smoke/)

real_smoke = pytest.mark.skipif(
    not all((SMOKE_DIR / f).is_file() for f in FILE_OF.values()),
    reason="models/smoke/ does not hold a real smoke artifact set (autiz_v2.pt, gemaps_scaler.json, prosody_baseline.joblib)",
)


@real_smoke
def test_real_smoke_checkpoint_loads_and_runs_with_the_allow_flag(monkeypatch):
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(SMOKE_DIR))
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    art = pipeline.load_artifacts()
    assert art["smoke"] is True

    rng = np.random.default_rng(0)
    raw62 = rng.normal(size=62)
    z = apply_scaler(raw62, art["scaler"])
    cls = rng.normal(size=1024).astype(np.float32)
    with torch.no_grad():
        d = art["modules"]["prosody_branch"](torch.as_tensor(z).reshape(1, -1))
        c = art["modules"]["content_branch"](torch.as_tensor(cls).reshape(1, -1))
        score = art["modules"]["content_head"](c)
    assert (d.shape[-1], c.shape[-1], score.shape[-1]) == (256, 256, 1)
    for t in (torch.as_tensor(z), d, c, score):
        assert torch.isfinite(t).all()

    out = pipeline.score_features(cls, raw62, art)
    assert np.isfinite([out["content_score"], out["content_score_raw"], out["baseline_score"]]).all()


@real_smoke
def test_real_smoke_checkpoint_is_refused_without_the_allow_flag(monkeypatch):
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(SMOKE_DIR))
    monkeypatch.delenv("ALLOW_SMOKE_ARTIFACTS", raising=False)
    with pytest.raises(pipeline.SmokeArtifactError):
        pipeline.load_artifacts()
