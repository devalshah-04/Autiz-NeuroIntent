"""API tests. pipeline.run is replaced by a stub, so no RoBERTa/Whisper/torch is loaded and nothing needs the network."""

import json
import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pipeline
from main import app
from mock_pipeline import stub_result
from schemas import AnalyzeResponse

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[3]


def fake_wav() -> BytesIO:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 1600)
    buffer.seek(0)
    return buffer


def post(path, mode="universal_fairness", filename="test.wav"):
    data = {} if mode is None else {"mode": mode}
    return client.post(path, files={"audio": (filename, fake_wav(), "audio/wav")}, data=data)


@pytest.fixture
def stub_run(monkeypatch):
    """Replace pipeline.run; records the temp path, whether it existed during the call, and the mode."""
    calls = []

    def run(audio_path, mode):
        calls.append({"path": audio_path, "existed": os.path.exists(audio_path), "mode": mode})
        return stub_result()

    monkeypatch.setattr(pipeline, "run", run)
    return calls


# Health, routes

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "system_stage": "research_pilot", "smoke_artifacts": False}


def test_no_stream_route_and_no_delete_endpoint():
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/stream" not in paths
    assert client.get("/stream").status_code == 404
    methods = {m for r in app.routes for m in (getattr(r, "methods", None) or ())}
    assert "DELETE" not in methods


def test_cors_preflight_allows_local_dev_origin():
    r = client.options(
        "/analyze",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


# /analyze

def test_analyze_values_and_schema(stub_run):
    r = post("/analyze")
    assert r.status_code == 200
    data = r.json()
    AnalyzeResponse.model_validate(data)
    assert data["content_score"] == 0.63
    assert data["content_score_raw"] == 0.63
    assert data["content_score_source"] == "trained_head"
    assert data["content_scorer_stamp"] is None
    assert data["intent_label"] is None
    assert data["intent_status"] == "not_trained"
    assert data["prosody_only_baseline_score"] == -0.21
    assert data["prosody_only_baseline_label"] == "illustrative comparison model, not an evaluator or vendor tool"
    assert data["interpretation_source"] == "template"
    assert data["explanation"]["method"] in ("shap", "proxy")
    assert data["mode"] == "universal_fairness"
    assert data["mode_effect"] == "none in this milestone"
    assert data["smoke_artifacts"] is False
    assert data["session"]["system_stage"] == "research_pilot"
    assert data["session"]["candidate_disclosure_required"] is True
    assert len(data["session"]["speaker_id"]) == 12


def test_analyze_has_none_of_the_removed_fields(stub_run):
    data = post("/analyze").json()
    removed = {
        "score_without_system", "score_with_system", "content_quality_score", "intent_confidence", "confidence_bound",
        "prosody_decoupling_applied", "audit_trail", "decoupling_verified", "audio_storage_consent",
    }
    assert not removed & set(data)
    assert "audio_storage_consent" not in data["session"]


@pytest.mark.parametrize("mode", ["universal_fairness", "speaker_declared"])
def test_analyze_mode_is_read_from_form_and_echoed(stub_run, mode):
    data = post("/analyze", mode=mode).json()
    assert data["mode"] == mode
    assert data["mode_effect"] == "none in this milestone"
    assert stub_run[0]["mode"] == mode


def test_analyze_mode_in_query_string_is_not_accepted(stub_run):
    r = client.post("/analyze", files={"audio": ("t.wav", fake_wav(), "audio/wav")}, params={"mode": "universal_fairness"})
    assert r.status_code == 422
    assert stub_run == []


def test_analyze_invalid_mode_is_400(stub_run):
    assert post("/analyze", mode="invalid_mode").status_code == 400
    assert stub_run == []


def test_analyze_missing_audio_is_422(stub_run):
    assert client.post("/analyze", data={"mode": "universal_fairness"}).status_code == 422


def test_analyze_accepts_scores_outside_unit_interval(monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda audio_path, mode: stub_result(content_score=1.0, content_score_raw=1.73))
    data = post("/analyze").json()
    assert (data["content_score"], data["content_score_raw"]) == (1.0, 1.73)
    monkeypatch.setattr(pipeline, "run", lambda audio_path, mode: stub_result(content_score=0.0, content_score_raw=-0.4))
    data = post("/analyze").json()
    assert (data["content_score"], data["content_score_raw"]) == (0.0, -0.4)


def test_analyze_proxy_stamp_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        pipeline, "run",
        lambda audio_path, mode: stub_result(content_score_source="proxy", content_scorer_stamp=pipeline.PROXY_STAMP),
    )
    data = post("/analyze").json()
    assert data["content_score_source"] == "proxy"
    assert data["content_scorer_stamp"] == "PROXY, not a trained scorer"


def test_smoke_flag_is_passed_through(monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda audio_path, mode: stub_result(smoke_artifacts=True))
    assert post("/analyze").json()["smoke_artifacts"] is True
    assert post("/score").json()["smoke_artifacts"] is True


# /score

def test_score_values_and_exact_fields(stub_run):
    r = post("/score")
    assert r.status_code == 200
    assert r.json() == {
        "content_score": 0.63,
        "content_score_raw": 0.63,
        "content_score_source": "trained_head",
        "content_scorer_stamp": None,
        "mode": "universal_fairness",
        "mode_effect": "none in this milestone",
        "system_stage": "research_pilot",
        "candidate_disclosure_required": True,
        "smoke_artifacts": False,
    }


def test_score_rejects_speaker_declared(stub_run):
    assert post("/score", mode="speaker_declared").status_code == 403
    assert stub_run == []


def test_score_invalid_mode_is_400_and_missing_mode_is_422(stub_run):
    assert post("/score", mode="nonsense").status_code == 400
    assert post("/score", mode=None).status_code == 422


# Temp-file handling

def test_audio_exists_during_run_and_is_deleted_afterwards(stub_run):
    assert post("/analyze").status_code == 200
    assert post("/score").status_code == 200
    assert len(stub_run) == 2
    for call in stub_run:
        assert call["existed"] is True
        assert not os.path.exists(call["path"])


def test_uploaded_filename_never_used_for_the_temp_path(stub_run):
    post("/analyze", filename="../../evil name.wav")
    path = Path(stub_run[0]["path"])
    assert path.parent == Path(tempfile.gettempdir())
    assert "evil" not in path.name and path.name.startswith("autiz_") and path.suffix == ".wav"


def test_audio_deleted_when_pipeline_raises_known_error(monkeypatch):
    seen = []

    def run(audio_path, mode):
        seen.append(audio_path)
        raise pipeline.NoSpeechError("No speech was detected in the audio, so nothing was scored.")

    monkeypatch.setattr(pipeline, "run", run)
    r = post("/analyze")
    assert r.status_code == 422
    assert "No speech" in r.json()["detail"]
    assert not os.path.exists(seen[0])


def test_audio_deleted_when_pipeline_crashes(monkeypatch):
    seen = []

    def run(audio_path, mode):
        seen.append(audio_path)
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run", run)
    crash_client = TestClient(app, raise_server_exceptions=False)
    r = crash_client.post("/score", files={"audio": ("t.wav", fake_wav(), "audio/wav")}, data={"mode": "universal_fairness"})
    assert r.status_code == 500
    assert seen and not os.path.exists(seen[0])


# Artifact errors without torch or any model files

def test_missing_artifacts_give_503_naming_the_files(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(tmp_path))
    r = post("/analyze")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "autiz_v2.pt" in detail and "gemaps_scaler.json" in detail and "prosody_baseline.joblib" in detail


def test_stamped_scaler_is_refused_with_503_unless_allowed(monkeypatch, tmp_path):
    for name in ("autiz_v2.pt", "prosody_baseline.joblib"):
        (tmp_path / name).write_bytes(b"not read: the scaler stamp is checked first")
    (tmp_path / "gemaps_scaler.json").write_text(json.dumps({"run_stamp": pipeline.SMOKE_STAMP}), encoding="utf-8")
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(tmp_path))
    r = post("/analyze")
    assert r.status_code == 503
    assert "ALLOW_SMOKE_ARTIFACTS=1" in r.json()["detail"] and "gemaps_scaler.json" in r.json()["detail"]


# /about

def test_about_without_results_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTIZ_RESULTS_DIR", str(tmp_path))
    r = client.get("/about")
    assert r.status_code == 200
    data = r.json()
    assert data["results_available"] is False
    assert data["independence_results"] is None
    assert "eval_results_independence.json" in data["results_note"]
    assert data["stores_nothing"] is True
    assert data["content_label_source"] is None
    assert data["limitations"] and data["smoke_artifacts"] is False


def test_about_with_results_file(monkeypatch, tmp_path):
    results = {"content_label_source": "chalearn", "decoupling_verified": True, "results": {"original_clips": {"x": 1}}}
    (tmp_path / "eval_results_independence.json").write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setenv("AUTIZ_RESULTS_DIR", str(tmp_path))
    data = client.get("/about").json()
    assert data["results_available"] is True
    assert data["independence_results"] == results
    assert data["content_label_source"] == "chalearn"
    assert data["results_note"] is None
    assert data["stores_nothing"] is True
    assert data["smoke_artifacts"] is False


def test_about_refuses_smoke_results_unless_allowed(monkeypatch, tmp_path):
    results = {"run_stamp": pipeline.SMOKE_STAMP, "content_label_source": "chalearn"}
    (tmp_path / "eval_results_independence.json").write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setenv("AUTIZ_RESULTS_DIR", str(tmp_path))
    data = client.get("/about").json()
    assert data["results_available"] is False and data["independence_results"] is None
    assert "refused" in data["results_note"] and data["smoke_artifacts"] is False

    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    data = client.get("/about").json()
    assert data["results_available"] is True and data["smoke_artifacts"] is True
    assert data["independence_results"]["run_stamp"] == pipeline.SMOKE_STAMP


def test_about_with_unreadable_results_file_is_not_an_error(monkeypatch, tmp_path):
    (tmp_path / "eval_results_independence.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("AUTIZ_RESULTS_DIR", str(tmp_path))
    data = client.get("/about").json()
    assert data["results_available"] is False and "could not be read" in data["results_note"]


def test_health_reports_smoke_only_when_allowed(monkeypatch, tmp_path):
    (tmp_path / "gemaps_scaler.json").write_text(json.dumps({"run_stamp": pipeline.SMOKE_STAMP}), encoding="utf-8")
    monkeypatch.setenv("AUTIZ_MODELS_DIR", str(tmp_path))
    assert client.get("/health").json()["smoke_artifacts"] is False
    monkeypatch.setenv("ALLOW_SMOKE_ARTIFACTS", "1")
    assert client.get("/health").json()["smoke_artifacts"] is True


# Contract and small helpers

def test_example_response_matches_the_schema():
    example = json.loads((REPO_ROOT / "docs" / "api_example_response.json").read_text(encoding="utf-8"))
    example.pop("_comment")
    parsed = AnalyzeResponse.model_validate(example)
    assert set(parsed.model_dump()) == set(example)


def test_resolve_dir_falls_back_to_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert pipeline._resolve_dir("src/api") == pipeline._REPO_ROOT / "src" / "api"
    assert pipeline._resolve_dir(str(tmp_path / "nowhere")) == tmp_path / "nowhere"


def test_delivery_text_handles_missing_features():
    obs = pipeline._observations(["MeanUnvoicedSegmentLength"], [0.25])
    assert obs == {"pitch_variation_stddev_norm": None, "mean_unvoiced_segment_sec": 0.25, "voiced_segments_per_sec": None}
    assert pipeline._delivery_pattern(obs) == ["Mean unvoiced segment length: 0.25 s"]


def test_interpretation_is_a_neutral_template_and_shows_the_proxy_stamp():
    text = pipeline._interpretation(0.5, pipeline.PROXY_STAMP, "word " * 40)
    assert text.startswith("PROXY, not a trained scorer.")
    assert "Content score: 50%" in text and "..." in text
    assert "traditional" not in text and "undervalue" not in text
