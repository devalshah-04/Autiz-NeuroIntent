# TestClient lets us test FastAPI endpoints without running a real server
from fastapi.testclient import TestClient

# Import our main app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main import app

# BytesIO lets us create a fake audio file in memory for testing
from io import BytesIO

# Create test client
client = TestClient(app)

# ── Helper ────────────────────────────────────────────────────────────────────

def fake_wav() -> BytesIO:
    """Create a minimal fake WAV file in memory for testing."""
    import wave, struct
    buffer = BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<h", 0) * 16000)
    buffer.seek(0)
    return buffer

# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check():
    """Health endpoint should return ok and research_pilot."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["system_stage"] == "research_pilot"

# ── /analyze tests ────────────────────────────────────────────────────────────

def test_analyze_returns_correct_schema():
    """Analyze endpoint should return all required fields."""
    response = client.post(
        "/analyze",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "universal_fairness"}
    )
    assert response.status_code == 200
    data = response.json()

    # Check all required top level fields exist
    assert "content_quality_score" in data
    assert "prosody_decoupling_applied" in data
    assert "confidence_bound" in data
    assert "acoustic_observations" in data
    assert "audit_trail" in data
    assert "session" in data

def test_analyze_hardcoded_fields():
    """system_stage and candidate_disclosure_required must always be hardcoded."""
    response = client.post(
        "/analyze",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "universal_fairness"}
    )
    data = response.json()

    # These two fields are hardcoded — never change
    assert data["session"]["system_stage"] == "research_pilot"
    assert data["session"]["candidate_disclosure_required"] is True

def test_analyze_invalid_mode():
    """Analyze endpoint should reject invalid mode with 400."""
    response = client.post(
        "/analyze",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "invalid_mode"}
    )
    assert response.status_code == 400

def test_analyze_speaker_id_anonymized():
    """Speaker ID must be 12 characters long — sha256 truncated."""
    response = client.post(
        "/analyze",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "universal_fairness"}
    )
    data = response.json()
    assert len(data["session"]["speaker_id"]) == 12

# ── /score tests ──────────────────────────────────────────────────────────────

def test_score_returns_correct_schema():
    """Score endpoint should return required fields."""
    response = client.post(
        "/score",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "universal_fairness"}
    )
    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "content_quality_score" in data
    assert "prosody_decoupling_applied" in data
    assert "audit_trail" in data

def test_score_rejects_speaker_declared():
    """Hard rule — /score must return 403 for speaker_declared mode."""
    response = client.post(
        "/score",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "speaker_declared"}
    )
    # This is the most important hard rule test
    assert response.status_code == 403

def test_score_hardcoded_fields():
    """system_stage and candidate_disclosure_required must always be hardcoded."""
    response = client.post(
        "/score",
        files={"audio": ("test.wav", fake_wav(), "audio/wav")},
        params={"mode": "universal_fairness"}
    )
    data = response.json()
    assert data["system_stage"] == "research_pilot"
    assert data["candidate_disclosure_required"] is True
