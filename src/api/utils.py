import hashlib
import os
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

# Only used to give ffmpeg a hint about the container; never taken from the uploader's filename otherwise.
_ALLOWED_SUFFIXES = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4", ".flac", ".aac"}


def generate_speaker_id() -> str:
    """Random per-request id: sha256(timestamp + random salt)[:12]. Not linked to any person."""
    raw = datetime.now(timezone.utc).isoformat() + secrets.token_hex(16)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@contextmanager
def temporary_upload(upload):
    """Save an UploadFile to a temp file with a generated name; always delete it on exit, success or failure."""
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".bin"
    fd, path = tempfile.mkstemp(prefix="autiz_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def build_session_metadata() -> dict:
    return {
        "speaker_id": generate_speaker_id(),
        "system_stage": "research_pilot",
        "candidate_disclosure_required": True,
    }
