import os
import sys

import pytest

API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (API_DIR, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Every test starts with no artifact/results/smoke env vars and an empty pipeline cache."""
    for var in ("AUTIZ_MODELS_DIR", "AUTIZ_RESULTS_DIR", "ALLOW_SMOKE_ARTIFACTS"):
        monkeypatch.delenv(var, raising=False)
    pipeline.reset_cache()
    yield
    pipeline.reset_cache()
