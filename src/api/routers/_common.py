import os
import sys

from fastapi import HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pipeline
from utils import temporary_upload

MODE_EFFECT = "none in this milestone"
VALID_MODES = ("universal_fairness", "speaker_declared")


def run_pipeline(upload, mode: str) -> dict:
    """Run the pipeline on an upload; the temp audio file is always deleted, and pipeline errors become HTTP errors."""
    try:
        with temporary_upload(upload) as path:
            return pipeline.run(audio_path=path, mode=mode)
    except pipeline.PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
