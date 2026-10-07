from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from routers._common import MODE_EFFECT, VALID_MODES, run_pipeline
from schemas import AnalyzeResponse
from utils import build_session_metadata

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_audio(audio: UploadFile = File(...), mode: str = Form(...)):
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode. Use speaker_declared or universal_fairness.")
    result = run_pipeline(audio, mode)
    return {**result, "mode": mode, "mode_effect": MODE_EFFECT, "session": build_session_metadata()}
