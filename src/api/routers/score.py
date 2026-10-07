from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from routers._common import MODE_EFFECT, VALID_MODES, run_pipeline
from schemas import ScoreResponse

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score_audio(audio: UploadFile = File(...), mode: str = Form(...)):
    if mode == "speaker_declared":
        raise HTTPException(status_code=403, detail="speaker_declared mode is not permitted on the /score endpoint.")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode. Use universal_fairness.")
    result = run_pipeline(audio, mode)
    return {
        "content_score": result["content_score"],
        "content_score_raw": result["content_score_raw"],
        "content_score_source": result["content_score_source"],
        "content_scorer_stamp": result["content_scorer_stamp"],
        "mode": mode,
        "mode_effect": MODE_EFFECT,
        "system_stage": "research_pilot",
        "candidate_disclosure_required": True,
        "smoke_artifacts": result["smoke_artifacts"],
    }
