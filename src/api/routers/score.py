# APIRouter for defining the /score endpoint
from fastapi import APIRouter, UploadFile, File, HTTPException

# OS and shutil for handling temporary audio files
import os
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pipeline

# Create router instance for the /score endpoint
router = APIRouter()

@router.post("/score")
async def score_audio(
    # Accept an uploaded audio file from the request
    audio: UploadFile = File(...),
    # Accept mode as a query parameter
    mode: str = "universal_fairness"
):
    # Hard rule — /score endpoint never accepts speaker_declared mode
    # Raise HTTP 403 Forbidden if someone tries to pass it
    if mode == "speaker_declared":
        raise HTTPException(
            status_code=403,
            detail="speaker_declared mode is not permitted on the /score endpoint."
        )

    # Save uploaded file temporarily so pipeline can read it
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Call the pipeline in universal_fairness mode only
    result = pipeline.run(audio_path=temp_path, mode="universal_fairness")

    # Delete the temp audio file immediately after feature extraction
    os.remove(temp_path)

    # Return only the fields relevant to platform API consumers
    # /score returns a cleaner subset of the full response
    return {
        "content_quality_score": result["content_quality_score"],
        "prosody_decoupling_applied": result["prosody_decoupling_applied"],
        "confidence_bound": result["confidence_bound"],
        "system_stage": result["system_stage"],
        "candidate_disclosure_required": result["candidate_disclosure_required"],
        "audit_trail": result["audit_trail"]
    }
