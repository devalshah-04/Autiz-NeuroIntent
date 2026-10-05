# APIRouter for defining the /analyze endpoint
from fastapi import APIRouter, UploadFile, File, HTTPException

# OS and shutil for handling temporary audio files
import os
import shutil

# Import our utility functions for anonymization and audio consent
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pipeline
from utils import delete_audio_if_no_consent, build_session_metadata

# Create router instance for the /analyze endpoint
router = APIRouter()

@router.post("/analyze")
async def analyze_audio(
    # Accept an uploaded audio file from the request
    audio: UploadFile = File(...),
    # Accept mode as a query parameter — defaults to universal_fairness
    mode: str = "universal_fairness",
    # Accept consent flag — defaults to False (no audio storage)
    audio_storage_consent: bool = False
):
    # Validate mode — only two values are allowed
    if mode not in ["speaker_declared", "universal_fairness"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use speaker_declared or universal_fairness."
        )

    # Save uploaded file temporarily so pipeline can read it
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Call the pipeline with the temp file path and mode
    result = pipeline.run(audio_path=temp_path, mode=mode)

    # Delete audio based on consent — hard rule enforcement
    delete_audio_if_no_consent(temp_path, consent=audio_storage_consent)

    # Build session metadata with anonymized speaker ID
    session = build_session_metadata(mode=mode, consent=audio_storage_consent)

    # Attach session metadata to the response
    result["session"] = session

    # Return the full response dict
    return result
