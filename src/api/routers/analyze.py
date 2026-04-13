# APIRouter lets us define endpoints in separate files and register them in main.py
from fastapi import APIRouter, UploadFile, File, HTTPException

# OS and shutil for handling temporary audio files
import os
import shutil

# Import our mock pipeline — replace with real pipeline later
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mock_pipeline

# Create router instance for the /analyze endpoint
router = APIRouter()

@router.post("/analyze")
async def analyze_audio(
    # Accept an uploaded audio file from the request
    audio: UploadFile = File(...),
    # Accept mode as a query parameter — defaults to universal_fairness
    mode: str = "universal_fairness"
):
    # Validate mode — only two values are allowed
    if mode not in ["speaker_declared", "universal_fairness"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use speaker_declared or universal_fairness.")

    # Save uploaded file temporarily so pipeline can read it
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Call the pipeline with the temp file path and mode
    result = mock_pipeline.run(audio_path=temp_path, mode=mode)

    # Delete the temp audio file immediately after feature extraction
    # Raw audio is never stored unless speaker explicitly consented
    os.remove(temp_path)

    # Return the full response dict from the pipeline
    return result
