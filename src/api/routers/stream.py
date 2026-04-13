# APIRouter and WebSocket classes for real-time streaming
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Torch for running Silero VAD model
import torch

# OS for temp file handling
import os

# Struct for converting raw bytes to audio samples
import struct

# Import mock pipeline and utility functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mock_pipeline
from utils import delete_audio_if_no_consent, generate_speaker_id

# Create router instance for the /stream WebSocket endpoint
router = APIRouter()

# Load Silero VAD model once at startup — not on every request
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

# Extract the get_speech_timestamps utility from Silero
(get_speech_timestamps, _, read_audio, *_) = vad_utils

def is_speech(audio_bytes: bytes, sample_rate: int = 16000) -> bool:
    """
    Check if an audio chunk contains speech using Silero VAD.
    Returns True if speech detected, False if silence.
    """
    # Convert raw bytes to 16-bit integer samples
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", audio_bytes)

    # Normalize to float32 tensor between -1 and 1
    audio_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0

    # Run Silero VAD
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate
    )

    # Return True if any speech found
    return len(speech_timestamps) > 0

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):

    # Accept the incoming WebSocket connection
    await websocket.accept()

    # Generate one anonymized speaker ID for this entire session
    # Same ID used for all chunks in this session
    speaker_id = generate_speaker_id()
    print(f"[stream] Client connected — speaker_id: {speaker_id}")

    try:
        # Keep listening for incoming audio chunks
        while True:

            # Receive raw bytes (30ms audio chunk)
            data = await websocket.receive_bytes()

            # Check for speech using Silero VAD
            # Skip silent chunks entirely — saves compute
            if not is_speech(data):
                continue

            # Speech detected — save chunk temporarily
            temp_path = "/tmp/stream_chunk.wav"
            with open(temp_path, "wb") as f:
                f.write(data)

            # Run pipeline in universal_fairness mode
            result = mock_pipeline.run(audio_path=temp_path, mode="universal_fairness")

            # Delete temp chunk immediately — no consent collected in live stream
            delete_audio_if_no_consent(temp_path, consent=False)

            # Build WebSocket payload
            # Hard rule — never send intent_label in WebSocket stream
            payload = {
                "content_quality_score": result["content_quality_score"],
                "prosody_summary": "flat pitch, steady pace",
                "misread_flag": False,
                "misread_description": None,
                # Include anonymized speaker ID so frontend can track session
                "speaker_id": speaker_id
            }

            # Send payload to Dev's frontend
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        print(f"[stream] Client disconnected — speaker_id: {speaker_id}")

    except Exception as e:
        print(f"[stream] Error: {e}")
        await websocket.close()
