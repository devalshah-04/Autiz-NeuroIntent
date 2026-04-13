# APIRouter and WebSocket classes for real-time streaming
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Torch for running Silero VAD model
import torch

# OS for temp file handling
import os

# Struct for converting raw bytes to audio samples
import struct

# Import our mock pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mock_pipeline

# Create router instance for the /stream WebSocket endpoint
router = APIRouter()

# Load Silero VAD model once at startup — not on every request
# This keeps latency low (~1ms per chunk)
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
    # Convert raw bytes to list of 16-bit integer samples
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", audio_bytes)

    # Convert samples to float32 tensor normalized between -1 and 1
    audio_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0

    # Run Silero VAD — returns list of speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate
    )

    # If any speech timestamps found, speech is present
    return len(speech_timestamps) > 0

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):

    # Accept the incoming WebSocket connection from Dev's frontend
    await websocket.accept()
    print("[stream] Client connected")

    try:
        # Keep listening for incoming audio chunks in a loop
        while True:

            # Receive raw bytes from the WebSocket (30ms audio chunk)
            data = await websocket.receive_bytes()

            # Check if this chunk contains speech using Silero VAD
            # If silence, skip processing entirely — saves compute and latency
            if not is_speech(data):
                continue

            # Speech detected — save chunk temporarily for pipeline
            temp_path = "/tmp/stream_chunk.wav"
            with open(temp_path, "wb") as f:
                f.write(data)

            # Run pipeline in universal_fairness mode
            result = mock_pipeline.run(audio_path=temp_path, mode="universal_fairness")

            # Delete temp chunk immediately — never store raw audio
            os.remove(temp_path)

            # Build WebSocket response payload
            # Hard rule — never send intent_label in WebSocket stream
            payload = {
                "content_quality_score": result["content_quality_score"],
                "prosody_summary": "flat pitch, steady pace",
                "misread_flag": False,
                "misread_description": None
            }

            # Send payload back to Dev's frontend as JSON
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        # Client disconnected — normal, just log it
        print("[stream] Client disconnected")

    except Exception as e:
        # Any other error — log and close cleanly
        print(f"[stream] Error: {e}")
        await websocket.close()
