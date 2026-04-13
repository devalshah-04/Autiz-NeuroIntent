# APIRouter and WebSocket classes for real-time streaming
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# JSON for parsing incoming messages from the frontend
import json

# OS for temp file handling
import os

# Import our mock pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mock_pipeline

# Create router instance for the /stream WebSocket endpoint
router = APIRouter()

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):

    # Accept the incoming WebSocket connection from Dev's frontend
    await websocket.accept()

    try:
        # Keep listening for incoming audio chunks in a loop
        # Dev's frontend sends 30ms audio chunks continuously
        while True:

            # Receive raw bytes from the WebSocket (audio chunk)
            data = await websocket.receive_bytes()

            # Save the chunk temporarily as a WAV file for the pipeline
            temp_path = "/tmp/stream_chunk.wav"
            with open(temp_path, "wb") as f:
                f.write(data)

            # Run the pipeline on this chunk in universal_fairness mode
            # speaker_declared is only used in the rehearsal tool, not live stream
            result = mock_pipeline.run(audio_path=temp_path, mode="universal_fairness")

            # Delete temp chunk immediately — never store raw audio
            os.remove(temp_path)

            # Build the WebSocket response payload
            # Hard rule — never send intent_label in WebSocket stream
            # Only these four fields go to Dev's frontend
            payload = {
                "content_quality_score": result["content_quality_score"],
                "prosody_summary": "flat pitch, steady pace",
                "misread_flag": False,
                "misread_description": None
            }

            # Send the payload back to Dev's frontend as JSON
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        # Client disconnected — this is normal, just clean up
        print("[stream] WebSocket client disconnected")

    except Exception as e:
        # Any other error — log it and close the connection cleanly
        print(f"[stream] Error: {e}")
        await websocket.close()
