from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import torch
import os
import struct
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mock_pipeline
from utils import delete_audio_if_no_consent, generate_speaker_id

router = APIRouter()

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True
)

(get_speech_timestamps, _, read_audio, *_) = vad_utils

def is_speech(audio_bytes: bytes, sample_rate: int = 16000) -> bool:
    num_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{num_samples}h", audio_bytes)
    audio_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)
    return len(speech_timestamps) > 0

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    speaker_id = generate_speaker_id()
    print(f"[stream] Client connected — speaker_id: {speaker_id}")
    try:
        while True:
            data = await websocket.receive_bytes()
            if not is_speech(data):
                continue
            temp_path = "/tmp/stream_chunk.wav"
            with open(temp_path, "wb") as f:
                f.write(data)
            result = mock_pipeline.run(audio_path=temp_path, mode="universal_fairness")
            delete_audio_if_no_consent(temp_path, consent=False)
            payload = {
                "content_quality_score": result["content_quality_score"],
                "prosody_summary": "flat pitch, steady pace",
                "misread_flag": False,
                "misread_description": None,
                "speaker_id": speaker_id
            }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        print(f"[stream] Client disconnected — speaker_id: {speaker_id}")
    except Exception as e:
        print(f"[stream] Error: {e}")
        await websocket.close()
