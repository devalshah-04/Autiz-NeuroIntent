import os
import modal

app = modal.App("neurointent-api")

# Resolve paths relative to this file so modal deploy works from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "models"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.135.3",
        "uvicorn==0.44.0",
        "python-multipart==0.0.26",
        "websockets==16.0",
        "pydantic==2.13.0",
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "silero-vad==6.2.1",
        "faster-whisper==1.2.1",
        "transformers>=4.40.0",
        "sentencepiece>=0.1.99",
        "opensmile>=2.4.0",
        "numpy>=1.24.0",
        "imageio-ffmpeg>=0.4.9",   # bundles its own ffmpeg binary — no apt required
    )
    .add_local_dir(_HERE, remote_path="/root/api")
    .add_local_dir(_MODELS_DIR, remote_path="/root/models")
)


@app.function(image=image, min_containers=1)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root/api")
    os.chdir("/root/api")
    from main import app as web_app
    return web_app


@app.function(image=image, gpu="T4")
def run_fusion_layer(audio_path: str, mode: str):
    import sys
    sys.path.insert(0, "/root/api")
    import pipeline
    return pipeline.run(audio_path=audio_path, mode=mode)


@app.function(image=image, gpu="A10G")
def run_llama_reasoning(content: str):
    return {
        "reasoning_summary": "Content demonstrates strong semantic coherence.",
        "confidence": "medium",
    }
