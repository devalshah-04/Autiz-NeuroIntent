import modal

app = modal.App("neurointent-api")

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
    )
    .add_local_dir(".", remote_path="/root/api")
)

@app.function(image=image, min_containers=1)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    sys.path.insert(0, "/root/api")
    os.chdir("/root/api")
    from main import app as web_app
    return web_app

@app.function(image=image, gpu="T4")
def run_fusion_layer(audio_path: str, mode: str):
    import sys
    sys.path.insert(0, "/root/api")
    import mock_pipeline
    return mock_pipeline.run(audio_path=audio_path, mode=mode)

@app.function(image=image, gpu="A10G")
def run_llama_reasoning(content: str):
    return {"reasoning_summary": "Content demonstrates strong semantic coherence.", "confidence": "medium"}
