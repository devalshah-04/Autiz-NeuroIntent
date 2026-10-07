# OS for reading the allowed-origins environment variable
import os

# Import FastAPI framework to build our API server
from fastapi import FastAPI

# CORS middleware lets the browser apps (served from a different origin) call this API
from fastapi.middleware.cors import CORSMiddleware

# Routers: /analyze, /score, /about. (/stream was removed in Phase 4; the live overlay is future work.)
from routers import about, analyze, score
import pipeline
from schemas import HealthResponse

# Create the FastAPI app instance — this is our server
app = FastAPI(
    title="NeuroIntent API",
    description="Prosody-independent content quality scoring for hiring evaluations",
    version="0.1.0"
)

# Allowed browser origins, comma-separated, from the CORS_ALLOWED_ORIGINS environment variable.
# Default: the three local Vite dev servers (rehearsal, evaluator, overlay).
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:5174,http://localhost:5175"


def get_allowed_origins() -> list[str]:
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register each router with the app
app.include_router(analyze.router)
app.include_router(score.router)
app.include_router(about.router)

# Health check endpoint — confirms server is alive
@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "system_stage": "research_pilot", "smoke_artifacts": pipeline.smoke_artifacts_flag()}
