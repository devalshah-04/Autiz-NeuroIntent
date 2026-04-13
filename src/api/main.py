# Import FastAPI framework to build our API server
from fastapi import FastAPI

# Import our three routers (we will build these next)
# Each router handles one endpoint
from routers import analyze, score, stream

# Create the FastAPI app instance — this is our server
app = FastAPI(
    title="NeuroIntent API",
    description="Prosody-independent content quality scoring for hiring evaluations",
    version="0.1.0"
)

# Register each router with the app
# This connects /analyze, /score, and /stream endpoints
app.include_router(analyze.router)
app.include_router(score.router)
app.include_router(stream.router)

# Health check endpoint — confirms server is alive
@app.get("/health")
def health_check():
    return {"status": "ok", "system_stage": "research_pilot"}
