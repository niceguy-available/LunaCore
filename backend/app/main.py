"""
FastAPI Application — Lunar Image Registration Platform
ISRO SIH 2026 | Problem Statement 26166
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from .tasks import task_manager
from .routers import registration, datasets, evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup: ensure directories exist
    Path("static/uploads").mkdir(parents=True, exist_ok=True)
    Path("static/results").mkdir(parents=True, exist_ok=True)
    Path("static/samples").mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: cleanup
    task_manager.executor.shutdown(wait=False)


app = FastAPI(
    title="Lunar Image Registration API",
    description=(
        "Multi-modal, sun-angle & scale-invariant image correspondence "
        "pipeline for Chandrayaan-2 OHRC/TMC-2/IIRS images. "
        "ISRO SIH 2026 — Problem Statement 26166."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploaded images and results
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(registration.router, prefix="/api", tags=["Registration"])
app.include_router(datasets.router, prefix="/api", tags=["Datasets"])
app.include_router(evaluation.router, prefix="/api", tags=["Evaluation"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Lunar Registration API", "version": "1.0.0"}


@app.get("/api/config/defaults")
async def get_defaults():
    """Return default pipeline configuration."""
    from .models import RegistrationConfig
    return RegistrationConfig().model_dump()
