"""
Routes package for FastAudio API.

This package contains organized route modules that separate different
API functionality into manageable files.

API Versioning:
    - /api/v1/* - Version 1 (current stable)
    - /health - Health check (unversioned)
    - /ws - WebSocket (unversioned)
"""

from fastapi import APIRouter
from .health import router as health_router
from .analysis import router as analysis_router
from .vad import router as vad_router
from .recordings import router as recordings_router
from .speakers import router as speakers_router
from .persistent_speakers import router as persistent_speakers_router
from .websocket import router as websocket_router
from .tasks import router as tasks_router

# Create API v1 router
api_v1_router = APIRouter(prefix="/api/v1")

# Include versioned routes
api_v1_router.include_router(analysis_router, prefix="/analysis", tags=["v1-analysis"])
api_v1_router.include_router(vad_router, prefix="/vad", tags=["v1-voice-activity-detection"])
api_v1_router.include_router(recordings_router, prefix="/recordings", tags=["v1-recordings"])
api_v1_router.include_router(speakers_router, prefix="/speakers", tags=["v1-speakers"])
api_v1_router.include_router(persistent_speakers_router, prefix="/persistent-speakers", tags=["v1-persistent-speakers"])
api_v1_router.include_router(tasks_router, prefix="/tasks", tags=["v1-tasks"])

# Create main API router (includes all versions)
api_router = APIRouter()

# Include version routers
api_router.include_router(api_v1_router)

# BACKWARD COMPATIBILITY: Include legacy routes (without /api/v1/ prefix)
# These forward to the same handlers for existing frontends
api_router.include_router(vad_router, prefix="/vad", tags=["legacy-vad"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["legacy-analysis"])
api_router.include_router(recordings_router, prefix="/api/recordings", tags=["legacy-recordings"])
api_router.include_router(speakers_router, prefix="/api/speakers", tags=["legacy-speakers"])
api_router.include_router(persistent_speakers_router, prefix="/api/persistent-speakers", tags=["legacy-persistent-speakers"])
api_router.include_router(tasks_router, prefix="/api/tasks", tags=["legacy-tasks"])

# Include unversioned routes (health, websocket)
api_router.include_router(health_router, tags=["health"])
api_router.include_router(websocket_router, tags=["websocket"])

__all__ = [
    "api_router",
    "api_v1_router",
    "health_router",
    "analysis_router",
    "vad_router",
    "recordings_router",
    "speakers_router",
    "persistent_speakers_router",
    "websocket_router",
    "tasks_router"
]