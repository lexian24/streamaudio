"""
Routes package for FastAudio API.

This package contains organized route modules that separate different
API functionality into manageable files.
"""

from fastapi import APIRouter
from .health import router as health_router
from .analysis import router as analysis_router  
from .vad import router as vad_router
from .recordings import router as recordings_router
from .speakers import router as speakers_router
from .persistent_speakers import router as persistent_speakers_router
from .websocket import router as websocket_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(health_router, tags=["health"])
api_router.include_router(analysis_router, tags=["analysis"])
api_router.include_router(vad_router, prefix="/vad", tags=["voice-activity-detection"])
api_router.include_router(recordings_router, prefix="/api/recordings", tags=["recordings"])
api_router.include_router(speakers_router, prefix="/api/speakers", tags=["speakers"])
api_router.include_router(persistent_speakers_router, prefix="/api/persistent-speakers", tags=["persistent-speakers"])
api_router.include_router(websocket_router, tags=["websocket"])

__all__ = [
    "api_router",
    "health_router",
    "analysis_router", 
    "vad_router",
    "recordings_router",
    "speakers_router",
    "persistent_speakers_router",
    "websocket_router"
]