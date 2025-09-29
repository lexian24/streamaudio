"""
FastAudio - Refactored Main Application

A clean, organized FastAPI application using route modules for better maintainability.
"""

import uvicorn
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import route modules
from routes import api_router
from routes.dependencies import set_auto_recorder

# Import services
from services.auto_recorder import AutoRecorder
from database import init_database

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce debug noise from specific modules
logging.getLogger("services.auto_recorder").setLevel(logging.INFO)
logging.getLogger("services.voice_activity_detection").setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    try:
        logger.info("Starting FastAudio server...")
        
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        logger.info("Database initialized successfully!")
        
        logger.info("AI models will be loaded on first request (lazy loading)")
        
        # Initialize and set auto recorder
        auto_recorder = AutoRecorder()
        set_auto_recorder(auto_recorder)
        
        logger.info("FastAudio server ready!")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down FastAudio server...")
    from routes.dependencies import get_auto_recorder
    auto_recorder = get_auto_recorder()
    if auto_recorder:
        auto_recorder.stop_monitoring()
    logger.info("FastAudio server shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="FastAudio API",
    description="Audio Analysis with Whisper + pyannote - Organized with Routes",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route modules
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        "main_refactored:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )