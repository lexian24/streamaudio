"""
Database package for StreamAudio application.

This package contains database models, configuration, and utilities
for managing audio recordings, processing results, and speaker data.
"""

from .models import Recording, ProcessingResult, Speaker, SpeakerSegment, Base
from .config import (
    engine, 
    async_engine, 
    SessionLocal, 
    AsyncSessionLocal,
    get_db_session,
    get_async_db_session,
    init_database,
    reset_database,
    create_tables,
    DatabaseManager
)

__all__ = [
    # Models
    "Recording",
    "ProcessingResult", 
    "Speaker",
    "SpeakerSegment",
    "Base",
    
    # Database engines and sessions
    "engine",
    "async_engine",
    "SessionLocal",
    "AsyncSessionLocal",
    
    # Dependency functions
    "get_db_session",
    "get_async_db_session",
    
    # Utility functions
    "init_database",
    "reset_database", 
    "create_tables",
    "DatabaseManager"
]