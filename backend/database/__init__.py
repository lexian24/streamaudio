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
from .services import (
    DatabaseService,
    RecordingService,
    ProcessingResultService,
    SpeakerService,
    SpeakerSegmentService
)
from .dependencies import (
    get_database_session,
    get_database_service,
    get_recording_service,
    get_processing_result_service,
    get_speaker_service
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
    
    # Legacy dependency functions
    "get_db_session",
    "get_async_db_session",
    
    # Service layer
    "DatabaseService",
    "RecordingService",
    "ProcessingResultService",
    "SpeakerService",
    "SpeakerSegmentService",
    
    # FastAPI dependencies
    "get_database_session",
    "get_database_service",
    "get_recording_service",
    "get_processing_result_service",
    "get_speaker_service",
    
    # Utility functions
    "init_database",
    "reset_database", 
    "create_tables",
    "DatabaseManager"
]