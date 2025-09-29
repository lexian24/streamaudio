"""
Database dependency injection for FastAPI endpoints.

This module provides dependency functions for injecting database
services into API endpoints following FastAPI best practices.
"""

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .config import AsyncSessionLocal
from .services import DatabaseService, RecordingService, ProcessingResultService, SpeakerService


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to provide database session to FastAPI endpoints.
    
    This function creates a new database session for each request
    and ensures it's properly closed when the request is completed.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_database_service(
    db: AsyncSession = Depends(get_database_session)
) -> DatabaseService:
    """
    Dependency to provide the main database service.
    
    This is the primary dependency for endpoints that need
    access to multiple database operations.
    """
    return DatabaseService(db)


async def get_recording_service(
    db: AsyncSession = Depends(get_database_session)
) -> RecordingService:
    """
    Dependency to provide recording service for recording-specific operations.
    """
    return RecordingService(db)


async def get_processing_result_service(
    db: AsyncSession = Depends(get_database_session)
) -> ProcessingResultService:
    """
    Dependency to provide processing result service.
    """
    return ProcessingResultService(db)


async def get_speaker_service(
    db: AsyncSession = Depends(get_database_session)
) -> SpeakerService:
    """
    Dependency to provide speaker service for speaker operations.
    """
    return SpeakerService(db)


def create_database_service():
    """
    Create a database service synchronously for use in audio processor initialization.
    Note: This creates a new session that should be used carefully.
    """
    import asyncio
    
    async def _get_service():
        async with AsyncSessionLocal() as session:
            return DatabaseService(session)
    
    return asyncio.run(_get_service())