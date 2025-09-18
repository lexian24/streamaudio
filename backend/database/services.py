"""
Database service layer for StreamAudio application.

This module implements the Repository pattern and provides a clean
interface for database operations following industry standards.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, desc, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from .models import Recording, ProcessingResult, Speaker, SpeakerSegment


class RecordingService:
    """
    Service layer for Recording operations.
    Implements Repository pattern for clean separation of concerns.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_recording(
        self,
        filename: str,
        file_path: str,
        original_filename: Optional[str] = None,
        file_size: Optional[int] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        format: Optional[str] = None
    ) -> Recording:
        """Create a new recording record."""
        try:
            recording = Recording(
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                format=format
            )
            
            self.db.add(recording)
            await self.db.commit()
            await self.db.refresh(recording)
            return recording
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create recording: {str(e)}")
    
    async def get_recording(self, recording_id: int) -> Optional[Recording]:
        """Get a recording by ID."""
        try:
            result = await self.db.execute(
                select(Recording).where(Recording.id == recording_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get recording: {str(e)}")
    
    async def get_recordings(
        self,
        limit: int = 50,
        offset: int = 0,
        format_filter: Optional[str] = None
    ) -> List[Recording]:
        """Get recordings with pagination and optional filtering."""
        try:
            query = select(Recording).order_by(desc(Recording.created_at))
            
            if format_filter:
                query = query.where(Recording.format == format_filter)
            
            query = query.offset(offset).limit(limit)
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get recordings: {str(e)}")
    
    async def update_recording_metadata(
        self,
        recording_id: int,
        **kwargs
    ) -> Optional[Recording]:
        """Update recording metadata."""
        try:
            recording = await self.get_recording(recording_id)
            if not recording:
                return None
            
            for key, value in kwargs.items():
                if hasattr(recording, key):
                    setattr(recording, key, value)
            
            recording.updated_at = datetime.utcnow()
            await self.db.commit()
            await self.db.refresh(recording)
            return recording
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to update recording: {str(e)}")
    
    async def delete_recording(self, recording_id: int) -> bool:
        """Delete a recording and its associated results."""
        try:
            recording = await self.get_recording(recording_id)
            if not recording:
                return False
            
            await self.db.delete(recording)
            await self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to delete recording: {str(e)}")


class ProcessingResultService:
    """
    Service layer for ProcessingResult operations.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_processing_result(
        self,
        recording_id: int,
        transcription: Optional[str] = None,
        confidence_score: Optional[float] = None,
        diarization_data: Optional[Dict] = None,
        emotions_data: Optional[Dict] = None,
        model_versions: Optional[Dict] = None
    ) -> ProcessingResult:
        """Create a new processing result."""
        try:
            # Extract metadata from results
            num_speakers = None
            dominant_emotion = None
            emotion_confidence = None
            
            if diarization_data and 'speakers' in diarization_data:
                num_speakers = len(diarization_data['speakers'])
            
            if emotions_data and 'dominant_emotion' in emotions_data:
                dominant_emotion = emotions_data['dominant_emotion']
                emotion_confidence = emotions_data.get('confidence', None)
            
            result = ProcessingResult(
                recording_id=recording_id,
                transcription=transcription,
                confidence_score=confidence_score,
                diarization_json=diarization_data,
                num_speakers=num_speakers,
                emotions_json=emotions_data,
                dominant_emotion=dominant_emotion,
                emotion_confidence=emotion_confidence,
                model_versions=model_versions,
                status="completed"
            )
            
            self.db.add(result)
            await self.db.commit()
            await self.db.refresh(result)
            return result
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create processing result: {str(e)}")
    
    async def get_processing_result(self, result_id: int) -> Optional[ProcessingResult]:
        """Get a processing result by ID with related data."""
        try:
            result = await self.db.execute(
                select(ProcessingResult)
                .options(selectinload(ProcessingResult.recording))
                .options(selectinload(ProcessingResult.speaker_segments))
                .where(ProcessingResult.id == result_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get processing result: {str(e)}")
    
    async def get_results_by_recording(self, recording_id: int) -> List[ProcessingResult]:
        """Get all processing results for a recording."""
        try:
            result = await self.db.execute(
                select(ProcessingResult)
                .where(ProcessingResult.recording_id == recording_id)
                .order_by(desc(ProcessingResult.processed_at))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get results by recording: {str(e)}")
    
    async def update_processing_status(
        self,
        result_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[ProcessingResult]:
        """Update processing status."""
        try:
            result = await self.get_processing_result(result_id)
            if not result:
                return None
            
            result.status = status
            if error_message:
                result.error_message = error_message
            
            await self.db.commit()
            await self.db.refresh(result)
            return result
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to update processing status: {str(e)}")


class SpeakerService:
    """
    Service layer for Speaker operations.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_speaker(
        self,
        name: str,
        email: Optional[str] = None,
        embedding_path: Optional[str] = None,
        enrollment_recording_id: Optional[int] = None
    ) -> Speaker:
        """Create a new speaker profile."""
        try:
            speaker = Speaker(
                name=name,
                email=email,
                embedding_path=embedding_path,
                enrollment_recording_id=enrollment_recording_id
            )
            
            self.db.add(speaker)
            await self.db.commit()
            await self.db.refresh(speaker)
            return speaker
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create speaker: {str(e)}")
    
    async def get_speaker(self, speaker_id: int) -> Optional[Speaker]:
        """Get a speaker by ID."""
        try:
            result = await self.db.execute(
                select(Speaker).where(Speaker.id == speaker_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get speaker: {str(e)}")
    
    async def get_speakers(self, active_only: bool = True) -> List[Speaker]:
        """Get all speakers."""
        try:
            query = select(Speaker).order_by(Speaker.name)
            if active_only:
                query = query.where(Speaker.is_active == True)
            
            result = await self.db.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get speakers: {str(e)}")
    
    async def find_speaker_by_name(self, name: str) -> Optional[Speaker]:
        """Find a speaker by name."""
        try:
            result = await self.db.execute(
                select(Speaker).where(Speaker.name.ilike(f"%{name}%"))
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to find speaker by name: {str(e)}")


class SpeakerSegmentService:
    """
    Service layer for SpeakerSegment operations.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_speaker_segments(
        self,
        result_id: int,
        segments_data: List[Dict[str, Any]]
    ) -> List[SpeakerSegment]:
        """Create multiple speaker segments for a processing result."""
        try:
            segments = []
            for segment_data in segments_data:
                segment = SpeakerSegment(
                    result_id=result_id,
                    speaker_id=segment_data.get('speaker_id'),
                    start_time=segment_data['start_time'],
                    end_time=segment_data['end_time'],
                    duration=segment_data['end_time'] - segment_data['start_time'],
                    confidence=segment_data.get('confidence'),
                    segment_text=segment_data.get('text'),
                    speaker_label=segment_data.get('speaker_label')
                )
                segments.append(segment)
                self.db.add(segment)
            
            await self.db.commit()
            for segment in segments:
                await self.db.refresh(segment)
            
            return segments
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create speaker segments: {str(e)}")
    
    async def get_segments_by_result(self, result_id: int) -> List[SpeakerSegment]:
        """Get all segments for a processing result."""
        try:
            result = await self.db.execute(
                select(SpeakerSegment)
                .options(selectinload(SpeakerSegment.speaker))
                .where(SpeakerSegment.result_id == result_id)
                .order_by(SpeakerSegment.start_time)
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get segments by result: {str(e)}")


class DatabaseService:
    """
    Main database service that aggregates all individual services.
    Provides a single interface for all database operations.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.recordings = RecordingService(db_session)
        self.processing_results = ProcessingResultService(db_session)
        self.speakers = SpeakerService(db_session)
        self.speaker_segments = SpeakerSegmentService(db_session)
    
    async def close(self):
        """Close the database session."""
        await self.db.close()
    
    async def rollback(self):
        """Rollback the current transaction."""
        await self.db.rollback()
    
    async def commit(self):
        """Commit the current transaction."""
        await self.db.commit()