"""
Database service layer for StreamAudio application.

This module implements the Repository pattern and provides a clean
interface for database operations following industry standards.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import io

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, desc, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    Recording, ProcessingResult, Speaker, SpeakerSegment,
    PersistentSpeaker, SpeakerEmbedding, SpeakerMapping, SpeakerReviewQueue,
    ProcessingTask
)


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
        embeddings_data: Optional[Dict] = None,
        enrollment_recording_id: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> Speaker:
        """Create a new speaker profile."""
        try:
            speaker = Speaker(
                name=name,
                email=email,
                embedding_path=embedding_path,
                embeddings_data=embeddings_data,
                enrollment_recording_id=enrollment_recording_id,
                confidence_threshold=confidence_threshold or 0.7,
                num_enrollments=len(embeddings_data.get('embeddings', [])) if embeddings_data else 0
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
    
    async def delete_speaker(self, speaker_id: int) -> bool:
        """Delete a speaker profile."""
        try:
            result = await self.db.execute(
                select(Speaker).where(Speaker.id == speaker_id)
            )
            speaker = result.scalar_one_or_none()
            
            if not speaker:
                return False
            
            await self.db.delete(speaker)
            await self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to delete speaker: {str(e)}")


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


class PersistentSpeakerService:
    """
    Service layer for PersistentSpeaker operations.
    Handles persistent speaker identities that remain consistent across recordings.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    def _compress_embedding(self, embedding: np.ndarray) -> bytes:
        """Compress numpy embedding to bytes for database storage."""
        # Convert to float16 for 50% space savings
        compressed = embedding.astype(np.float16)
        buffer = io.BytesIO()
        np.save(buffer, compressed)
        return buffer.getvalue()
    
    def _decompress_embedding(self, compressed_data: bytes) -> np.ndarray:
        """Decompress bytes back to numpy embedding."""
        buffer = io.BytesIO(compressed_data)
        return np.load(buffer).astype(np.float32)
    
    async def generate_speaker_id(self) -> str:
        """Generate next available persistent speaker ID."""
        try:
            # Find highest existing number
            result = await self.db.execute(
                select(PersistentSpeaker.id).order_by(desc(PersistentSpeaker.id))
            )
            existing_ids = [row[0] for row in result.fetchall()]
            
            # Extract numbers and find next available
            existing_numbers = []
            for speaker_id in existing_ids:
                if speaker_id.startswith("SPEAKER_"):
                    try:
                        num = int(speaker_id.split("_")[1])
                        existing_numbers.append(num)
                    except (IndexError, ValueError):
                        continue
            
            # Find next available number (fill gaps first)
            if not existing_numbers:
                next_num = 1
            else:
                existing_numbers.sort()
                next_num = 1
                for num in existing_numbers:
                    if num == next_num:
                        next_num += 1
                    elif num > next_num:
                        break
            
            return f"SPEAKER_{next_num:03d}"
            
        except SQLAlchemyError as e:
            raise Exception(f"Failed to generate speaker ID: {str(e)}")
    
    async def create_persistent_speaker(
        self,
        name: Optional[str] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        embedding_metadata: Optional[Dict] = None,
        first_seen_recording_id: Optional[int] = None,
        enrollment_method: str = "manual"
    ) -> PersistentSpeaker:
        """Create a new persistent speaker."""
        try:
            speaker_id = await self.generate_speaker_id()
            
            # Calculate average embedding if provided
            avg_embedding_data = None
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding_data = self._compress_embedding(avg_embedding)
            
            speaker = PersistentSpeaker(
                id=speaker_id,
                name=name,
                first_seen_recording_id=first_seen_recording_id,
                enrollment_method=enrollment_method,
                avg_embedding=avg_embedding_data,
                embedding_metadata=embedding_metadata or {}
            )
            
            self.db.add(speaker)
            await self.db.commit()
            await self.db.refresh(speaker)
            
            return speaker
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create persistent speaker: {str(e)}")
    
    async def get_persistent_speaker(self, speaker_id: str) -> Optional[PersistentSpeaker]:
        """Get persistent speaker by ID."""
        try:
            result = await self.db.execute(
                select(PersistentSpeaker)
                .options(selectinload(PersistentSpeaker.embeddings))
                .where(PersistentSpeaker.id == speaker_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get persistent speaker: {str(e)}")
    
    async def get_all_persistent_speakers(
        self,
        active_only: bool = True,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[PersistentSpeaker]:
        """Get all persistent speakers with optional filtering."""
        try:
            query = select(PersistentSpeaker).options(selectinload(PersistentSpeaker.embeddings))
            
            if active_only:
                query = query.where(PersistentSpeaker.is_active == True)
            
            query = query.order_by(desc(PersistentSpeaker.last_seen_at))
            
            if limit:
                query = query.limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get persistent speakers: {str(e)}")
    
    async def update_speaker_stats(
        self,
        speaker_id: str,
        additional_speaking_time: float = 0.0,
        additional_segments: int = 0,
        new_recording: bool = False
    ) -> None:
        """Update speaker usage statistics."""
        try:
            speaker = await self.get_persistent_speaker(speaker_id)
            if not speaker:
                return
            
            speaker.total_speaking_time += additional_speaking_time
            speaker.segments_count += additional_segments
            speaker.last_seen_at = datetime.utcnow()
            
            if new_recording:
                speaker.recordings_count += 1
            
            await self.db.commit()
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to update speaker stats: {str(e)}")
    
    async def delete_persistent_speaker(self, speaker_id: str) -> bool:
        """Delete a persistent speaker and all associated data."""
        try:
            speaker = await self.get_persistent_speaker(speaker_id)
            if not speaker:
                return False
            
            # Delete the speaker (cascading should handle embeddings, mappings, etc.)
            await self.db.delete(speaker)
            await self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to delete speaker: {str(e)}")


class SpeakerEmbeddingService:
    """
    Service layer for SpeakerEmbedding operations.
    Handles multiple embeddings per persistent speaker.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    def _compress_embedding(self, embedding: np.ndarray) -> bytes:
        """Compress numpy embedding to bytes for database storage."""
        compressed = embedding.astype(np.float16)
        buffer = io.BytesIO()
        np.save(buffer, compressed)
        return buffer.getvalue()
    
    def _decompress_embedding(self, compressed_data: bytes) -> np.ndarray:
        """Decompress bytes back to numpy embedding."""
        buffer = io.BytesIO(compressed_data)
        return np.load(buffer).astype(np.float32)
    
    async def add_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        quality_score: Optional[float] = None,
        snr_db: Optional[float] = None,
        duration: Optional[float] = None,
        source_recording_id: Optional[int] = None,
        source_segment_start: Optional[float] = None,
        source_segment_end: Optional[float] = None,
        enrollment_method: str = "manual"
    ) -> SpeakerEmbedding:
        """Add a new embedding for a persistent speaker."""
        try:
            compressed_embedding = self._compress_embedding(embedding)
            
            embedding_record = SpeakerEmbedding(
                speaker_id=speaker_id,
                embedding=compressed_embedding,
                embedding_dim=len(embedding),
                quality_score=quality_score,
                snr_db=snr_db,
                duration=duration,
                source_recording_id=source_recording_id,
                source_segment_start=source_segment_start,
                source_segment_end=source_segment_end,
                enrollment_method=enrollment_method
            )
            
            self.db.add(embedding_record)
            await self.db.commit()
            await self.db.refresh(embedding_record)
            
            return embedding_record
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to add speaker embedding: {str(e)}")
    
    async def get_embeddings_for_speaker(self, speaker_id: str) -> List[SpeakerEmbedding]:
        """Get all embeddings for a persistent speaker."""
        try:
            result = await self.db.execute(
                select(SpeakerEmbedding)
                .where(SpeakerEmbedding.speaker_id == speaker_id)
                .order_by(desc(SpeakerEmbedding.quality_score))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get speaker embeddings: {str(e)}")
    
    async def get_embeddings_as_numpy(self, speaker_id: str) -> List[np.ndarray]:
        """Get all embeddings for a speaker as numpy arrays."""
        try:
            embeddings = await self.get_embeddings_for_speaker(speaker_id)
            return [self._decompress_embedding(emb.embedding) for emb in embeddings]
        except Exception as e:
            raise Exception(f"Failed to get embeddings as numpy: {str(e)}")


class SpeakerMappingService:
    """
    Service layer for SpeakerMapping operations.
    Handles session-to-persistent speaker mappings.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_mapping(
        self,
        recording_id: int,
        session_speaker_label: str,
        persistent_speaker_id: str,
        assignment_confidence: Optional[float] = None,
        assignment_method: str = "auto",
        similarity_score: Optional[float] = None,
        needs_review: bool = False
    ) -> SpeakerMapping:
        """Create a new speaker mapping."""
        try:
            mapping = SpeakerMapping(
                recording_id=recording_id,
                session_speaker_label=session_speaker_label,
                persistent_speaker_id=persistent_speaker_id,
                assignment_confidence=assignment_confidence,
                assignment_method=assignment_method,
                similarity_score=similarity_score,
                needs_review=needs_review
            )
            
            self.db.add(mapping)
            await self.db.commit()
            await self.db.refresh(mapping)
            
            return mapping
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create speaker mapping: {str(e)}")
    
    async def get_mappings_for_recording(self, recording_id: int) -> List[SpeakerMapping]:
        """Get all speaker mappings for a recording."""
        try:
            result = await self.db.execute(
                select(SpeakerMapping)
                .options(selectinload(SpeakerMapping.persistent_speaker))
                .where(SpeakerMapping.recording_id == recording_id)
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get speaker mappings: {str(e)}")
    
    async def get_mapping(
        self,
        recording_id: int,
        session_speaker_label: str
    ) -> Optional[SpeakerMapping]:
        """Get specific mapping for a recording and session speaker."""
        try:
            result = await self.db.execute(
                select(SpeakerMapping)
                .options(selectinload(SpeakerMapping.persistent_speaker))
                .where(
                    and_(
                        SpeakerMapping.recording_id == recording_id,
                        SpeakerMapping.session_speaker_label == session_speaker_label
                    )
                )
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get speaker mapping: {str(e)}")


class SpeakerReviewQueueService:
    """
    Service layer for SpeakerReviewQueue operations.
    Handles assignments that need manual review.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def add_to_review_queue(
        self,
        recording_id: int,
        session_speaker_label: str,
        suggested_assignments: List[Dict],
        segment_count: Optional[int] = None,
        total_duration: Optional[float] = None,
        audio_quality: Optional[float] = None,
        priority: int = 1
    ) -> SpeakerReviewQueue:
        """Add a speaker assignment to the review queue."""
        try:
            review_item = SpeakerReviewQueue(
                recording_id=recording_id,
                session_speaker_label=session_speaker_label,
                suggested_assignments=suggested_assignments,
                segment_count=segment_count,
                total_duration=total_duration,
                audio_quality=audio_quality,
                priority=priority
            )
            
            self.db.add(review_item)
            await self.db.commit()
            await self.db.refresh(review_item)
            
            return review_item
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to add to review queue: {str(e)}")
    
    async def get_pending_reviews(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[SpeakerReviewQueue]:
        """Get pending reviews ordered by priority."""
        try:
            query = select(SpeakerReviewQueue).where(
                SpeakerReviewQueue.status == "pending"
            ).order_by(
                SpeakerReviewQueue.priority,
                SpeakerReviewQueue.created_at
            )
            
            if limit:
                query = query.limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            raise Exception(f"Failed to get pending reviews: {str(e)}")


class ProcessingTaskService:
    """
    Service layer for ProcessingTask operations.
    Manages Celery background task tracking and status.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_task(
        self,
        task_id: str,
        task_type: str,
        recording_id: Optional[int] = None,
        task_name: Optional[str] = None
    ) -> ProcessingTask:
        """Create a new processing task record."""
        try:
            task = ProcessingTask(
                task_id=task_id,
                task_type=task_type,
                recording_id=recording_id,
                task_name=task_name,
                status='queued'
            )

            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)

            return task

        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to create task: {str(e)}")

    async def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by Celery task ID."""
        try:
            result = await self.db.execute(
                select(ProcessingTask).where(ProcessingTask.task_id == task_id)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as e:
            raise Exception(f"Failed to get task: {str(e)}")

    async def get_task_by_id(self, id: int) -> Optional[ProcessingTask]:
        """Get task by database ID."""
        try:
            result = await self.db.execute(
                select(ProcessingTask).where(ProcessingTask.id == id)
            )
            return result.scalar_one_or_none()

        except SQLAlchemyError as e:
            raise Exception(f"Failed to get task by ID: {str(e)}")

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[int] = None,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> ProcessingTask:
        """Update task status and progress."""
        try:
            task = await self.get_task(task_id)
            if not task:
                raise Exception(f"Task {task_id} not found")

            task.status = status
            if progress is not None:
                task.progress = progress
            if result_data is not None:
                task.result_data = result_data
            if error_message is not None:
                task.error_message = error_message

            if status == 'processing' and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in ('completed', 'failed', 'cancelled'):
                task.completed_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(task)

            return task

        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to update task status: {str(e)}")

    async def get_tasks_by_recording(self, recording_id: int) -> List[ProcessingTask]:
        """Get all tasks for a specific recording."""
        try:
            result = await self.db.execute(
                select(ProcessingTask)
                .where(ProcessingTask.recording_id == recording_id)
                .order_by(desc(ProcessingTask.created_at))
            )
            return result.scalars().all()

        except SQLAlchemyError as e:
            raise Exception(f"Failed to get tasks by recording: {str(e)}")

    async def get_recent_tasks(self, limit: int = 50, offset: int = 0) -> List[ProcessingTask]:
        """Get recent tasks ordered by creation time."""
        try:
            result = await self.db.execute(
                select(ProcessingTask)
                .order_by(desc(ProcessingTask.created_at))
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()

        except SQLAlchemyError as e:
            raise Exception(f"Failed to get recent tasks: {str(e)}")

    async def delete_old_tasks(self, days: int = 7) -> int:
        """Delete completed tasks older than specified days."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            result = await self.db.execute(
                select(ProcessingTask).where(
                    and_(
                        ProcessingTask.status.in_(['completed', 'failed']),
                        ProcessingTask.completed_at < cutoff_date
                    )
                )
            )
            tasks = result.scalars().all()

            for task in tasks:
                await self.db.delete(task)

            await self.db.commit()
            return len(tasks)

        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to delete old tasks: {str(e)}")

    async def cancel_task(self, task_id: str) -> ProcessingTask:
        """Cancel a pending or processing task."""
        try:
            task = await self.get_task(task_id)
            if not task:
                raise Exception(f"Task {task_id} not found")

            if task.status not in ('queued', 'processing'):
                raise Exception(f"Cannot cancel task with status: {task.status}")

            task.status = 'cancelled'
            task.completed_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(task)

            # Also revoke the Celery task
            from celery_app import celery_app
            celery_app.control.revoke(task_id, terminate=True)

            return task

        except SQLAlchemyError as e:
            await self.db.rollback()
            raise Exception(f"Failed to cancel task: {str(e)}")


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

        # New persistent speaker services
        self.persistent_speakers = PersistentSpeakerService(db_session)
        self.speaker_embeddings = SpeakerEmbeddingService(db_session)
        self.speaker_mappings = SpeakerMappingService(db_session)
        self.speaker_review_queue = SpeakerReviewQueueService(db_session)

        # Processing task service
        self.processing_tasks = ProcessingTaskService(db_session)
    
    async def close(self):
        """Close the database session."""
        await self.db.close()
    
    async def rollback(self):
        """Rollback the current transaction."""
        await self.db.rollback()
    
    async def commit(self):
        """Commit the current transaction."""
        await self.db.commit()