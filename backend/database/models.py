"""
Database models for StreamAudio application.

This module defines the SQLAlchemy models for storing audio recordings,
processing results, speaker information, and speaker segments.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON, LargeBinary, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Recording(Base):
    """
    Model for storing audio recording metadata.
    
    Stores information about uploaded/recorded audio files including
    file path, duration, size, and creation timestamp.
    """
    __tablename__ = "recordings"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=True)  # Original name from upload
    file_path = Column(String(500), nullable=False)  # Full path to audio file
    file_size = Column(Integer, nullable=True)  # File size in bytes
    duration = Column(Float, nullable=True)  # Duration in seconds
    sample_rate = Column(Integer, nullable=True)  # Audio sample rate
    channels = Column(Integer, nullable=True)  # Number of audio channels
    format = Column(String(10), nullable=True)  # Audio format (wav, mp3, etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to processing results
    processing_results = relationship("ProcessingResult", back_populates="recording", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Recording(id={self.id}, filename='{self.filename}', duration={self.duration}s)>"


class ProcessingResult(Base):
    """
    Model for storing AI processing results for each recording.
    
    Contains transcription, diarization results, emotion analysis,
    and processing metadata.
    """
    __tablename__ = "processing_results"
    
    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    
    # Processing results
    transcription = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)  # Overall transcription confidence
    
    # Diarization results (stored as JSON)
    diarization_json = Column(JSON, nullable=True)
    num_speakers = Column(Integer, nullable=True)
    
    # Emotion analysis results (stored as JSON)
    emotions_json = Column(JSON, nullable=True)
    dominant_emotion = Column(String(50), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    
    # Processing metadata
    processing_duration = Column(Float, nullable=True)  # Time taken to process
    model_versions = Column(JSON, nullable=True)  # Versions of models used
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Processing status
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Relationships
    recording = relationship("Recording", back_populates="processing_results")
    speaker_segments = relationship("SpeakerSegment", back_populates="result", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ProcessingResult(id={self.id}, recording_id={self.recording_id}, status='{self.status}')>"


class Speaker(Base):
    """
    Model for storing speaker profiles and identification data.
    
    Used for speaker identification and enrollment. Stores speaker
    embeddings and metadata for future speaker recognition.
    """
    __tablename__ = "speakers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    
    # Speaker identification data
    embedding_path = Column(String(500), nullable=True)  # Path to speaker embedding file
    embeddings_data = Column(JSON, nullable=True)  # JSON data containing embeddings and metadata
    enrollment_recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True)
    
    # Metadata
    num_enrollments = Column(Integer, default=0)  # Number of enrollment samples
    total_speaking_time = Column(Float, default=0.0)  # Total speaking time in seconds
    confidence_threshold = Column(Float, default=0.7)  # Threshold for identification
    
    # Status and timestamps
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    enrollment_recording = relationship("Recording")
    speaker_segments = relationship("SpeakerSegment", back_populates="speaker")
    
    def __repr__(self):
        return f"<Speaker(id={self.id}, name='{self.name}', enrollments={self.num_enrollments})>"


class SpeakerSegment(Base):
    """
    Model for storing speaker segments within processing results.
    
    Links specific time segments to identified speakers with confidence scores.
    Each segment represents a continuous speech period by one speaker.
    """
    __tablename__ = "speaker_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("processing_results.id"), nullable=False)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)  # Null for unidentified speakers
    
    # Segment timing
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    duration = Column(Float, nullable=False)    # Segment duration in seconds
    
    # Identification confidence
    confidence = Column(Float, nullable=True)   # Confidence score for speaker identification
    
    # Segment metadata
    segment_text = Column(Text, nullable=True)  # Transcribed text for this segment
    speaker_label = Column(String(50), nullable=True)  # Original diarization label (e.g., "SPEAKER_00")
    
    # Relationships
    result = relationship("ProcessingResult", back_populates="speaker_segments")
    speaker = relationship("Speaker", back_populates="speaker_segments")
    
    def __repr__(self):
        speaker_name = self.speaker.name if self.speaker else "Unknown"
        return f"<SpeakerSegment(id={self.id}, speaker='{speaker_name}', {self.start_time:.1f}s-{self.end_time:.1f}s)>"


class PersistentSpeaker(Base):
    """
    Model for persistent speaker identities that maintain consistency across recordings.
    
    Unlike session-based speakers (SPEAKER_00, SPEAKER_01), persistent speakers have
    globally unique IDs that remain consistent across all recordings and sessions.
    """
    __tablename__ = "persistent_speakers"
    
    # Use string ID for human-readable persistent speaker identifiers
    id = Column(String(20), primary_key=True, index=True)  # e.g., "SPEAKER_001"
    name = Column(String(100), nullable=True, index=True)  # Optional human name
    
    # Enrollment metadata
    first_seen_recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True)
    enrollment_method = Column(String(20), default="manual")  # manual, auto_recording, auto_upload
    enrollment_quality = Column(Float, nullable=True)  # Overall enrollment quality score
    
    # Usage statistics
    total_speaking_time = Column(Float, default=0.0)  # Total speaking time across all recordings
    recordings_count = Column(Integer, default=0)  # Number of recordings this speaker appears in
    segments_count = Column(Integer, default=0)  # Total number of segments
    
    # Speaker recognition settings
    confidence_threshold = Column(Float, default=0.75)  # Minimum confidence for auto-assignment
    is_active = Column(Boolean, default=True)  # Whether to include in recognition
    
    # Compressed average embedding for fast similarity search
    avg_embedding = Column(LargeBinary, nullable=True)  # Compressed numpy array
    embedding_metadata = Column(JSON, nullable=True)  # Quality scores, consistency metrics
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    first_seen_recording = relationship("Recording")
    embeddings = relationship("SpeakerEmbedding", back_populates="speaker", cascade="all, delete-orphan")
    mappings = relationship("SpeakerMapping", back_populates="persistent_speaker", cascade="all, delete-orphan")
    
    def __repr__(self):
        display_name = self.name or self.id
        return f"<PersistentSpeaker(id='{self.id}', name='{display_name}', recordings={self.recordings_count})>"


class SpeakerEmbedding(Base):
    """
    Model for storing multiple embeddings per persistent speaker.
    
    Each persistent speaker can have multiple enrollment samples to improve
    recognition robustness. Stores individual embeddings with quality metrics.
    """
    __tablename__ = "speaker_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    speaker_id = Column(String(20), ForeignKey("persistent_speakers.id"), nullable=False)
    
    # Embedding data
    embedding = Column(LargeBinary, nullable=False)  # Compressed numpy array
    embedding_dim = Column(Integer, default=192)  # ECAPA-TDNN dimension
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)  # 0-1 quality score
    snr_db = Column(Float, nullable=True)  # Signal-to-noise ratio
    duration = Column(Float, nullable=True)  # Source audio duration in seconds
    
    # Source information
    source_recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True)
    source_segment_start = Column(Float, nullable=True)  # Start time in source recording
    source_segment_end = Column(Float, nullable=True)  # End time in source recording
    enrollment_method = Column(String(20), default="manual")  # manual, auto_segment, upload
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    speaker = relationship("PersistentSpeaker", back_populates="embeddings")
    source_recording = relationship("Recording")
    
    def __repr__(self):
        return f"<SpeakerEmbedding(id={self.id}, speaker='{self.speaker_id}', quality={self.quality_score:.3f})>"

# Index for fast embedding lookups
Index('idx_speaker_embeddings_speaker_quality', SpeakerEmbedding.speaker_id, SpeakerEmbedding.quality_score)


class SpeakerMapping(Base):
    """
    Model for mapping session speakers to persistent speakers.
    
    Tracks the assignment of session-based diarization labels (SPEAKER_00, SPEAKER_01)
    to persistent speaker identities for each recording. Enables consistent speaker
    identification across different recording sessions.
    """
    __tablename__ = "speaker_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    
    # Session speaker information (from diarization)
    session_speaker_label = Column(String(20), nullable=False)  # e.g., "SPEAKER_00"
    
    # Persistent speaker assignment
    persistent_speaker_id = Column(String(20), ForeignKey("persistent_speakers.id"), nullable=False)
    
    # Assignment metadata
    assignment_confidence = Column(Float, nullable=True)  # Confidence score for this assignment
    assignment_method = Column(String(20), default="auto")  # auto, manual, reviewed
    similarity_score = Column(Float, nullable=True)  # Best embedding similarity score
    
    # Assignment status
    is_verified = Column(Boolean, default=False)  # Whether assignment has been manually verified
    needs_review = Column(Boolean, default=False)  # Whether assignment needs manual review
    
    # Timestamps
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    recording = relationship("Recording")
    persistent_speaker = relationship("PersistentSpeaker", back_populates="mappings")
    
    def __repr__(self):
        return f"<SpeakerMapping(recording={self.recording_id}, {self.session_speaker_label}→{self.persistent_speaker_id})>"

# Composite index for fast lookup of mappings by recording and session speaker
Index('idx_speaker_mappings_recording_session', SpeakerMapping.recording_id, SpeakerMapping.session_speaker_label)


class SpeakerReviewQueue(Base):
    """
    Model for tracking speaker assignments that need manual review.
    
    When automatic speaker recognition produces ambiguous results (medium confidence),
    assignments are queued for manual review. This table tracks pending reviews
    and user decisions.
    """
    __tablename__ = "speaker_review_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    session_speaker_label = Column(String(20), nullable=False)
    
    # Suggested assignments (can have multiple candidates)
    suggested_assignments = Column(JSON, nullable=True)  # List of {speaker_id, confidence, similarity}
    
    # Review status
    status = Column(String(20), default="pending")  # pending, reviewed, dismissed
    priority = Column(Integer, default=1)  # 1=high, 2=medium, 3=low
    
    # Additional context for review
    segment_count = Column(Integer, nullable=True)  # Number of segments for this speaker
    total_duration = Column(Float, nullable=True)  # Total speaking time
    audio_quality = Column(Float, nullable=True)  # Average audio quality
    
    # Review resolution
    resolved_speaker_id = Column(String(20), ForeignKey("persistent_speakers.id"), nullable=True)
    resolution_method = Column(String(20), nullable=True)  # assigned_existing, created_new, dismissed
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    recording = relationship("Recording")
    resolved_speaker = relationship("PersistentSpeaker")
    
    def __repr__(self):
        return f"<SpeakerReviewQueue(id={self.id}, recording={self.recording_id}, speaker={self.session_speaker_label}, status='{self.status}')>"

# Index for efficient queue processing
Index('idx_review_queue_status_priority', SpeakerReviewQueue.status, SpeakerReviewQueue.priority)


class ProcessingTask(Base):
    """
    Model for tracking Celery background processing tasks.

    Stores task metadata, status, and results for async audio processing.
    Allows frontend to poll for task completion and retrieve results.
    """
    __tablename__ = "processing_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)  # Celery task ID
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True)  # Associated recording

    # Task information
    task_type = Column(String(50), nullable=False)  # 'analyze_upload', 'analyze_vad', etc.
    task_name = Column(String(100), nullable=True)  # Human-readable task name

    # Task status
    status = Column(String(20), default="queued", nullable=False, index=True)  # queued, processing, completed, failed, cancelled
    progress = Column(Integer, default=0)  # 0-100 percentage (optional)

    # Results and errors
    result_data = Column(JSON, nullable=True)  # Final processing results (full AudioAnalysisResult)
    error_message = Column(Text, nullable=True)  # Error message if failed
    traceback = Column(Text, nullable=True)  # Full traceback for debugging

    # Timing metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Processing metadata
    worker_name = Column(String(100), nullable=True)  # Celery worker that processed this task
    retries = Column(Integer, default=0)  # Number of retry attempts

    # Relationships
    recording = relationship("Recording")

    def __repr__(self):
        return f"<ProcessingTask(id={self.id}, task_id='{self.task_id}', status='{self.status}')>"

# Index for efficient task lookup and cleanup
Index('idx_processing_tasks_status_created', ProcessingTask.status, ProcessingTask.created_at)