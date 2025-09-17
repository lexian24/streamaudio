"""
Database models for StreamAudio application.

This module defines the SQLAlchemy models for storing audio recordings,
processing results, speaker information, and speaker segments.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON
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