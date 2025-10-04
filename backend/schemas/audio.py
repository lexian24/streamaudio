"""
Audio analysis request/response schemas.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis."""
    auto_enroll_new_speakers: bool = Field(
        default=True,
        description="Automatically enroll new speakers found in the audio"
    )


class AudioAnalysisResponse(BaseModel):
    """Response model for audio analysis submission."""
    task_id: str = Field(..., description="Celery task ID for tracking")
    status: str = Field(default="pending", description="Initial task status")
    message: str = Field(default="Audio processing started", description="Status message")


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")


class SpeakerSegment(BaseModel):
    """Speaker segment in transcription."""
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None


class AudioProcessingResult(BaseModel):
    """Complete audio processing result."""
    recording_id: int
    duration: float
    transcription: str
    segments: List[SpeakerSegment]
    speakers: List[str]
    num_speakers: int
    processing_duration: float
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
