"""
Recording schemas.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class RecordingBase(BaseModel):
    """Base recording model."""
    filename: str
    original_filename: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None


class RecordingResponse(RecordingBase):
    """Response model for recording data."""
    id: int
    file_path: str
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RecordingListResponse(BaseModel):
    """Response model for recording list."""
    recordings: List[RecordingResponse]
    total: int
    skip: int = 0
    limit: int = 50


class TranscriptSegment(BaseModel):
    """Transcript segment."""
    speaker_id: str
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


class SpeakerSummary(BaseModel):
    """Speaker summary in recording."""
    id: str
    name: str
    speaking_time: float
    segments_count: int
    confidence: float


class RecordingDetailResponse(BaseModel):
    """Detailed recording response with processing results."""
    recording: RecordingResponse
    transcription: Optional[str] = None
    segments: List[TranscriptSegment] = []
    speakers: List[SpeakerSummary] = []
    processing_status: str
    num_speakers: Optional[int] = None
