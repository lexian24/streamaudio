"""
Pydantic schemas for request/response validation.

This module contains all data transfer objects (DTOs) used for
API request validation and response serialization.
"""

from .audio import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    TaskStatusResponse,
    AudioProcessingResult,
)
from .speaker import (
    SpeakerBase,
    SpeakerCreate,
    SpeakerResponse,
    SpeakerListResponse,
)
from .recording import (
    RecordingResponse,
    RecordingListResponse,
    RecordingDetailResponse,
)
from .common import (
    SuccessResponse,
    ErrorResponse,
    PaginationParams,
)

__all__ = [
    # Audio
    "AudioAnalysisRequest",
    "AudioAnalysisResponse",
    "TaskStatusResponse",
    "AudioProcessingResult",
    # Speaker
    "SpeakerBase",
    "SpeakerCreate",
    "SpeakerResponse",
    "SpeakerListResponse",
    # Recording
    "RecordingResponse",
    "RecordingListResponse",
    "RecordingDetailResponse",
    # Common
    "SuccessResponse",
    "ErrorResponse",
    "PaginationParams",
]
