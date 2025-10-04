"""
Speaker management schemas.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class SpeakerBase(BaseModel):
    """Base speaker model."""
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    organization: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = None


class SpeakerCreate(SpeakerBase):
    """Request model for creating a speaker."""
    pass


class SpeakerUpdate(BaseModel):
    """Request model for updating a speaker."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    organization: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = None


class SpeakerResponse(SpeakerBase):
    """Response model for speaker data."""
    id: str
    embedding_count: int = Field(default=0, description="Number of voice samples")
    created_at: datetime
    last_seen: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SpeakerListResponse(BaseModel):
    """Response model for speaker list."""
    speakers: List[SpeakerResponse]
    total: int
    skip: int = 0
    limit: int = 50
