"""
Common schemas used across the API.
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    details: Optional[str] = None
    code: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of items to return")
