"""
Speaker management endpoints.
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from database import get_database_service
from database.services import DatabaseService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("")
async def create_speaker(
    speaker_data: Dict[str, Any],
    db_service: DatabaseService = Depends(get_database_service)
):
    """Create a new speaker"""
    try:
        speaker = await db_service.speakers.create_speaker(
            name=speaker_data.get("name"),
            confidence=speaker_data.get("confidence", 0.0)
        )
        
        return {
            "id": speaker.id,
            "name": speaker.name,
            "confidence": float(speaker.confidence) if speaker.confidence else None,
            "created_at": speaker.created_at.isoformat() if speaker.created_at else None
        }
        
    except Exception as e:
        logger.error(f"Failed to create speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create speaker: {str(e)}")

@router.get("")
async def get_speakers(
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of all speakers"""
    try:
        speakers = await db_service.speakers.get_all_speakers()
        
        speakers_data = []
        for speaker in speakers:
            speakers_data.append({
                "id": speaker.id,
                "name": speaker.name,
                "confidence": float(speaker.confidence) if speaker.confidence else None,
                "total_speaking_time": float(speaker.total_speaking_time) if speaker.total_speaking_time else None,
                "created_at": speaker.created_at.isoformat() if speaker.created_at else None
            })
        
        return {"speakers": speakers_data, "total": len(speakers_data)}
        
    except Exception as e:
        logger.error(f"Failed to retrieve speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve speakers: {str(e)}")

@router.get("/{speaker_id}")
async def get_speaker(
    speaker_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get detailed information about a specific speaker"""
    try:
        speaker = await db_service.speakers.get_speaker(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        return {
            "id": speaker.id,
            "name": speaker.name,
            "confidence": float(speaker.confidence) if speaker.confidence else None,
            "total_speaking_time": float(speaker.total_speaking_time) if speaker.total_speaking_time else None,
            "created_at": speaker.created_at.isoformat() if speaker.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speaker: {str(e)}")

@router.delete("/{speaker_id}")
async def delete_speaker(
    speaker_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Delete a speaker"""
    try:
        success = await db_service.speakers.delete_speaker(speaker_id)
        if not success:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        return {"status": "deleted", "speaker_id": speaker_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete speaker: {str(e)}")

# Additional speaker endpoints (enroll, etc.) would be added here