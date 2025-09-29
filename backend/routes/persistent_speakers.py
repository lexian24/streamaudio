"""
Persistent speaker management endpoints.
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from database import get_database_service
from database.services import DatabaseService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("")
async def get_persistent_speakers(
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of all persistent speakers"""
    try:
        speakers = await db_service.persistent_speakers.get_all_persistent_speakers(active_only=True)
        
        speakers_data = []
        for speaker in speakers:
            # Get embedding count
            embeddings = await db_service.speaker_embeddings.get_embeddings_as_numpy(speaker.id)
            embedding_count = len(embeddings) if embeddings else 0
            
            speakers_data.append({
                "id": speaker.id,
                "name": speaker.name,
                "confidence_threshold": float(speaker.confidence_threshold) if speaker.confidence_threshold else None,
                "total_speaking_time": float(speaker.total_speaking_time) if speaker.total_speaking_time else None,
                "total_segments": speaker.total_segments,
                "total_recordings": speaker.total_recordings,
                "embedding_count": embedding_count,
                "first_seen": speaker.first_seen_at.isoformat() if speaker.first_seen_at else None,
                "last_seen": speaker.last_seen_at.isoformat() if speaker.last_seen_at else None,
                "is_active": speaker.is_active
            })
        
        return {"speakers": speakers_data, "total": len(speakers_data)}
        
    except Exception as e:
        logger.error(f"Failed to retrieve persistent speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve persistent speakers: {str(e)}")

@router.post("")
async def create_persistent_speaker(
    speaker_data: Dict[str, Any],
    db_service: DatabaseService = Depends(get_database_service)
):
    """Create a new persistent speaker"""
    try:
        speaker = await db_service.persistent_speakers.create_persistent_speaker(
            name=speaker_data.get("name"),
            embeddings=speaker_data.get("embeddings", []),
            embedding_metadata=speaker_data.get("embedding_metadata", {}),
            first_seen_recording_id=speaker_data.get("first_seen_recording_id"),
            enrollment_method=speaker_data.get("enrollment_method", "manual")
        )
        
        return {
            "id": speaker.id,
            "name": speaker.name,
            "confidence_threshold": float(speaker.confidence_threshold) if speaker.confidence_threshold else None,
            "created_at": speaker.created_at.isoformat() if speaker.created_at else None
        }
        
    except Exception as e:
        logger.error(f"Failed to create persistent speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create persistent speaker: {str(e)}")

@router.delete("/{speaker_id}")
async def delete_persistent_speaker(
    speaker_id: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Delete (deactivate) a persistent speaker"""
    try:
        success = await db_service.persistent_speakers.deactivate_speaker(speaker_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persistent speaker not found")
        
        return {"status": "deactivated", "speaker_id": speaker_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persistent speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete persistent speaker: {str(e)}")

# Additional persistent speaker endpoints would be added here