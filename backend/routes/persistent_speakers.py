"""
Persistent speaker management endpoints.
"""

import logging
from typing import List, Dict, Any
import os
import tempfile

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from typing import Optional

from database import get_database_service
from database.services import DatabaseService
from services.persistent_speaker_manager import PersistentSpeakerManager
from services.speaker_identification import SpeakerIdentifier

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
                "total_segments": speaker.segments_count,
                "total_recordings": speaker.recordings_count,
                "embedding_count": embedding_count,
                "first_seen": speaker.created_at.isoformat() if speaker.created_at else None,
                "last_seen": speaker.last_seen_at.isoformat() if speaker.last_seen_at else None,
                "is_active": speaker.is_active
            })
        
        return {"speakers": speakers_data, "total": len(speakers_data)}
        
    except Exception as e:
        logger.error(f"Failed to retrieve persistent speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve persistent speakers: {str(e)}")

@router.post("")
async def create_persistent_speaker(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    speaker_name: Optional[str] = Query(None),
    db_service: DatabaseService = Depends(get_database_service)
):
    """Enroll a new persistent speaker with audio files"""
    try:
        # Accept name from either form data or query parameter
        speaker_name_value = name or speaker_name
        if not speaker_name_value:
            raise HTTPException(status_code=422, detail="Speaker name is required (provide as 'name' form field or 'speaker_name' query parameter)")

        logger.info(f"Enrolling speaker '{speaker_name_value}' with {len(files)} audio files")

        # Save uploaded files temporarily
        temp_files = []
        try:
            for file in files:
                suffix = os.path.splitext(file.filename)[1] if file.filename else '.wav'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
                logger.info(f"Saved temp file: {temp_file.name}")

            # Initialize speaker identifier with consistent model directory
            speaker_identifier = SpeakerIdentifier()

            # Create speaker manager
            speaker_manager = PersistentSpeakerManager(db_service, speaker_identifier)

            # Enroll speaker
            result = await speaker_manager.enroll_speaker_from_files(
                speaker_name=speaker_name_value,
                audio_files=temp_files
            )

            if result.get('status') == 'enrolled':
                return {
                    "id": result['speaker_id'],
                    "name": speaker_name_value,
                    "embeddings_count": result['embeddings_count'],
                    "quality_score": result['quality_score'],
                    "created_at": result.get('created_at')
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('message', 'Enrollment failed'))

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enroll persistent speaker: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to enroll persistent speaker: {str(e)}")

@router.delete("/{speaker_id}")
async def delete_persistent_speaker(
    speaker_id: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Delete (deactivate) a persistent speaker"""
    try:
        success = await db_service.persistent_speakers.delete_persistent_speaker(speaker_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persistent speaker not found")

        return {"status": "deleted", "speaker_id": speaker_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persistent speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete persistent speaker: {str(e)}")

# Additional persistent speaker endpoints would be added here