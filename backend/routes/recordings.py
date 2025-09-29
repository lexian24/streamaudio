"""
Recording management endpoints.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from database import get_database_service
from database.services import DatabaseService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("")
async def get_recordings(
    skip: int = Query(0, description="Number of recordings to skip"),
    limit: int = Query(50, description="Maximum number of recordings to return"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of all recordings with pagination"""
    try:
        recordings = await db_service.recordings.get_recordings(
            skip=skip, 
            limit=limit
        )
        
        # Convert to JSON-serializable format
        recordings_data = []
        for recording in recordings:
            recordings_data.append({
                "id": recording.id,
                "filename": recording.filename,
                "original_filename": recording.original_filename,
                "file_path": recording.file_path,
                "file_size": recording.file_size,
                "duration": recording.duration,
                "sample_rate": recording.sample_rate,
                "channels": recording.channels,
                "format": recording.format,
                "created_at": recording.created_at.isoformat() if recording.created_at else None
            })
        
        return {
            "recordings": recordings_data,
            "total": len(recordings_data),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recordings: {str(e)}")

@router.get("/{recording_id}")
async def get_recording_details(
    recording_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get detailed information about a specific recording"""
    try:
        # Get recording details
        recording = await db_service.recordings.get_recording(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Get processing results
        results = await db_service.processing_results.get_results_by_recording(recording_id)
        
        # Get speaker data if available
        speakers_data = []
        if results:
            for result in results:
                speakers = await db_service.speakers.get_speakers_by_result(result.id)
                for speaker in speakers:
                    segments = await db_service.speaker_segments.get_segments_by_speaker(speaker.id)
                    speakers_data.append({
                        "id": speaker.id,
                        "name": speaker.name,
                        "confidence": float(speaker.confidence) if speaker.confidence else None,
                        "speaking_time": float(speaker.total_speaking_time) if speaker.total_speaking_time else None,
                        "segments": len(segments)
                    })
        
        return {
            "recording": {
                "id": recording.id,
                "filename": recording.filename,
                "original_filename": recording.original_filename,
                "duration": recording.duration,
                "created_at": recording.created_at.isoformat() if recording.created_at else None
            },
            "processing_results": len(results),
            "speakers": speakers_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording details: {str(e)}")

# Additional recording endpoints would be added here following the same pattern