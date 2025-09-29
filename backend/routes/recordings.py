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
            offset=skip, 
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
        
        # Get speaker data from speaker_segments
        speakers_data = []
        if results:
            for result in results:
                # Get speaker segments for this processing result
                segments = await db_service.speaker_segments.get_segments_by_result(result.id)
                
                # Group segments by speaker to create speaker summary
                speaker_summary = {}
                for segment in segments:
                    speaker_key = segment.speaker_label
                    if speaker_key not in speaker_summary:
                        speaker_summary[speaker_key] = {
                            "name": segment.speaker_label,
                            "confidence": 0,
                            "speaking_time": 0,
                            "segments": 0
                        }
                    
                    speaker_summary[speaker_key]["speaking_time"] += float(segment.duration or 0)
                    speaker_summary[speaker_key]["confidence"] += float(segment.confidence or 0)
                    speaker_summary[speaker_key]["segments"] += 1
                
                # Average confidence and add to speakers_data
                for speaker_name, data in speaker_summary.items():
                    if data["segments"] > 0:
                        data["confidence"] = data["confidence"] / data["segments"]
                    speakers_data.append({
                        "id": speaker_name,
                        "name": data["name"],
                        "confidence": data["confidence"],
                        "speaking_time": data["speaking_time"],
                        "segments": data["segments"]
                    })
        
        # Convert processing results to JSON-serializable format
        processing_results_data = []
        for result in results:
            # Get speaker segments for this result directly
            segments = await db_service.speaker_segments.get_segments_by_result(result.id)
            speaker_segments = []
            for segment in segments:
                speaker_segments.append({
                    "id": segment.id,
                    "start_time": float(segment.start_time) if segment.start_time else None,
                    "end_time": float(segment.end_time) if segment.end_time else None,
                    "duration": float(segment.duration) if segment.duration else None,
                    "speaker_label": segment.speaker_label,
                    "segment_text": segment.segment_text,
                    "confidence": float(segment.confidence) if segment.confidence else None,
                    "speaker_name": segment.speaker_label  # Use speaker_label as the display name
                })
            
            processing_results_data.append({
                "id": result.id,
                "transcription": result.transcription,
                "confidence_score": float(result.confidence_score) if result.confidence_score else None,
                "diarization_json": result.diarization_json,
                "num_speakers": result.num_speakers,
                "emotions_json": result.emotions_json,
                "dominant_emotion": result.dominant_emotion,
                "emotion_confidence": float(result.emotion_confidence) if result.emotion_confidence else None,
                "processing_duration": float(result.processing_duration) if result.processing_duration else None,
                "model_versions": result.model_versions,
                "status": result.status,
                "processed_at": result.processed_at.isoformat() if result.processed_at else None,
                "speaker_segments": speaker_segments
            })

        return {
            "recording": {
                "id": recording.id,
                "filename": recording.filename,
                "original_filename": recording.original_filename,
                "file_path": recording.file_path,
                "file_size": recording.file_size,
                "duration": recording.duration,
                "sample_rate": recording.sample_rate,
                "channels": recording.channels,
                "format": recording.format,
                "created_at": recording.created_at.isoformat() if recording.created_at else None,
                "updated_at": recording.updated_at.isoformat() if recording.updated_at else None
            },
            "processing_results": processing_results_data,
            "speakers": speakers_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording details: {str(e)}")

@router.delete("/{recording_id}")
async def delete_recording(
    recording_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Delete a specific recording"""
    try:
        # Check if recording exists
        recording = await db_service.recordings.get_recording(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Delete the recording
        success = await db_service.recordings.delete_recording(recording_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete recording")
        
        return {"status": "deleted", "recording_id": recording_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")

@router.post("/bulk-delete")
async def bulk_delete_recordings(
    recording_ids: List[int],
    db_service: DatabaseService = Depends(get_database_service)
):
    """Delete multiple recordings at once"""
    try:
        deleted_count = 0
        failed_ids = []
        
        for recording_id in recording_ids:
            try:
                success = await db_service.recordings.delete_recording(recording_id)
                if success:
                    deleted_count += 1
                else:
                    failed_ids.append(recording_id)
            except Exception:
                failed_ids.append(recording_id)
        
        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "failed_ids": failed_ids,
            "total_requested": len(recording_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to bulk delete recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete recordings: {str(e)}")

@router.delete("/cleanup/temporary")
async def cleanup_temporary_recordings(
    db_service: DatabaseService = Depends(get_database_service)
):
    """Clean up temporary recordings (those with temp file paths)"""
    try:
        # This would need to be implemented in the database service
        # For now, return a placeholder response
        return {
            "status": "completed", 
            "message": "Use direct database cleanup for now",
            "deleted_count": 0
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup temporary recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {str(e)}")