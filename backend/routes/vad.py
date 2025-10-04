"""
Voice Activity Detection (VAD) endpoints.
"""

import logging
import tempfile
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import librosa

from database import get_database_service
from database.services import DatabaseService
from .dependencies import get_auto_recorder
from tasks.audio_tasks import process_vad_recording

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/start")
async def start_vad_monitoring():
    """Start automatic voice activity monitoring"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    try:
        auto_recorder.start_monitoring()
        return {"status": "started", "message": "VAD monitoring started - ready for audio stream"}
    except Exception as e:
        logger.error(f"Failed to start VAD monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/stop")
async def stop_vad_monitoring():
    """Stop automatic voice activity monitoring"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    try:
        auto_recorder.stop_monitoring()
        return {"status": "stopped", "message": "VAD monitoring stopped"}
    except Exception as e:
        logger.error(f"Failed to stop VAD monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/status")
async def get_vad_status():
    """Get current VAD monitoring status"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    return auto_recorder.get_status()

@router.get("/recordings")
async def get_recordings():
    """Get list of recorded meetings"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    recordings = auto_recorder.get_recordings()
    return {"recordings": recordings, "total": len(recordings)}

@router.delete("/recordings/{filename}")
async def delete_recording(filename: str):
    """Delete a specific recording"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    success = auto_recorder.delete_recording(filename)
    if success:
        return {"status": "deleted", "filename": filename}
    else:
        raise HTTPException(status_code=404, detail="Recording not found")

@router.post("/upload-recording")
async def upload_recording(
    file: UploadFile = File(...),
    db_service: DatabaseService = Depends(get_database_service)
) -> Dict[str, Any]:
    """
    Process VAD recording using the same pipeline as upload/analyze endpoint
    Returns analysis results immediately like the upload flow
    """
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    # Validate file format
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload .wav, .mp3, .m4a, .flac, .ogg, or .webm"
        )
    
    # Check file size (50MB limit)
    max_size = 50 * 1024 * 1024
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
    
    try:
        start_time = time.time()
        
        # Save uploaded file permanently with timestamp - use same logic as analyze endpoint
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        original_ext = Path(file.filename).suffix.lower()
        
        # Convert all uploads to WAV for consistency
        if original_ext in ['.wav']:
            temp_filename = f"vad_{timestamp}{original_ext}"
            final_filename = f"vad_{timestamp}.wav"
            needs_conversion = False
        else:
            temp_filename = f"vad_{timestamp}{original_ext}"
            final_filename = f"vad_{timestamp}.wav"
            needs_conversion = True
        
        # Create recordings directory
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        
        temp_path = recordings_dir / temp_filename
        final_path = recordings_dir / final_filename
        
        # Save temporary file
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        try:
            # Convert to WAV if needed
            if needs_conversion:
                logger.info(f"Converting {original_ext} to WAV...")
                subprocess.run([
                    'ffmpeg', '-i', str(temp_path), 
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ar', '16000',          # 16kHz sample rate
                    '-ac', '1',              # Mono
                    str(final_path)
                ], check=True, capture_output=True)
                
                # Use the converted file for processing
                processing_path = str(final_path)
                stored_filename = final_filename
            else:
                # Use original WAV file
                processing_path = str(temp_path)
                stored_filename = temp_filename
                final_path = temp_path
            
            # Get file size and audio metadata
            file_size = final_path.stat().st_size
            try:
                duration = librosa.get_duration(path=processing_path)
                sample_rate = librosa.get_samplerate(processing_path)
            except Exception:
                duration = None
                sample_rate = 16000
            
            # Create a permanent recording entry
            recording = await db_service.recordings.create_recording(
                filename=stored_filename,
                original_filename=file.filename,
                file_path=str(final_path),
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
                format="wav"
            )

            logger.info(f"✅ VAD recording saved: {recording.id}, submitting to Celery for async processing...")

            # Submit Celery task for async processing
            task = process_vad_recording.delay(
                recording_id=recording.id,
                audio_path=processing_path,
                auto_enroll_new_speakers=True
            )

            # Create task record in database
            task_record = await db_service.processing_tasks.create_task(
                task_id=task.id,
                task_type='analyze_vad',
                recording_id=recording.id,
                task_name=f"Process VAD: {file.filename}"
            )

            logger.info(f"🚀 VAD task submitted: {task.id}")

            # Return task info immediately (non-blocking)
            return {
                "status": "queued",
                "task_id": task.id,
                "recording_id": recording.id,
                "filename": file.filename,
                "message": "VAD recording uploaded successfully. Processing in background.",
                "poll_url": f"/api/tasks/{task.id}"
            }

        finally:
            # Clean up temporary file only if conversion happened
            if needs_conversion and temp_path != final_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"VAD recording upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
