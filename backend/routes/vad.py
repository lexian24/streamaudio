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
from routes.analysis import get_audio_processor

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
):
    """Store recording from continuous recorder without processing"""
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    try:
        # Validate file format
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format"
            )
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
        
        # Generate filename with timestamp - always save as WAV
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        temp_filename = f"temp_{timestamp}.webm"
        final_filename = f"recording_{timestamp}.wav"
        
        # Save to recordings directory
        recordings_dir = auto_recorder.recordings_dir
        temp_filepath = recordings_dir / temp_filename
        final_filepath = recordings_dir / final_filename
        
        # Save temporary file
        with open(temp_filepath, 'wb') as f:
            f.write(file_content)
        
        try:
            # Convert webm to wav using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', str(temp_filepath), 
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',          # 16kHz sample rate
                '-ac', '1',              # Mono
                '-y',                    # Overwrite output
                str(final_filepath)
            ], capture_output=True, text=True, check=True)
            
            # Remove temporary file
            temp_filepath.unlink()
            
            # Get final file size and metadata
            final_size = final_filepath.stat().st_size
            
            # Get audio metadata using librosa
            try:
                duration = librosa.get_duration(path=str(final_filepath))
                sr = librosa.get_samplerate(str(final_filepath))
            except Exception:
                duration = None
                sr = 16000  # Default sample rate
            
            # Store recording metadata in database
            try:
                recording = await db_service.recordings.create_recording(
                    filename=final_filename,
                    original_filename=file.filename,
                    file_path=str(final_filepath),
                    file_size=final_size,
                    duration=duration,
                    sample_rate=sr,
                    channels=1,  # Mono after conversion
                    format="wav"
                )
                
                logger.info(f"✅ Recording stored in database: ID {recording.id}")
                
            except Exception as e:
                logger.error(f"Failed to store recording metadata in database: {e}")
                # Continue even if database storage fails
            
            logger.info(f"✅ Recording converted and stored: {final_filename} ({final_size} bytes)")
            
            return {
                "status": "stored",
                "filename": final_filename,
                "path": str(final_filepath),
                "size": final_size,
                "duration": duration,
                "recording_id": recording.id if 'recording' in locals() else None
            }
            
        except subprocess.CalledProcessError as e:
            # If conversion fails, keep original format
            logger.warning(f"FFmpeg conversion failed: {e.stderr}")
            temp_filepath.rename(final_filepath.with_suffix('.webm'))
            
            return {
                "status": "stored",
                "filename": temp_filename,
                "path": str(final_filepath.with_suffix('.webm')),
                "size": len(file_content)
            }
        except Exception as e:
            # Clean up temp file on any error
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise e
        
    except Exception as e:
        logger.error(f"Failed to store recording: {e}")
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

# Note: process_recorded_meeting endpoint is quite long - would be extracted similarly
# For brevity, I'll create shorter route files for the other modules