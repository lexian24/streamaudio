"""
Audio analysis endpoints.

Modified to use Celery for async background processing.
"""

import os
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import librosa

from database.services import DatabaseService
from database import get_database_service
from tasks.audio_tasks import process_audio_file

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    db_service: DatabaseService = Depends(get_database_service)
) -> Dict[str, Any]:
    """
    Submit uploaded audio file for async processing via Celery.
    Returns task_id immediately for status polling.

    Args:
        file: Uploaded audio file
        db_service: Database service dependency

    Returns:
        Task information with task_id for polling status
    """
    # Validate file
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
        # Save uploaded file permanently with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        original_ext = Path(file.filename).suffix.lower()

        # Convert all uploads to WAV for consistency
        if original_ext in ['.wav']:
            temp_filename = f"upload_{timestamp}{original_ext}"
            final_filename = f"upload_{timestamp}.wav"
            needs_conversion = False
        else:
            temp_filename = f"upload_{timestamp}{original_ext}"
            final_filename = f"upload_{timestamp}.wav"
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

            logger.info(f"✅ Recording saved: {recording.id}, submitting to Celery for async processing...")

            # Submit Celery task for async processing
            task = process_audio_file.delay(
                recording_id=recording.id,
                audio_path=processing_path,
                auto_enroll_new_speakers=True
            )

            # Create task record in database
            task_record = await db_service.processing_tasks.create_task(
                task_id=task.id,
                task_type='analyze_upload',
                recording_id=recording.id,
                task_name=f"Process upload: {file.filename}"
            )

            logger.info(f"🚀 Task submitted: {task.id}")

            # Return task info immediately (non-blocking)
            return {
                "status": "queued",
                "task_id": task.id,
                "recording_id": recording.id,
                "filename": file.filename,
                "message": "Audio file uploaded successfully. Processing in background.",
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
        logger.error(f"Audio upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "status": "models loaded on first task execution",
        "message": "Models are loaded by Celery workers when first task is processed"
    }
