"""
FastAudio - Simple Audio Analysis API
Uses Whisper for transcription, pyannote for speaker diarization, 
and a lightweight emotion recognition model
"""
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import numpy as np
import base64
from datetime import datetime
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.audio_processor import AudioProcessor
from services.enhanced_audio_processor import EnhancedAudioProcessor
from services.persistent_speaker_manager import PersistentSpeakerManager
from services.auto_recorder import AutoRecorder
from database import init_database, get_database_service, get_recording_service
from database.services import DatabaseService
from fastapi import Depends

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce debug noise from specific modules
logging.getLogger("services.auto_recorder").setLevel(logging.INFO)
logging.getLogger("services.voice_activity_detection").setLevel(logging.INFO)

# Global processor and recorder
audio_processor = None
auto_recorder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    global audio_processor, auto_recorder
    
    # Startup
    try:
        logger.info("Starting FastAudio server...")
        
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        logger.info("Database initialized successfully!")
        
        logger.info("AI models will be loaded on first request (lazy loading)")
        
        # Initialize auto recorder
        auto_recorder = AutoRecorder()
        
        logger.info("FastAudio server ready!")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down FastAudio server...")
    if auto_recorder:
        auto_recorder.stop_monitoring()
    logger.info("FastAudio server shutdown complete")


app = FastAPI(
    title="FastAudio API", 
    description="Audio Analysis with Whisper + pyannote",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_audio_processor():
    """Lazy load the enhanced audio processor with persistent speaker identification"""
    global audio_processor
    if audio_processor is None:
        logger.info("Loading AI models for file processing with speaker identification...")
        # The enhanced processor will get database services as needed in async context
        audio_processor = EnhancedAudioProcessor(db_service=None)
        logger.info("Enhanced processor ready!")
    return audio_processor


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAudio", 
        "models": {
            "audio_processor": "loaded" if audio_processor else "lazy",
            "auto_recorder": "loaded" if auto_recorder else "not_initialized"
        }
    }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    db_service: DatabaseService = Depends(get_database_service)
) -> Dict[str, Any]:
    """
    Analyze uploaded audio file for speakers, transcription, and emotions
    """
    processor = get_audio_processor()
    
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
        start_time = time.time()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
        
        try:
            # Process audio with persistent speaker identification
            logger.info(f"Processing audio with speaker identification: {file.filename}")
            
            # Create a temporary recording entry for tracking  
            recording = await db_service.recordings.create_recording(
                filename=file.filename,
                file_path=temp_path,
                duration=0  # Will be updated after processing
            )
            
            # Use enhanced processor with persistent speaker identification
            result = await processor.process_audio_with_persistent_speakers(
                temp_path, 
                recording.id,
                db_service,
                auto_enroll_new_speakers=True
            )
            
            # Add metadata
            result.update({
                "filename": file.filename,
                "processing_time": time.time() - start_time,
                "status": "completed"
            })
            
            logger.info(f"Audio processing completed in {result['processing_time']:.2f}s")
            return result
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    result = {"status": "lazy loading"}
    if audio_processor:
        result["audio_processing"] = audio_processor.get_model_info()
    return result

# VAD endpoints for automatic recording
@app.post("/vad/start")
async def start_vad_monitoring():
    """Start automatic voice activity monitoring"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    try:
        auto_recorder.start_monitoring()
        return {"status": "started", "message": "VAD monitoring started - ready for audio stream"}
    except Exception as e:
        logger.error(f"Failed to start VAD monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@app.post("/vad/stop")
async def stop_vad_monitoring():
    """Stop automatic voice activity monitoring"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    try:
        auto_recorder.stop_monitoring()
        return {"status": "stopped", "message": "VAD monitoring stopped"}
    except Exception as e:
        logger.error(f"Failed to stop VAD monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@app.get("/vad/status")
async def get_vad_status():
    """Get current VAD monitoring status"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    return auto_recorder.get_status()

@app.get("/vad/recordings")
async def get_recordings():
    """Get list of recorded meetings"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    recordings = auto_recorder.get_recordings()
    return {"recordings": recordings, "total": len(recordings)}

@app.post("/vad/upload-recording")
async def upload_recording(
    file: UploadFile = File(...),
    db_service = Depends(get_database_service)
):
    """Store recording from continuous recorder without processing"""
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
            import librosa
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

@app.delete("/vad/recordings/{filename}")
async def delete_recording(filename: str):
    """Delete a specific recording"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    success = auto_recorder.delete_recording(filename)
    if success:
        return {"status": "deleted", "filename": filename}
    else:
        raise HTTPException(status_code=404, detail="Recording not found")

@app.post("/vad/process-recording/{filename}")
async def process_recorded_meeting(
    filename: str,
    db_service = Depends(get_database_service)
):
    """Process a recorded meeting through the audio pipeline"""
    if not auto_recorder:
        raise HTTPException(status_code=500, detail="Auto recorder not initialized")
    
    processor = get_audio_processor()
    
    # Find the recording file
    recordings = auto_recorder.get_recordings()
    recording = next((r for r in recordings if r["filename"] == filename), None)
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    try:
        start_time = time.time()
        
        # Find recording in database by filename
        recordings_list = await db_service.recordings.get_recordings(limit=100)
        db_recording = next((r for r in recordings_list if r.filename == filename), None)
        
        logger.info(f"DEBUG: Looking for recording with filename: {filename}")
        logger.info(f"DEBUG: Found db_recording: {db_recording.id if db_recording else None}")
        
        # Get enrolled speakers for identification
        enrolled_speakers = await db_service.speakers.get_speakers(active_only=True)
        
        # Process the recorded audio with speaker identification
        logger.info(f"Processing recorded meeting: {filename}")
        if enrolled_speakers:
            logger.info(f"Using speaker identification with {len(enrolled_speakers)} enrolled speakers")
            result = processor.process_audio_with_identification(recording["path"], enrolled_speakers)
        else:
            logger.info("No enrolled speakers - using standard processing")
            result = processor.process_audio(recording["path"])
        
        processing_time = time.time() - start_time
        
        # Store processing results in database if we have the recording
        if db_recording:
            try:
                # Extract data from the audio processor result
                segments = result.get("segments", [])
                speakers = result.get("speakers", [])
                
                # Build transcription from all segments
                transcription_parts = []
                emotions_list = []
                total_confidence = 0
                confidence_count = 0
                
                for segment in segments:
                    if segment.get("text"):
                        transcription_parts.append(segment["text"])
                    if segment.get("emotion"):
                        emotions_list.append({
                            "emotion": segment["emotion"],
                            "confidence": segment.get("emotion_confidence", 0),
                            "speaker_id": segment["speaker_id"],
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"]
                        })
                    if segment.get("emotion_confidence"):
                        total_confidence += segment["emotion_confidence"]
                        confidence_count += 1
                
                # Create combined transcription
                full_transcription = " ".join(transcription_parts) if transcription_parts else None
                
                # Calculate average confidence and dominant emotion
                avg_confidence = total_confidence / confidence_count if confidence_count > 0 else None
                dominant_emotion = None
                emotion_confidence = None
                
                if emotions_list:
                    # Find most confident emotion
                    best_emotion = max(emotions_list, key=lambda x: x["confidence"])
                    dominant_emotion = best_emotion["emotion"]
                    emotion_confidence = best_emotion["confidence"]
                
                # Debug logging
                logger.info(f"DEBUG: full_transcription = {full_transcription}")
                logger.info(f"DEBUG: avg_confidence = {avg_confidence}")
                logger.info(f"DEBUG: dominant_emotion = {dominant_emotion}")
                logger.info(f"DEBUG: speakers count = {len(speakers)}")
                logger.info(f"DEBUG: segments count = {len(segments)}")
                
                # Create processing result
                processing_result = await db_service.processing_results.create_processing_result(
                    recording_id=db_recording.id,
                    transcription=full_transcription,
                    confidence_score=avg_confidence,
                    diarization_data={
                        "speakers": speakers,
                        "total_speakers": result.get("total_speakers", 0)
                    },
                    emotions_data={
                        "emotions": emotions_list,
                        "dominant_emotion": dominant_emotion,
                        "confidence": emotion_confidence
                    } if emotions_list else None,
                    model_versions={
                        "whisper_model": "base",
                        "diarization_model": "pyannote/speaker-diarization-3.1",
                        "emotion_model": "custom"
                    }
                )
                
                # Extract and create speaker segments from the new data structure
                if segments:
                    segments_data = []
                    for segment in segments:
                        # Use identified speaker name if available, otherwise use diarization label
                        speaker_name = segment.get("identified_speaker_name")
                        display_label = speaker_name if speaker_name else segment["speaker_id"]
                        
                        segments_data.append({
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "speaker_label": display_label,
                            "speaker_id": segment.get("identified_speaker_id"),  # Link to Speaker table if identified
                            "text": segment.get("text", ""),
                            "confidence": segment.get("identification_confidence") or segment.get("emotion_confidence"),
                            "speaker_name": speaker_name  # Store identified name separately
                        })
                    
                    await db_service.speaker_segments.create_speaker_segments(
                        processing_result.id,
                        segments_data
                    )
                
                logger.info(f"✅ Processing results stored in database: ID {processing_result.id}")
                result["processing_result_id"] = processing_result.id
                
            except Exception as e:
                logger.error(f"Failed to store processing results in database: {e}")
                # Continue even if database storage fails
        
        # Add metadata
        result.update({
            "filename": filename,
            "processing_time": processing_time,
            "status": "completed",
            "original_duration": recording["duration"],
            "recording_id": db_recording.id if db_recording else None
        })
        
        logger.info(f"Meeting processing completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Meeting processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Database API endpoints
@app.get("/api/recordings")
async def get_all_recordings(
    limit: int = 50,
    offset: int = 0,
    format_filter: str = None,
    db_service = Depends(get_database_service)
):
    """Get all recordings with pagination"""
    try:
        recordings = await db_service.recordings.get_recordings(
            limit=limit,
            offset=offset,
            format_filter=format_filter
        )
        
        # Convert to dict format
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
                "created_at": recording.created_at.isoformat() if recording.created_at else None,
                "updated_at": recording.updated_at.isoformat() if recording.updated_at else None
            })
        
        return {
            "recordings": recordings_data,
            "total": len(recordings_data),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recordings: {str(e)}")

@app.get("/api/recordings/{recording_id}")
async def get_recording(
    recording_id: int,
    db_service = Depends(get_database_service)
):
    """Get a specific recording with its processing results"""
    try:
        recording = await db_service.recordings.get_recording(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Get processing results for this recording
        results = await db_service.processing_results.get_results_by_recording(recording_id)
        
        recording_data = {
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
            "updated_at": recording.updated_at.isoformat() if recording.updated_at else None,
            "processing_results": []
        }
        
        # Add processing results
        for result in results:
            segments = await db_service.speaker_segments.get_segments_by_result(result.id)
            
            recording_data["processing_results"].append({
                "id": result.id,
                "transcription": result.transcription,
                "confidence_score": result.confidence_score,
                "diarization_json": result.diarization_json,
                "num_speakers": result.num_speakers,
                "emotions_json": result.emotions_json,
                "dominant_emotion": result.dominant_emotion,
                "emotion_confidence": result.emotion_confidence,
                "processing_duration": result.processing_duration,
                "model_versions": result.model_versions,
                "status": result.status,
                "processed_at": result.processed_at.isoformat() if result.processed_at else None,
                "speaker_segments": [
                    {
                        "id": seg.id,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "duration": seg.duration,
                        "speaker_label": seg.speaker_label,
                        "segment_text": seg.segment_text,
                        "confidence": seg.confidence,
                        "speaker_name": seg.speaker.name if seg.speaker else None
                    }
                    for seg in segments
                ]
            })
        
        return recording_data
        
    except Exception as e:
        logger.error(f"Failed to get recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording: {str(e)}")

@app.get("/api/processing-results")
async def get_processing_results(
    limit: int = 50,
    offset: int = 0,
    db_service = Depends(get_database_service)
):
    """Get recent processing results"""
    try:
        # This would require adding a method to get all results
        # For now, we'll return empty as we focus on recording-based queries
        return {
            "message": "Use /api/recordings/{recording_id} to get processing results for a specific recording",
            "endpoint": "/api/recordings/{recording_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing results: {str(e)}")

# Speaker Management API endpoints
speaker_identifier = None

def get_speaker_identifier():
    """Get or initialize speaker identifier."""
    global speaker_identifier
    if speaker_identifier is None:
        from services.speaker_identification import SpeakerIdentifier
        speaker_identifier = SpeakerIdentifier()
    return speaker_identifier

@app.post("/api/speakers")
async def create_speaker(
    speaker_name: str,
    files: list[UploadFile] = File(...),
    db_service=Depends(get_database_service)
):
    """
    Create a new speaker profile with enrollment samples.
    Requires 2-5 audio files for robust enrollment.
    """
    try:
        # Validate input
        if len(files) < 2 or len(files) > 5:
            raise HTTPException(
                status_code=400, 
                detail="Please provide 2-5 audio files for speaker enrollment"
            )
        
        # Check if speaker already exists
        existing_speakers = await db_service.speakers.get_speakers()
        if any(s.name.lower() == speaker_name.lower() for s in existing_speakers):
            raise HTTPException(
                status_code=400,
                detail=f"Speaker '{speaker_name}' already exists"
            )
        
        # Save uploaded files temporarily
        temp_files = []
        try:
            for i, file in enumerate(files):
                if not file.content_type.startswith('audio/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} is not an audio file"
                    )
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f"_enrollment_{i}.wav"
                )
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # Enroll speaker using speaker identifier
            identifier = get_speaker_identifier()
            enrollment_result = identifier.enroll_speaker(speaker_name, temp_files)
            
            # Create speaker in database
            speaker = await db_service.speakers.create_speaker(
                name=speaker_name,
                embeddings_data={
                    'embeddings': [emb.tolist() for emb in enrollment_result['embeddings']],
                    'mean_embedding': enrollment_result['mean_embedding'].tolist(),
                    'std_embedding': enrollment_result['std_embedding'].tolist(),
                    'enrollment_threshold': enrollment_result['enrollment_threshold'],
                    'avg_quality': enrollment_result['avg_quality'],
                    'consistency_score': enrollment_result['consistency_score']
                },
                confidence_threshold=enrollment_result['enrollment_threshold']
            )
            
            logger.info(f"✅ Speaker '{speaker_name}' enrolled with ID {speaker.id}")
            
            return {
                "speaker_id": speaker.id,
                "speaker_name": speaker_name,
                "num_samples": enrollment_result['num_samples'],
                "avg_quality": enrollment_result['avg_quality'],
                "consistency_score": enrollment_result['consistency_score'],
                "threshold": enrollment_result['enrollment_threshold'],
                "status": "enrolled"
            }
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speaker enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@app.get("/api/speakers")
async def get_speakers(db_service=Depends(get_database_service)):
    """Get list of all enrolled speakers."""
    try:
        speakers = await db_service.speakers.get_speakers()
        return {
            "speakers": [
                {
                    "id": speaker.id,
                    "name": speaker.name,
                    "email": speaker.email,
                    "num_enrollments": speaker.num_enrollments,
                    "total_speaking_time": speaker.total_speaking_time,
                    "confidence_threshold": speaker.confidence_threshold,
                    "is_active": speaker.is_active,
                    "created_at": speaker.created_at.isoformat() if speaker.created_at else None
                }
                for speaker in speakers
            ],
            "total": len(speakers)
        }
    except Exception as e:
        logger.error(f"Failed to get speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speakers: {str(e)}")

@app.get("/api/speakers/{speaker_id}")
async def get_speaker(speaker_id: int, db_service=Depends(get_database_service)):
    """Get detailed information about a specific speaker."""
    try:
        speaker = await db_service.speakers.get_speaker(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        return {
            "id": speaker.id,
            "name": speaker.name,
            "email": speaker.email,
            "num_enrollments": speaker.num_enrollments,
            "total_speaking_time": speaker.total_speaking_time,
            "confidence_threshold": speaker.confidence_threshold,
            "is_active": speaker.is_active,
            "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
            "updated_at": speaker.updated_at.isoformat() if speaker.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker {speaker_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speaker: {str(e)}")

@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int, db_service=Depends(get_database_service)):
    """Delete a speaker profile."""
    try:
        speaker = await db_service.speakers.get_speaker(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        # TODO: Add cascade delete for speaker segments
        success = await db_service.speakers.delete_speaker(speaker_id)
        
        if success:
            logger.info(f"✅ Speaker '{speaker.name}' (ID: {speaker_id}) deleted")
            return {"message": f"Speaker '{speaker.name}' deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete speaker")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker {speaker_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete speaker: {str(e)}")

@app.post("/api/speakers/{speaker_id}/enroll")
async def add_enrollment_samples(
    speaker_id: int,
    files: list[UploadFile] = File(...),
    db_service=Depends(get_database_service)
):
    """Add additional enrollment samples to an existing speaker."""
    try:
        # Get existing speaker
        speaker = await db_service.speakers.get_speaker(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        if len(files) < 1 or len(files) > 3:
            raise HTTPException(
                status_code=400,
                detail="Please provide 1-3 additional audio files"
            )
        
        # Save uploaded files temporarily
        temp_files = []
        try:
            for i, file in enumerate(files):
                if not file.content_type.startswith('audio/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} is not an audio file"
                    )
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_additional_{i}.wav"
                )
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # Extract embeddings from new samples
            identifier = get_speaker_identifier()
            new_embeddings = []
            quality_scores = []
            
            for temp_file in temp_files:
                embedding, quality = identifier.extract_embedding(temp_file)
                new_embeddings.append(embedding)
                quality_scores.append(quality['quality_score'])
            
            # TODO: Update speaker embeddings in database
            # This would require extending the database service
            
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            logger.info(f"✅ Added {len(new_embeddings)} samples to speaker '{speaker.name}'")
            
            return {
                "speaker_id": speaker_id,
                "samples_added": len(new_embeddings),
                "avg_quality": avg_quality,
                "status": "updated"
            }
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add enrollment samples: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add samples: {str(e)}")

# WebSocket endpoint for audio streaming
@app.websocket("/ws/vad-stream")
async def websocket_vad_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming audio to VAD system"""
    await websocket.accept()
    logger.info("VAD WebSocket connection established")
    
    if not auto_recorder:
        await websocket.send_text(json.dumps({"error": "Auto recorder not initialized"}))
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio_chunk":
                try:
                    # Decode base64 audio data
                    audio_b64 = data.get("audio_data", "")
                    if not audio_b64:
                        logger.warning("Received empty audio data")
                        continue
                        
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert to numpy array (assuming Float32 format)
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    timestamp = data.get("timestamp", time.time())
                    
                    logger.debug(f"📡 Received audio chunk: {len(audio_chunk)} samples")
                    
                    # Convert JavaScript timestamp (milliseconds) to Python timestamp (seconds)
                    original_timestamp = timestamp
                    if timestamp > 10000000000:  # If timestamp is in milliseconds
                        timestamp = timestamp / 1000.0
                    
                    logger.debug(f"Timestamp conversion: {original_timestamp} -> {timestamp}")
                    
                    # Process through AutoRecorder VAD
                    result = auto_recorder.process_audio_chunk(audio_chunk, timestamp)
                    
                    # Send status back to client
                    response = {
                        "type": "vad_status",
                        "result": result,
                        "timestamp": timestamp
                    }
                    await websocket.send_text(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    error_response = {
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}"
                    }
                    await websocket.send_text(json.dumps(error_response))
            
            elif data.get("type") == "ping":
                # Respond to ping with pong
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        logger.info("VAD WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

# ====== PERSISTENT SPEAKER MANAGEMENT APIs ======

@app.get("/api/persistent-speakers")
async def get_persistent_speakers(
    active_only: bool = True,
    limit: Optional[int] = None,
    offset: int = 0,
    db_service=Depends(get_database_service)
):
    """Get all persistent speakers with usage statistics."""
    try:
        speakers = await db_service.persistent_speakers.get_all_persistent_speakers(
            active_only=active_only,
            limit=limit,
            offset=offset
        )
        
        speaker_list = []
        for speaker in speakers:
            speaker_dict = {
                "id": speaker.id,
                "name": speaker.name,
                "display_name": speaker.name or speaker.id,
                "first_seen_recording_id": speaker.first_seen_recording_id,
                "enrollment_method": speaker.enrollment_method,
                "enrollment_quality": speaker.enrollment_quality,
                "total_speaking_time": speaker.total_speaking_time,
                "recordings_count": speaker.recordings_count,
                "segments_count": speaker.segments_count,
                "confidence_threshold": speaker.confidence_threshold,
                "is_active": speaker.is_active,
                "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
                "last_seen_at": speaker.last_seen_at.isoformat() if speaker.last_seen_at else None,
                "embeddings_count": len(speaker.embeddings)
            }
            speaker_list.append(speaker_dict)
        
        return {
            "speakers": speaker_list,
            "total": len(speaker_list),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get persistent speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speakers: {str(e)}")


@app.post("/api/persistent-speakers")
async def create_persistent_speaker(
    speaker_name: str,
    files: list[UploadFile] = File(...),
    db_service=Depends(get_database_service)
):
    """Enroll a new persistent speaker from audio samples."""
    try:
        # Validate input
        if not speaker_name.strip():
            raise HTTPException(status_code=400, detail="Speaker name is required")
        
        if len(files) < 2 or len(files) > 5:
            raise HTTPException(status_code=400, detail="Please provide 2-5 audio files")
        
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
        
        # Save files temporarily
        temp_files = []
        try:
            for file in files:
                content = await file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                    tmp.write(content)
                    temp_files.append(tmp.name)
            
            # Use the persistent speaker manager to enroll
            processor = get_audio_processor()
            if processor.persistent_speaker_manager is None:
                processor.persistent_speaker_manager = PersistentSpeakerManager(
                    db_service, processor.speaker_identifier
                )
            
            # Enroll speaker from audio files
            result = await processor.persistent_speaker_manager.enroll_speaker_from_files(
                speaker_name=speaker_name.strip(),
                audio_files=temp_files
            )
            
            return {
                "status": "success",
                "speaker_id": result["speaker_id"],
                "speaker_name": result["speaker_name"],
                "enrollment_quality": result.get("avg_quality", 0),
                "consistency_score": result.get("consistency_score", 0),
                "files_processed": len(temp_files)
            }
            
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enroll persistent speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enroll speaker: {str(e)}")


@app.delete("/api/persistent-speakers/{speaker_id}")
async def delete_persistent_speaker(
    speaker_id: str,
    db_service=Depends(get_database_service)
):
    """Delete a persistent speaker and all associated data."""
    try:
        # Check if speaker exists
        speaker = await db_service.persistent_speakers.get_persistent_speaker(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        # Delete speaker (this should cascade to delete embeddings, mappings, etc.)
        await db_service.persistent_speakers.delete_persistent_speaker(speaker_id)
        
        return {
            "status": "success",
            "message": f"Speaker {speaker_id} ({speaker.name}) deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persistent speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete speaker: {str(e)}")


@app.post("/api/recordings/{recording_id}/speakers/{session_speaker}/assign")
async def assign_session_speaker(
    recording_id: int,
    session_speaker: str,
    request: dict,
    db_service=Depends(get_database_service)
):
    """Assign a session speaker to a persistent speaker (manual assignment)."""
    try:
        # Validate recording exists
        recording = await db_service.recordings.get_recording(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        assignment_type = request.get("type", "existing")  # existing, new, dismiss
        
        if assignment_type == "existing":
            persistent_speaker_id = request.get("persistent_speaker_id")
            if not persistent_speaker_id:
                raise HTTPException(status_code=400, detail="persistent_speaker_id required for existing assignment")
            
            # Verify persistent speaker exists
            persistent_speaker = await db_service.persistent_speakers.get_persistent_speaker(persistent_speaker_id)
            if not persistent_speaker:
                raise HTTPException(status_code=404, detail="Persistent speaker not found")
            
            # Create mapping
            await db_service.speaker_mappings.create_mapping(
                recording_id=recording_id,
                session_speaker_label=session_speaker,
                persistent_speaker_id=persistent_speaker_id,
                assignment_confidence=1.0,  # Manual assignment = high confidence
                assignment_method="manual",
                similarity_score=1.0,
                needs_review=False
            )
            
            return {
                "success": True,
                "assignment_type": "existing",
                "session_speaker": session_speaker,
                "persistent_speaker_id": persistent_speaker_id,
                "persistent_speaker_name": persistent_speaker.name or persistent_speaker.id
            }
            
        elif assignment_type == "new":
            speaker_name = request.get("name", "").strip()
            
            # Create new persistent speaker (will auto-generate ID)
            persistent_speaker = await db_service.persistent_speakers.create_persistent_speaker(
                name=speaker_name if speaker_name else None,
                first_seen_recording_id=recording_id,
                enrollment_method="manual"
            )
            
            # Create mapping
            await db_service.speaker_mappings.create_mapping(
                recording_id=recording_id,
                session_speaker_label=session_speaker,
                persistent_speaker_id=persistent_speaker.id,
                assignment_confidence=1.0,
                assignment_method="manual",
                similarity_score=1.0,
                needs_review=False
            )
            
            return {
                "success": True,
                "assignment_type": "new",
                "session_speaker": session_speaker,
                "persistent_speaker_id": persistent_speaker.id,
                "persistent_speaker_name": persistent_speaker.name or persistent_speaker.id,
                "created_new": True
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid assignment type")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign session speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Assignment failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)