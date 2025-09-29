"""
Audio analysis endpoints.
"""

import tempfile
import os
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from services.enhanced_audio_processor import EnhancedAudioProcessor
from database.services import DatabaseService
from database import get_database_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Global processor for lazy loading
audio_processor = None

def get_audio_processor():
    """Lazy load the enhanced audio processor with persistent speaker identification"""
    global audio_processor
    if audio_processor is None:
        logger.info("Loading AI models for file processing with speaker identification...")
        # The enhanced processor will get database services as needed in async context
        audio_processor = EnhancedAudioProcessor(db_service=None)
        logger.info("Enhanced processor ready!")
    return audio_processor

@router.post("/analyze")
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
                import subprocess
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
                import librosa
                duration = librosa.get_duration(path=processing_path)
                sample_rate = librosa.get_samplerate(processing_path)
            except Exception:
                duration = None
                sample_rate = 16000
            
            # Process audio with persistent speaker identification
            logger.info(f"Processing audio with speaker identification: {file.filename}")
            
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
            
            # Use enhanced processor with persistent speaker identification
            result = await processor.process_audio_with_persistent_speakers(
                processing_path, 
                recording.id,
                db_service,
                auto_enroll_new_speakers=True
            )
            
            # Save processing results to database
            logger.info(f"Saving processing results to database for recording {recording.id}")
            try:
                # Prepare transcription (combine all segment texts, filter empty segments)
                transcription = ""
                if result.get('segments'):
                    # Only include segments with actual text content
                    segment_texts = [seg.get('text', '').strip() for seg in result['segments'] if seg.get('text', '').strip()]
                    transcription = " ".join(segment_texts).strip()
                
                # Calculate overall confidence from segments
                confidence_score = None
                if result.get('segments'):
                    confidences = [seg.get('confidence', 0) for seg in result['segments'] if seg.get('confidence')]
                    if confidences:
                        confidence_score = sum(confidences) / len(confidences)
                
                # Extract emotion data
                emotions_data = None
                dominant_emotion = None
                emotion_confidence = None
                if result.get('segments'):
                    emotions = [seg.get('emotion', '') for seg in result['segments'] if seg.get('emotion')]
                    if emotions:
                        # Find most common emotion
                        from collections import Counter
                        emotion_counts = Counter(emotions)
                        dominant_emotion = emotion_counts.most_common(1)[0][0]
                        
                        # Get confidence for that emotion
                        emotion_confidences = [
                            seg.get('emotion_confidence', 0) 
                            for seg in result['segments'] 
                            if seg.get('emotion') == dominant_emotion
                        ]
                        if emotion_confidences:
                            emotion_confidence = sum(emotion_confidences) / len(emotion_confidences)
                        
                        emotions_data = {
                            'dominant_emotion': dominant_emotion,
                            'emotion_confidence': emotion_confidence,
                            'all_emotions': dict(emotion_counts)
                        }
                
                # Prepare diarization data (filter out empty segments)
                filtered_segments = [seg for seg in result.get('segments', []) if seg.get('text', '').strip()]
                diarization_data = {
                    'num_speakers': len(result.get('speakers', [])),
                    'speaker_segments': filtered_segments,
                    'speaker_summary': result.get('speakers', [])
                }
                
                # Create processing result in database
                processing_result = await db_service.processing_results.create_processing_result(
                    recording_id=recording.id,
                    transcription=transcription or None,
                    confidence_score=confidence_score,
                    diarization_data=diarization_data,
                    emotions_data=emotions_data,
                    model_versions={
                        "whisper": "openai/whisper-small",
                        "pyannote": "pyannote/speaker-diarization-3.1",
                        "emotion": "j-hartmann/emotion-english-distilroberta-base"
                    }
                )
                
                # Save speaker segments to database if they exist (filter out empty segments)
                if result.get('segments'):
                    logger.info(f"Found {len(result['segments'])} segments to process")
                    
                    # Prepare segments data for bulk insert, filtering out empty segments
                    segments_to_save = []
                    for segment in result['segments']:
                        # Only save segments with actual text content
                        segment_text = segment.get('text', '').strip()
                        logger.info(f"Processing segment: speaker={segment.get('speaker_id')}, text_length={len(segment_text)}, text_preview='{segment_text[:50]}...'")
                        if not segment_text:
                            logger.info(f"Skipping empty segment for speaker {segment.get('speaker_id')}")
                            continue
                            
                        segments_to_save.append({
                            'speaker_id': None,  # Foreign key to speakers table - not used for now
                            'speaker_label': segment.get('persistent_speaker_name') or segment.get('speaker_id', 'Unknown'),
                            'start_time': segment.get('start_time', 0),
                            'end_time': segment.get('end_time', 0),
                            'confidence': segment.get('confidence', 0),
                            'text': segment_text
                        })
                    
                    # Save all segments in one call
                    if segments_to_save:
                        try:
                            saved_segments = await db_service.speaker_segments.create_speaker_segments(
                                result_id=processing_result.id,
                                segments_data=segments_to_save
                            )
                            logger.info(f"✅ Total speaker segments saved: {len(saved_segments)} out of {len(result['segments'])}")
                        except Exception as e:
                            logger.error(f"❌ Failed to save speaker segments: {e}")
                    else:
                        logger.info("No segments with text content to save")
                
                logger.info(f"✅ Processing results saved to database with ID {processing_result.id}")
                
            except Exception as e:
                logger.error(f"Failed to save processing results to database: {e}")
                # Continue anyway - we still have the results to return
            
            # Add metadata
            result.update({
                "filename": file.filename,
                "processing_time": time.time() - start_time,
                "status": "completed"
            })
            
            logger.info(f"Audio processing completed in {result['processing_time']:.2f}s")
            logger.info(f"✅ Audio file saved permanently: {stored_filename}")
            return result
            
        finally:
            # Clean up temporary file only if conversion happened
            if needs_conversion and temp_path != final_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    result = {"status": "lazy loading"}
    if audio_processor:
        result["audio_processing"] = audio_processor.get_model_info()
    return result