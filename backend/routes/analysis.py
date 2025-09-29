"""
Audio analysis endpoints.
"""

import tempfile
import os
import time
import logging
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

@router.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    result = {"status": "lazy loading"}
    if audio_processor:
        result["audio_processing"] = audio_processor.get_model_info()
    return result