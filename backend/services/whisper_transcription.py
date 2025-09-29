"""
Whisper Transcription Service
Uses Whisper via Transformers pipeline for speech-to-text
"""
import logging
from transformers import pipeline
import numpy as np
import librosa
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Whisper-based transcriber using Transformers pipeline
    Efficient audio-to-text transcription for post-processing
    """
    
    def __init__(self, model_name: str = "openai/whisper-base.en"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Hugging Face Whisper model name
        """
        self.model_name = model_name
        self.transcriber = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper transcription pipeline"""
        try:
            logger.info(f"Loading {self.model_name} Whisper pipeline...")
            self.transcriber = pipeline(
                "automatic-speech-recognition", 
                model=self.model_name
            )
            logger.info("✅ Whisper ASR pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper pipeline: {e}")
            raise
    
    def transcribe_audio_array(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio array directly
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ensure float32 and normalize
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data /= np.max(np.abs(audio_data))
            
            # Skip very short audio
            if len(audio_data) < 0.1 * sample_rate:  # Less than 0.1 seconds
                return ""
            
            # Transcribe using Whisper pipeline with timestamps for long audio
            audio_duration = len(audio_data) / sample_rate
            
            if audio_duration > 30:  # Use timestamps for long audio (>30 seconds)
                result = self.transcriber({
                    "sampling_rate": sample_rate, 
                    "raw": audio_data
                }, return_timestamps=True)
            else:
                result = self.transcriber({
                    "sampling_rate": sample_rate, 
                    "raw": audio_data
                })
            
            text = result["text"].strip() if result and "text" in result else ""
            logger.debug(f"Transcribed {len(audio_data)/sample_rate:.1f}s audio: {text[:50]}...")
            
            return text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def transcribe_full(self, audio_path: str) -> str:
        """
        Transcribe entire audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Use array-based transcription
            return self.transcribe_audio_array(audio_data, sr)
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return ""
    
    def transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> str:
        """
        Transcribe a specific segment of audio file (for diarization pipeline)
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Transcribed text for the segment
        """
        try:
            # Load audio segment - process the FULL segment (no chunking)
            duration = end_time - start_time
            audio_data, sr = librosa.load(
                audio_path, 
                sr=16000, 
                offset=start_time, 
                duration=duration
            )
            
            logger.debug(f"Transcribing segment [{start_time:.1f}s-{end_time:.1f}s] ({duration:.1f}s duration)")
            
            # Use array-based transcription for the entire segment
            return self.transcribe_audio_array(audio_data, sr)
            
        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            return ""
    
    def get_info(self) -> dict:
        """Get transcriber information"""
        return {
            "model": self.model_name,
            "provider": "Whisper via Transformers",
            "language": "en",
            "loaded": self.transcriber is not None,
            "type": "post-processing"
        }