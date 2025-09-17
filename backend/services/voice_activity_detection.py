"""
Voice Activity Detection Service
Uses Silero VAD for real-time speech detection
"""
import logging
import torch
import numpy as np
from typing import Tuple, List, Optional
import librosa

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Silero VAD
    """
    
    def __init__(self, model_name: str = "silero_vad", sample_rate: int = 16000):
        """
        Initialize Silero VAD model
        
        Args:
            model_name: Silero VAD model name
            sample_rate: Audio sample rate (16000 Hz recommended)
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.model = None
        self.utils = None
        
        # VAD parameters
        self.chunk_size = 512  # Required chunk size for 16kHz Silero VAD
        self.speech_threshold = 0.4  # Lower threshold for better sensitivity
        self.reset_counter = 0  # Track chunks for periodic state reset
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        try:
            logger.info("Loading Silero VAD model...")
            
            # Load Silero VAD
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
    
    def detect_speech_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech in a single audio chunk
        
        Args:
            audio_chunk: Audio data as numpy array (exactly 512 samples for 16kHz)
            
        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        try:
            # Ensure audio is float32 and mono
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.mean(axis=1)
            
            audio_chunk = audio_chunk.astype(np.float32)
            
            # Ensure exact chunk size for Silero VAD
            if len(audio_chunk) != self.chunk_size:
                if len(audio_chunk) < self.chunk_size:
                    # Pad with zeros if too short
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)), 'constant')
                else:
                    # Truncate if too long
                    audio_chunk = audio_chunk[:self.chunk_size]
            
            # Normalize audio if needed (but preserve relative levels)
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                # Gentle normalization to preserve speech characteristics
                audio_chunk = audio_chunk / max(max_val, 0.1)  # Avoid over-normalization
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_chunk)
            
            # Periodic state reset for long streaming sessions
            self.reset_counter += 1
            if self.reset_counter >= 100:  # Reset every 100 chunks (~3 seconds)
                if hasattr(self.model, 'reset_states'):
                    self.model.reset_states()
                self.reset_counter = 0
                logger.debug("VAD model state reset")
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            is_speech = speech_prob > self.speech_threshold
            
            # Only log speech detection changes, not every chunk
            if hasattr(self, '_last_speech_state') and self._last_speech_state != is_speech:
                logger.info(f"VAD: Speech state changed to {is_speech} (conf={speech_prob:.3f})")
            self._last_speech_state = is_speech
            
            return is_speech, speech_prob
            
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            return False, 0.0
    
    def process_audio_stream(self, audio_data: np.ndarray, chunk_size: Optional[int] = None) -> List[Tuple[bool, float]]:
        """
        Process continuous audio stream in chunks
        
        Args:
            audio_data: Full audio data
            chunk_size: Size of each chunk (default: 1 second)
            
        Returns:
            List of (is_speech, confidence) for each chunk
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        results = []
        
        # Process audio in chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            is_speech, confidence = self.detect_speech_chunk(chunk)
            results.append((is_speech, confidence))
        
        return results
    
    def detect_speech_from_file(self, audio_path: str) -> List[Tuple[float, bool, float]]:
        """
        Detect speech activity from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (timestamp, is_speech, confidence) tuples
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Process in chunks
            results = self.process_audio_stream(audio_data)
            
            # Add timestamps
            timestamped_results = []
            chunk_duration = self.chunk_size / self.sample_rate
            
            for i, (is_speech, confidence) in enumerate(results):
                timestamp = i * chunk_duration
                timestamped_results.append((timestamp, is_speech, confidence))
            
            return timestamped_results
            
        except Exception as e:
            logger.error(f"File VAD processing failed: {e}")
            return []
    
    def get_speech_segments(self, audio_path: str, min_duration: float = 1.0) -> List[Tuple[float, float]]:
        """
        Get continuous speech segments from audio file
        
        Args:
            audio_path: Path to audio file
            min_duration: Minimum segment duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        vad_results = self.detect_speech_from_file(audio_path)
        
        if not vad_results:
            return []
        
        segments = []
        chunk_duration = self.chunk_size / self.sample_rate
        current_start = None
        
        for timestamp, is_speech, confidence in vad_results:
            if is_speech and current_start is None:
                # Start of speech segment
                current_start = timestamp
            elif not is_speech and current_start is not None:
                # End of speech segment
                duration = timestamp - current_start
                if duration >= min_duration:
                    segments.append((current_start, timestamp))
                current_start = None
        
        # Handle case where speech continues to end of audio
        if current_start is not None:
            duration = vad_results[-1][0] + chunk_duration - current_start
            if duration >= min_duration:
                segments.append((current_start, vad_results[-1][0] + chunk_duration))
        
        return segments
    
    def set_threshold(self, threshold: float):
        """Update speech detection threshold"""
        if 0.0 <= threshold <= 1.0:
            self.speech_threshold = threshold
            logger.info(f"VAD threshold updated to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, must be between 0.0 and 1.0")
    
    def get_info(self) -> dict:
        """Get VAD model information"""
        return {
            "model": self.model_name,
            "provider": "Silero VAD",
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "speech_threshold": self.speech_threshold,
            "loaded": self.model is not None
        }