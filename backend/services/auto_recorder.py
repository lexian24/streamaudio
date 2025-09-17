"""
Automatic Meeting Recorder
Uses VAD to automatically detect and record meetings with smart start/stop logic
"""
import logging
import numpy as np
import threading
import time
import tempfile
import soundfile as sf
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Callable, Dict, Any
from collections import deque

from .voice_activity_detection import VoiceActivityDetector

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """Recording states"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class AutoRecorder:
    """
    VAD-based speech detection service for triggering frontend recording
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: float = 1.0,
                 silence_threshold: float = 60.0,
                 buffer_duration: float = 65.0,
                 recordings_dir: str = "recordings"):
        """
        Initialize automatic recorder
        
        Args:
            sample_rate: Audio sample rate
            chunk_duration: Duration of each audio chunk in seconds
            silence_threshold: Silence duration to end recording (seconds)
            buffer_duration: Total buffer duration (silence_threshold + 5s)
            recordings_dir: Directory to save recordings
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.silence_threshold = 5.0  # Changed to 5 seconds for development
        self.buffer_duration = buffer_duration
        self.buffer_size = int(buffer_duration / chunk_duration)
        
        # Create recordings directory
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True)
        
        # State management
        self.state = RecordingState.IDLE
        self.silence_start_time = None
        
        # Audio buffer (circular buffer for VAD context only)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.buffer_timestamps = deque(maxlen=self.buffer_size)
        
        # Audio chunk buffer for VAD processing only
        self.vad_buffer = np.array([], dtype=np.float32)  # Accumulate audio for 512-sample chunks
        
        # VAD detector
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        # Threading
        self.recording_thread = None
        self.stop_flag = threading.Event()
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            "total_recordings": 0,
            "total_recording_time": 0.0,
            "current_session_start": None,
            "last_recording_path": None
        }
        
        logger.info("AutoRecorder initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring for voice activity"""
        if self.state != RecordingState.IDLE:
            logger.warning("Already monitoring, stopping first")
            self.stop_monitoring()
        
        logger.info("Starting automatic voice monitoring...")
        self.stop_flag.clear()
        self.stats["current_session_start"] = datetime.now()
        
        # Note: This is set up for integration with audio stream
        # In practice, you'll feed audio chunks via process_audio_chunk()
        self._change_state(RecordingState.IDLE)
        
        logger.info("✅ Automatic monitoring started (waiting for audio stream)")
    
    def stop_monitoring(self):
        """Stop VAD monitoring"""
        logger.info("Stopping VAD monitoring...")
        
        self.stop_flag.set()
        
        # Clear audio buffers
        self.audio_buffer.clear()
        self.buffer_timestamps.clear()
        self.vad_buffer = np.array([], dtype=np.float32)
        self.silence_start_time = None
        
        self._change_state(RecordingState.IDLE)
        
        logger.info("✅ VAD monitoring stopped")
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Process incoming audio chunk and manage recording state
        
        Args:
            audio_chunk: Audio data chunk
            timestamp: Chunk timestamp (optional)
            
        Returns:
            Status information
        """
        if self.stop_flag.is_set():
            logger.info("🛑 Processing stopped (stop flag set)")
            return {"status": "stopped"}
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Add incoming audio to buffer for proper VAD processing
            self.vad_buffer = np.concatenate([self.vad_buffer, audio_chunk])
            
            vad_chunk_size = 512  # Required by Silero VAD
            speech_detections = []
            confidences = []
            
            # Process complete 512-sample chunks from buffer
            while len(self.vad_buffer) >= vad_chunk_size:
                # Extract exactly 512 samples
                vad_chunk = self.vad_buffer[:vad_chunk_size]
                self.vad_buffer = self.vad_buffer[vad_chunk_size:]  # Remove processed samples
                
                # Process through VAD
                is_speech, confidence = self.vad.detect_speech_chunk(vad_chunk)
                speech_detections.append(is_speech)
                confidences.append(confidence)
            
            # If no complete chunks were processed, return current state
            if not speech_detections:
                logger.debug(f"VAD: Buffering audio ({len(self.vad_buffer)} samples waiting)")
                return {
                    "status": "buffering",
                    "state": self.state.value,
                    "buffer_samples": len(self.vad_buffer)
                }
            
            # Aggregate results: consider it speech if any chunk detected speech
            is_speech = any(speech_detections)
            confidence = max(confidences) if confidences else 0.0
            
            logger.debug(f"VAD: {len(speech_detections)} chunks processed, speech={is_speech}, conf={confidence:.3f}, state={self.state.value}, buffer={len(self.vad_buffer)}")
            
            # Add chunk to rolling buffer for VAD context only
            self.audio_buffer.append(audio_chunk.copy())
            self.buffer_timestamps.append(timestamp)
            
            # Simple VAD state tracking (recording managed by frontend)
            if is_speech:
                if self.state == RecordingState.IDLE:
                    self._change_state(RecordingState.RECORDING)
                    logger.info("🟢 Speech detected - VAD active")
                # Reset silence timer when speech is detected
                self.silence_start_time = None
            else:
                if self.state == RecordingState.RECORDING:
                    # Track silence duration for frontend
                    if self.silence_start_time is None:
                        self.silence_start_time = timestamp
                        logger.info(f"🔇 Silence started at {timestamp}")
                    else:
                        silence_duration = timestamp - self.silence_start_time
                        # Only log every 1 second to reduce spam
                        if int(silence_duration) != int(silence_duration - 0.1):
                            logger.info(f"🔇 Silence duration: {silence_duration:.1f}s (threshold: {self.silence_threshold}s)")
                        if silence_duration >= self.silence_threshold:
                            self._change_state(RecordingState.IDLE)
                            self.silence_start_time = None
                            logger.info("🔴 VAD inactive after silence threshold")
            
            return {
                "status": "active",
                "state": self.state.value,
                "is_speech": is_speech,
                "confidence": confidence,
                "silence_duration": self._get_silence_duration(timestamp),
                "buffer_size": len(self.audio_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Reset to idle state instead of staying in error
            self._change_state(RecordingState.IDLE)
            self.silence_start_time = None
            return {"status": "error", "error": str(e)}
    
    
    def _get_silence_duration(self, current_timestamp: float) -> Optional[float]:
        """Get current silence duration"""
        if self.silence_start_time is not None:
            return current_timestamp - self.silence_start_time
        return None
    
    
    def _change_state(self, new_state: RecordingState):
        """Change recording state and notify callbacks"""
        old_state = self.state
        self.state = new_state
        
        logger.debug(f"State change: {old_state.value} → {new_state.value}")
        
        if self.on_state_change:
            self.on_state_change(old_state, new_state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current VAD status"""
        current_time = time.time()
        
        return {
            "state": self.state.value,
            "monitoring": not self.stop_flag.is_set(),
            "silence_duration": self._get_silence_duration(current_time),
            "buffer_size": len(self.audio_buffer),
            "buffer_capacity": self.buffer_size,
            "stats": self.stats.copy(),
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_duration": self.chunk_duration,
                "silence_threshold": self.silence_threshold,
                "buffer_duration": self.buffer_duration
            }
        }
    
    def get_recordings(self) -> List[Dict[str, Any]]:
        """Get list of recorded meetings"""
        recordings = []
        
        if not self.recordings_dir.exists():
            return recordings
        
        for wav_file in self.recordings_dir.glob("recording_*.wav"):
            try:
                # Get file info
                stat = wav_file.stat()
                
                # Parse timestamp from filename
                timestamp_str = wav_file.stem.replace("recording_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                
                # Get audio duration
                with sf.SoundFile(wav_file) as audio_info:
                    duration = len(audio_info) / audio_info.samplerate
                
                recordings.append({
                    "filename": wav_file.name,
                    "path": str(wav_file),
                    "timestamp": timestamp.isoformat(),
                    "duration": duration,
                    "size": stat.st_size
                })
                
            except Exception as e:
                logger.warning(f"Error reading recording info for {wav_file}: {e}")
        
        # Sort by timestamp (newest first)
        recordings.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return recordings
    
    def delete_recording(self, filename: str) -> bool:
        """Delete a specific recording"""
        try:
            filepath = self.recordings_dir / filename
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted recording: {filename}")
                return True
            else:
                logger.warning(f"Recording not found: {filename}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete recording {filename}: {e}")
            return False