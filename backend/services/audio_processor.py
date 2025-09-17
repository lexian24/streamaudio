"""
FastAudio Audio Processor
Simple pipeline: Whisper transcription + pyannote diarization + emotion recognition
"""
import logging
from typing import Dict, List, Any
import librosa
import numpy as np

from .whisper_transcription import WhisperTranscriber
from .diarization import SpeakerDiarizer
from .emotion_recognition import EmotionRecognizer

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Complete post-processing audio pipeline:
    1. Speaker diarization (pyannote identifies speakers)
    2. Whisper transcription (per speaker segment) 
    3. Emotion recognition (per speaker segment)
    4. Segment merging (consecutive same-speaker segments)
    """
    
    def __init__(self):
        logger.info("Initializing audio processing components...")
        
        # Initialize components
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.emotion_recognizer = EmotionRecognizer()
        
        logger.info("All audio processing components loaded")
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Process audio file through the complete pipeline
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with speakers, segments, and analysis results
        """
        try:
            logger.info(f"Starting audio processing: {audio_path}")
            
            # Step 1: Speaker diarization
            logger.info("Running speaker diarization...")
            diarization = self.diarizer.diarize(audio_path)
            
            # Step 2: Extract segments for transcription
            segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start_time': segment.start,
                    'end_time': segment.end,
                    'speaker_id': speaker
                })
            
            logger.info(f"Found {len(segments)} speech segments from {len(set(s['speaker_id'] for s in segments))} speakers")
            
            # Step 3: Merge consecutive segments from the same speaker
            logger.info("Merging consecutive segments from same speakers...")
            merged_segments = self._merge_consecutive_speaker_segments(segments)
            logger.info(f"Merged {len(segments)} segments into {len(merged_segments)} continuous speaker blocks")
            
            # Step 4: Transcribe and analyze merged segments
            logger.info("Transcribing merged segments...")
            final_segments = []
            
            for i, segment in enumerate(merged_segments):
                # Transcribe the entire merged segment
                text = self.transcriber.transcribe_segment(
                    audio_path,
                    segment['start_time'],
                    segment['end_time']
                )
                
                # Analyze emotion for the entire merged segment
                emotion, confidence = self.emotion_recognizer.predict_emotion(
                    audio_path,
                    segment['start_time'],
                    segment['end_time'],
                    text
                )
                
                final_segments.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'speaker_id': segment['speaker_id'],
                    'text': text,
                    'emotion': emotion,
                    'emotion_confidence': confidence,
                    'duration': segment['end_time'] - segment['start_time'],
                    'original_segment_count': segment.get('segment_count', 1)
                })
                
                logger.info(f"Processed speaker {segment['speaker_id']}: {len(text)} chars, emotion: {emotion}")
            
            # Step 5: Generate speaker summary
            speakers = self._generate_speaker_summary(final_segments)
            
            result = {
                'speakers': speakers,
                'segments': final_segments,
                'total_speakers': len(speakers),
                'total_segments': len(final_segments)
            }
            
            logger.info("Audio processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    def _generate_speaker_summary(self, segments: List[Dict]) -> List[Dict]:
        """Generate speaker summary with timing information"""
        speakers = {}
        
        for segment in segments:
            speaker_id = segment['speaker_id']
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    'speaker_id': speaker_id,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time']
                }
            else:
                # Update timing
                speakers[speaker_id]['start_time'] = min(
                    speakers[speaker_id]['start_time'], 
                    segment['start_time']
                )
                speakers[speaker_id]['end_time'] = max(
                    speakers[speaker_id]['end_time'],
                    segment['end_time']
                )
        
        return list(speakers.values())
    
    def _merge_consecutive_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge consecutive segments from the same speaker into continuous blocks
        
        Args:
            segments: List of segment dictionaries with speaker info
            
        Returns:
            List of merged segments where consecutive same-speaker segments are combined
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        merged_segments = []
        current_segment = sorted_segments[0].copy()
        current_segment['segment_count'] = 1
        
        for next_segment in sorted_segments[1:]:
            # Check if this segment is from the same speaker and consecutive
            if (next_segment['speaker_id'] == current_segment['speaker_id'] and
                next_segment['start_time'] <= current_segment['end_time'] + 5.0):  # Allow up to 5 second gaps
                
                # Merge: extend the end time and increment count
                current_segment['end_time'] = max(current_segment['end_time'], next_segment['end_time'])
                current_segment['segment_count'] += 1
                
            else:
                # Different speaker or large gap: finalize current segment and start new one
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
                current_segment['segment_count'] = 1
        
        # Don't forget the last segment
        merged_segments.append(current_segment)
        
        logger.info(f"Segment merging stats:")
        for segment in merged_segments:
            duration = segment['end_time'] - segment['start_time']
            logger.info(f"  {segment['speaker_id']}: {duration:.1f}s (merged {segment['segment_count']} segments)")
        
        return merged_segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'transcription': self.transcriber.get_info(),
            'diarization': self.diarizer.get_info(),
            'emotion': self.emotion_recognizer.get_info(),
            'pipeline': 'post_processing: diarization → transcription → emotion'
        }