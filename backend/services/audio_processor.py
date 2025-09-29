"""
FastAudio Audio Processor
Simple pipeline: Whisper transcription + pyannote diarization + emotion recognition
"""
import logging
import warnings
from typing import Dict, List, Any
import librosa
import numpy as np
import tempfile
import os

from .whisper_transcription import WhisperTranscriber
from .diarization import SpeakerDiarizer
from .emotion_recognition import EmotionRecognizer
from .speaker_identification import SpeakerIdentifier

# Suppress common audio processing warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

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
        
        # Initialize speaker identifier (lazy loading)
        self.speaker_identifier = None
        
        logger.info("All audio processing components loaded")
    
    def _get_speaker_identifier(self):
        """Get or initialize speaker identifier (lazy loading)."""
        if self.speaker_identifier is None:
            try:
                self.speaker_identifier = SpeakerIdentifier()
                logger.info("Speaker identifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load speaker identifier: {e}")
                self.speaker_identifier = None
        return self.speaker_identifier
    
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
    
    def process_audio_with_identification(
        self, 
        audio_path: str, 
        enrolled_speakers: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process audio file with speaker identification.
        
        Args:
            audio_path: Path to the audio file
            enrolled_speakers: List of enrolled speaker data from database
            
        Returns:
            Dictionary with speakers, segments, and analysis results including speaker names
        """
        try:
            logger.info(f"Starting audio processing with speaker identification: {audio_path}")
            
            # First run the standard audio processing
            result = self.process_audio(audio_path)
            
            # If no enrolled speakers or no speaker identifier, return standard result
            if not enrolled_speakers or not self._get_speaker_identifier():
                logger.info("No speaker identification performed (no enrolled speakers or identifier)")
                return result
            
            # Prepare enrolled speakers data for identification
            identifier = self._get_speaker_identifier()
            identification_speakers = []
            
            for speaker_data in enrolled_speakers:
                if speaker_data.embeddings_data and 'embeddings' in speaker_data.embeddings_data:
                    # Convert embeddings back to numpy arrays
                    embeddings = [
                        np.array(emb) for emb in speaker_data.embeddings_data['embeddings']
                    ]
                    
                    identification_speakers.append({
                        'speaker_name': speaker_data.name,
                        'speaker_id': speaker_data.id,
                        'embeddings': embeddings,
                        'enrollment_threshold': speaker_data.embeddings_data.get(
                            'enrollment_threshold', speaker_data.confidence_threshold
                        )
                    })
            
            if not identification_speakers:
                logger.info("No valid enrolled speakers found for identification")
                return result
            
            logger.info(f"Running speaker identification against {len(identification_speakers)} enrolled speakers")
            
            # OPTIMIZED: Identify each unique speaker only once instead of every segment
            import tempfile
            import os
            from pydub import AudioSegment
            
            speaker_identification_cache = {}  # Cache: speaker_id → identification_result
            audio = AudioSegment.from_file(audio_path)
            
            # Phase 1: Identify each unique speaker only once (HUGE efficiency gain!)
            unique_speakers = set(segment['speaker_id'] for segment in result['segments'])
            logger.info(f"🚀 Optimized identification: {len(unique_speakers)} unique speakers vs {len(result['segments'])} segments")
            
            for speaker_id in unique_speakers:
                # Find the longest segment for this speaker (best quality for identification)
                speaker_segments = [s for s in result['segments'] if s['speaker_id'] == speaker_id]
                representative_segment = max(speaker_segments, key=lambda s: s['end_time'] - s['start_time'])
                
                try:
                    # Extract representative audio segment
                    start_ms = int(representative_segment['start_time'] * 1000)
                    end_ms = int(representative_segment['end_time'] * 1000)
                    segment_audio = audio[start_ms:end_ms]
                    
                    # Save segment to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    segment_audio.export(temp_file.name, format="wav")
                    temp_file.close()
                    
                    try:
                        # Identify speaker ONCE per unique speaker (not per segment!)
                        identification_result = identifier.identify_speaker(
                            temp_file.name, identification_speakers
                        )
                        speaker_identification_cache[speaker_id] = identification_result
                        
                        if identification_result['identified_speaker']:
                            logger.info(f"✅ {speaker_id} → {identification_result['identified_speaker']} "
                                      f"(confidence: {identification_result['confidence']:.3f})")
                        else:
                            logger.info(f"❓ {speaker_id} → Unidentified (best: {identification_result.get('best_score', 0):.3f})")
                        
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                            
                except Exception as e:
                    logger.warning(f"Failed to identify speaker {speaker_id}: {e}")
                    speaker_identification_cache[speaker_id] = {
                        'identified_speaker': None,
                        'confidence': 0.0,
                        'similarity_score': 0.0
                    }
            
            # Phase 2: Apply cached results to all segments (lightning fast!)
            identified_segments = []
            enrolled_speaker_assignments = {}  # Track which enrolled speakers are assigned
            
            for segment in result['segments']:
                segment_copy = segment.copy()
                speaker_id = segment['speaker_id']
                identification_result = speaker_identification_cache.get(speaker_id, {})
                
                if identification_result.get('identified_speaker'):
                    enrolled_name = identification_result['identified_speaker']
                    segment_copy['identified_speaker_name'] = enrolled_name
                    segment_copy['identification_confidence'] = identification_result['confidence']
                    segment_copy['similarity_score'] = identification_result.get('similarity_score', 0.0)
                    
                    # Track assignments for multiple speaker detection
                    if enrolled_name not in enrolled_speaker_assignments:
                        enrolled_speaker_assignments[enrolled_name] = []
                    enrolled_speaker_assignments[enrolled_name].append(speaker_id)
                    
                    # Find speaker ID in enrolled speakers
                    for enrolled_speaker in identification_speakers:
                        if enrolled_speaker['speaker_name'] == enrolled_name:
                            segment_copy['identified_speaker_id'] = enrolled_speaker['speaker_id']
                            break
                else:
                    segment_copy['identified_speaker_name'] = None
                    segment_copy['identified_speaker_id'] = None
                    segment_copy['identification_confidence'] = 0.0
                
                identified_segments.append(segment_copy)
            
            # Log multiple speaker assignments (helpful for similar voices)
            for enrolled_name, assigned_speakers in enrolled_speaker_assignments.items():
                if len(assigned_speakers) > 1:
                    logger.warning(f"🔄 Multiple speakers identified as '{enrolled_name}': {assigned_speakers}")
                    logger.warning(f"   This may indicate similar voices or audio quality issues")
            
            # Update result with identified segments
            result['segments'] = identified_segments
            
            # Generate updated speaker summary with identification
            result['speakers'] = self._generate_speaker_summary_with_identification(identified_segments)
            
            logger.info("Audio processing with speaker identification completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Audio processing with identification failed: {e}")
            # Fall back to standard processing
            return self.process_audio(audio_path)
    
    def _generate_speaker_summary_with_identification(self, segments: List[Dict]) -> List[Dict]:
        """Generate speaker summary with identification information."""
        speakers = {}
        
        for segment in segments:
            # Use identified speaker name if available, otherwise use diarization label
            speaker_key = segment.get('identified_speaker_name') or segment['speaker_id']
            
            if speaker_key not in speakers:
                speakers[speaker_key] = {
                    'speaker_id': segment['speaker_id'],  # Original diarization label
                    'speaker_name': segment.get('identified_speaker_name'),
                    'identified_speaker_id': segment.get('identified_speaker_id'),
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'total_speaking_time': segment['end_time'] - segment['start_time'],
                    'avg_confidence': segment.get('identification_confidence', 0.0),
                    'segment_count': 1
                }
            else:
                # Update timing and confidence
                speakers[speaker_key]['start_time'] = min(
                    speakers[speaker_key]['start_time'], 
                    segment['start_time']
                )
                speakers[speaker_key]['end_time'] = max(
                    speakers[speaker_key]['end_time'],
                    segment['end_time']
                )
                speakers[speaker_key]['total_speaking_time'] += (
                    segment['end_time'] - segment['start_time']
                )
                
                # Average confidence across segments
                current_avg = speakers[speaker_key]['avg_confidence']
                current_count = speakers[speaker_key]['segment_count']
                new_confidence = segment.get('identification_confidence', 0.0)
                
                speakers[speaker_key]['avg_confidence'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
                speakers[speaker_key]['segment_count'] += 1
        
        return list(speakers.values())
    
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