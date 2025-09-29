"""
Enhanced Audio Processor with Persistent Speaker Management

This module extends the basic audio processor to include intelligent speaker enrollment
and persistent speaker identity management across recordings.
"""

import logging
from typing import Dict, List, Any, Optional
import tempfile
import os
import numpy as np
from pydub import AudioSegment

from .audio_processor import AudioProcessor
from .persistent_speaker_manager import PersistentSpeakerManager
from .speaker_identification import SpeakerIdentifier
from database.services import DatabaseService

logger = logging.getLogger(__name__)


class EnhancedAudioProcessor(AudioProcessor):
    """
    Enhanced audio processor with persistent speaker management.
    
    Extends the basic audio processor to:
    1. Automatically identify speakers against persistent database
    2. Enroll new speakers from high-quality segments
    3. Maintain session-to-persistent speaker mappings
    4. Queue ambiguous cases for manual review
    """
    
    def __init__(self, db_service: DatabaseService = None):
        super().__init__()
        self.db = db_service  # Can be None, will be set later
        self.speaker_identifier = SpeakerIdentifier()
        self.persistent_speaker_manager = None  # Will be initialized when needed
    
    async def process_audio_with_persistent_speakers(
        self,
        audio_path: str,
        recording_id: int,
        db_service: DatabaseService,
        auto_enroll_new_speakers: bool = True,
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Process audio with persistent speaker management.
        
        Args:
            audio_path: Path to audio file
            recording_id: Database ID of the recording
            db_service: Database service instance
            auto_enroll_new_speakers: Whether to auto-enroll unrecognized speakers
            confidence_threshold: Minimum confidence for speaker matching
            
        Returns:
            Enhanced processing results with persistent speaker assignments
        """
        # Initialize database service and manager if needed
        if self.db is None:
            self.db = db_service
        if self.persistent_speaker_manager is None:
            self.persistent_speaker_manager = PersistentSpeakerManager(
                self.db, self.speaker_identifier
            )
        try:
            logger.info(f"Processing audio with persistent speakers: {audio_path}")
            
            # Step 1: Run standard audio processing (diarization, transcription, emotion)
            base_result = self.process_audio(audio_path)
            
            if not base_result['segments']:
                logger.info("No speech segments found, skipping speaker processing")
                return base_result
            
            # Step 2: Group segments by session speaker
            session_speakers = {}
            for segment in base_result['segments']:
                speaker_id = segment['speaker_id']
                if speaker_id not in session_speakers:
                    session_speakers[speaker_id] = []
                session_speakers[speaker_id].append(segment)
            
            logger.info(f"Found {len(session_speakers)} unique session speakers")
            
            # Step 3: Process each session speaker for persistent assignment
            persistent_assignments = {}
            enrollment_results = {}
            review_queue_items = []
            
            for session_speaker, segments in session_speakers.items():
                try:
                    assignment_result = await self._process_session_speaker(
                        session_speaker,
                        segments,
                        recording_id,
                        audio_path,
                        auto_enroll_new_speakers,
                        confidence_threshold
                    )
                    
                    if assignment_result['method'] == 'matched_existing':
                        persistent_assignments[session_speaker] = assignment_result
                    elif assignment_result['method'] == 'enrolled_new':
                        enrollment_results[session_speaker] = assignment_result
                    elif assignment_result['method'] == 'needs_review':
                        review_queue_items.append(assignment_result)
                    
                except Exception as e:
                    logger.error(f"Failed to process session speaker {session_speaker}: {e}")
                    continue
            
            # Step 4: Apply persistent speaker assignments to segments
            enhanced_segments = await self._apply_persistent_assignments(
                base_result['segments'],
                persistent_assignments,
                enrollment_results
            )
            
            # Step 5: Generate enhanced result
            enhanced_result = {
                **base_result,
                'segments': enhanced_segments,
                'persistent_speaker_assignments': persistent_assignments,
                'new_speaker_enrollments': enrollment_results,
                'review_queue_items': review_queue_items,
                'speakers': self._generate_enhanced_speaker_summary(enhanced_segments)
            }
            
            logger.info(f"✅ Enhanced processing completed: "
                       f"{len(persistent_assignments)} matched, "
                       f"{len(enrollment_results)} enrolled, "
                       f"{len(review_queue_items)} for review")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced audio processing failed: {e}")
            # Fallback to basic processing
            return self.process_audio(audio_path)
    
    async def _process_session_speaker(
        self,
        session_speaker: str,
        segments: List[Dict[str, Any]],
        recording_id: int,
        audio_path: str,
        auto_enroll: bool,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """
        Process a single session speaker for persistent assignment.
        
        Returns assignment result with method and details.
        """
        logger.info(f"Processing session speaker {session_speaker} with {len(segments)} segments")
        
        # Step 1: Try to match against existing persistent speakers
        match_result = await self.persistent_speaker_manager.find_matching_persistent_speaker(
            segments, audio_path, confidence_threshold
        )
        
        if match_result:
            # High confidence match found
            logger.info(f"Matched {session_speaker} to {match_result['persistent_speaker_id']} "
                       f"(confidence: {match_result['confidence']:.3f})")
            
            # Create mapping
            await self.persistent_speaker_manager.create_speaker_mapping(
                recording_id,
                session_speaker,
                match_result['persistent_speaker_id'],
                match_result['confidence'],
                match_result['similarity_score'],
                match_result['method']
            )
            
            return {
                'method': 'matched_existing',
                'session_speaker': session_speaker,
                'persistent_speaker_id': match_result['persistent_speaker_id'],
                'confidence': match_result['confidence'],
                'similarity_score': match_result['similarity_score']
            }
        
        # Step 2: No good match found - check for medium confidence matches
        medium_confidence_matches = await self._find_medium_confidence_matches(
            segments, audio_path, confidence_threshold * 0.8  # 80% of threshold
        )
        
        if medium_confidence_matches and not auto_enroll:
            # Queue for manual review
            await self._add_to_review_queue(
                recording_id, session_speaker, segments, medium_confidence_matches
            )
            
            return {
                'method': 'needs_review',
                'session_speaker': session_speaker,
                'suggested_matches': medium_confidence_matches,
                'reason': 'medium_confidence_matches'
            }
        
        # Step 3: No match or auto-enrollment enabled - enroll as new speaker
        if auto_enroll:
            enrollment_result = await self.persistent_speaker_manager.enroll_speaker_from_segments(
                session_speaker,
                segments,
                recording_id,
                audio_path,
                speaker_name=None  # Will be auto-generated (SPEAKER_001, etc.)
            )
            
            if enrollment_result['success']:
                # Create mapping for new speaker
                await self.persistent_speaker_manager.create_speaker_mapping(
                    recording_id,
                    session_speaker,
                    enrollment_result['persistent_speaker_id'],
                    1.0,  # High confidence for new enrollment
                    1.0,  # Perfect similarity for self-enrollment
                    'auto_enrolled'
                )
                
                logger.info(f"Auto-enrolled {session_speaker} as {enrollment_result['persistent_speaker_id']}")
                
                return {
                    'method': 'enrolled_new',
                    'session_speaker': session_speaker,
                    'persistent_speaker_id': enrollment_result['persistent_speaker_id'],
                    'enrollment_quality': enrollment_result['overall_quality'],
                    'segments_used': enrollment_result['embeddings_count']
                }
            else:
                logger.warning(f"Failed to enroll {session_speaker}: {enrollment_result['message']}")
        
        # Step 4: Fallback - queue for manual review
        await self._add_to_review_queue(
            recording_id, session_speaker, segments, medium_confidence_matches or []
        )
        
        return {
            'method': 'needs_review',
            'session_speaker': session_speaker,
            'suggested_matches': medium_confidence_matches or [],
            'reason': 'enrollment_failed' if auto_enroll else 'auto_enroll_disabled'
        }
    
    async def _find_medium_confidence_matches(
        self,
        segments: List[Dict[str, Any]],
        audio_path: str,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Find medium confidence matches for manual review."""
        try:
            # Get all persistent speakers
            persistent_speakers = await self.db.persistent_speakers.get_all_persistent_speakers()
            
            if not persistent_speakers:
                return []
            
            # Find best segment for identification
            best_segment = max(segments, key=lambda s: s['end_time'] - s['start_time'])
            
            # Extract and identify
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(best_segment['start_time'] * 1000)
            end_ms = int(best_segment['end_time'] * 1000)
            segment_audio = audio[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                segment_audio.export(temp_file.name, format="wav")
                
                identification_speakers = []
                for speaker in persistent_speakers:
                    embeddings = await self.db.speaker_embeddings.get_embeddings_as_numpy(speaker.id)
                    if embeddings:
                        identification_speakers.append({
                            'speaker_id': speaker.id,
                            'speaker_name': speaker.name or speaker.id,
                            'embeddings': embeddings,
                            'enrollment_threshold': speaker.confidence_threshold
                        })
                
                result = self.speaker_identifier.identify_speaker(
                    temp_file.name, identification_speakers
                )
                
                os.unlink(temp_file.name)
                
                # Find candidates above minimum confidence
                candidates = []
                if result['all_scores']:
                    for speaker_name, score_data in result['all_scores'].items():
                        if score_data['max_score'] >= min_confidence:
                            candidates.append({
                                'persistent_speaker_id': speaker_name,
                                'confidence': score_data['max_score'],
                                'similarity_score': score_data['max_score']
                            })
                
                # Sort by confidence
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                return candidates[:3]  # Top 3 candidates
                
        except Exception as e:
            logger.error(f"Medium confidence matching failed: {e}")
            return []
    
    async def _add_to_review_queue(
        self,
        recording_id: int,
        session_speaker: str,
        segments: List[Dict[str, Any]],
        suggested_matches: List[Dict[str, Any]]
    ) -> None:
        """Add speaker assignment to review queue."""
        try:
            total_duration = sum(seg['end_time'] - seg['start_time'] for seg in segments)
            avg_quality = np.mean([0.7] * len(segments))  # Placeholder quality score
            
            await self.db.speaker_review_queue.add_to_review_queue(
                recording_id=recording_id,
                session_speaker_label=session_speaker,
                suggested_assignments=suggested_matches,
                segment_count=len(segments),
                total_duration=total_duration,
                audio_quality=avg_quality,
                priority=1 if len(suggested_matches) > 0 else 2
            )
            
        except Exception as e:
            logger.error(f"Failed to add to review queue: {e}")
    
    async def _apply_persistent_assignments(
        self,
        segments: List[Dict[str, Any]],
        persistent_assignments: Dict[str, Any],
        enrollment_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply persistent speaker assignments to segments."""
        enhanced_segments = []
        
        for segment in segments:
            enhanced_segment = segment.copy()
            session_speaker = segment['speaker_id']
            
            # Check for persistent assignment
            if session_speaker in persistent_assignments:
                assignment = persistent_assignments[session_speaker]
                enhanced_segment['persistent_speaker_id'] = assignment['persistent_speaker_id']
                enhanced_segment['assignment_confidence'] = assignment['confidence']
                enhanced_segment['assignment_method'] = 'matched_existing'
                
                # Get speaker name and update speaker_id to show actual name
                speaker = await self.db.persistent_speakers.get_persistent_speaker(
                    assignment['persistent_speaker_id']
                )
                if speaker:
                    speaker_name = speaker.name or speaker.id
                else:
                    speaker_name = assignment['persistent_speaker_id']  # Fallback
                enhanced_segment['persistent_speaker_name'] = speaker_name
                enhanced_segment['speaker_id'] = speaker_name  # Update speaker_id to show actual name
                
            elif session_speaker in enrollment_results:
                enrollment = enrollment_results[session_speaker]
                enhanced_segment['persistent_speaker_id'] = enrollment['persistent_speaker_id']
                enhanced_segment['assignment_confidence'] = 1.0
                enhanced_segment['assignment_method'] = 'enrolled_new'
                
                # Get enrolled speaker name and update speaker_id
                speaker = await self.db.persistent_speakers.get_persistent_speaker(
                    enrollment['persistent_speaker_id']
                )
                if speaker:
                    speaker_name = speaker.name or speaker.id
                else:
                    speaker_name = enrollment['persistent_speaker_id']  # Fallback
                enhanced_segment['persistent_speaker_name'] = speaker_name
                enhanced_segment['speaker_id'] = speaker_name  # Update speaker_id to show actual name
            
            else:
                # No assignment - mark as unassigned
                enhanced_segment['persistent_speaker_id'] = None
                enhanced_segment['assignment_confidence'] = 0.0
                enhanced_segment['assignment_method'] = 'unassigned'
                enhanced_segment['persistent_speaker_name'] = None
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _generate_enhanced_speaker_summary(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate speaker summary with persistent speaker information."""
        speakers = {}
        
        for segment in segments:
            # Use persistent speaker ID if available, otherwise session speaker
            speaker_key = segment.get('persistent_speaker_id') or segment['speaker_id']
            
            if speaker_key not in speakers:
                speakers[speaker_key] = {
                    'speaker_id': segment['speaker_id'],  # Original session speaker
                    'persistent_speaker_id': segment.get('persistent_speaker_id'),
                    'persistent_speaker_name': segment.get('persistent_speaker_name'),
                    'assignment_method': segment.get('assignment_method'),
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'total_speaking_time': segment['end_time'] - segment['start_time'],
                    'avg_confidence': segment.get('assignment_confidence', 0.0),
                    'segment_count': 1
                }
            else:
                # Update timing and confidence
                speakers[speaker_key]['start_time'] = min(
                    speakers[speaker_key]['start_time'], segment['start_time']
                )
                speakers[speaker_key]['end_time'] = max(
                    speakers[speaker_key]['end_time'], segment['end_time']
                )
                speakers[speaker_key]['total_speaking_time'] += (
                    segment['end_time'] - segment['start_time']
                )
                
                # Average confidence
                current_count = speakers[speaker_key]['segment_count']
                current_avg = speakers[speaker_key]['avg_confidence']
                new_confidence = segment.get('assignment_confidence', 0.0)
                speakers[speaker_key]['avg_confidence'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
                speakers[speaker_key]['segment_count'] += 1
        
        return list(speakers.values())