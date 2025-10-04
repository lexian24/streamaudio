"""
Persistent Speaker Manager for StreamAudio application.

This module handles the intelligent enrollment of speakers from recordings,
quality filtering of segments, and session-to-persistent speaker mapping.
"""

import logging
import numpy as np
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydub import AudioSegment
from scipy.spatial.distance import cosine

from database.services import DatabaseService
from .speaker_identification import SpeakerIdentifier

logger = logging.getLogger(__name__)


class SegmentQualityAssessor:
    """
    Assesses the quality of audio segments for speaker enrollment.
    
    Filters out poor quality segments that would negatively impact
    speaker recognition accuracy.
    """
    
    @staticmethod
    def assess_segment_quality(
        audio_path: str,
        segment_start: float,
        segment_end: float,
        min_duration: float = 1.5
    ) -> Dict[str, Any]:
        """
        Assess the quality of an audio segment for speaker enrollment.
        
        Args:
            audio_path: Path to the source audio file
            segment_start: Segment start time in seconds
            segment_end: Segment end time in seconds
            min_duration: Minimum acceptable segment duration
            
        Returns:
            Quality assessment dictionary with scores and filters
        """
        try:
            duration = segment_end - segment_start
            
            # Basic duration filter
            if duration < min_duration:
                return {
                    'overall_score': 0.0,
                    'duration': duration,
                    'passes_filters': False,
                    'rejection_reason': 'too_short',
                    'filters': {
                        'duration_ok': False,
                        'signal_quality_ok': False,
                        'speech_clarity_ok': False
                    }
                }
            
            # Extract segment for detailed analysis
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(segment_start * 1000)
            end_ms = int(segment_end * 1000)
            segment_audio = audio[start_ms:end_ms]
            
            # Convert to numpy for analysis
            audio_data = np.array(segment_audio.get_array_of_samples())
            if len(audio_data) == 0:
                return {
                    'overall_score': 0.0,
                    'duration': duration,
                    'passes_filters': False,
                    'rejection_reason': 'empty_audio',
                    'filters': {'duration_ok': True, 'signal_quality_ok': False, 'speech_clarity_ok': False}
                }
            
            # Signal quality assessment
            rms_energy = np.sqrt(np.mean(audio_data.astype(float) ** 2))
            max_amplitude = np.max(np.abs(audio_data))
            dynamic_range = max_amplitude / (rms_energy + 1e-10)
            
            # Signal-to-noise estimation
            signal_power = np.mean(audio_data.astype(float) ** 2)
            noise_floor = np.percentile(audio_data.astype(float) ** 2, 10)
            snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            
            # Zero crossing rate (speech characteristics)
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zcr = zero_crossings / len(audio_data)
            
            # Quality scores (0-1 scale)
            energy_score = min(1.0, rms_energy / 1000)  # Normalize based on typical speech levels
            snr_score = min(1.0, max(0.0, (snr_db - 10) / 20))  # 10-30 dB range
            dynamic_score = min(1.0, max(0.0, (dynamic_range - 2) / 8))  # 2-10 range
            zcr_score = 1.0 - abs(zcr - 0.05) / 0.05  # Optimal around 0.05
            
            # Apply filters
            filters = {
                'duration_ok': duration >= min_duration,
                'signal_quality_ok': snr_db > 15 and rms_energy > 100,
                'speech_clarity_ok': 0.02 < zcr < 0.15 and dynamic_range > 2
            }
            
            # Overall quality score
            overall_score = (
                energy_score * 0.3 +
                snr_score * 0.4 +
                dynamic_score * 0.2 +
                zcr_score * 0.1
            )
            
            passes_filters = all(filters.values())
            
            return {
                'overall_score': overall_score,
                'duration': duration,
                'snr_db': snr_db,
                'rms_energy': rms_energy,
                'dynamic_range': dynamic_range,
                'zero_crossing_rate': zcr,
                'passes_filters': passes_filters,
                'rejection_reason': None if passes_filters else 'quality_filters',
                'filters': filters
            }
            
        except Exception as e:
            logger.warning(f"Quality assessment failed for segment {segment_start}-{segment_end}: {e}")
            return {
                'overall_score': 0.0,
                'duration': segment_end - segment_start,
                'passes_filters': False,
                'rejection_reason': 'assessment_error',
                'filters': {'duration_ok': False, 'signal_quality_ok': False, 'speech_clarity_ok': False}
            }


class SegmentSelector:
    """
    Selects the best segments for speaker enrollment from a collection of candidates.
    
    Uses quality scoring and diversity criteria to choose representative samples.
    """
    
    @staticmethod
    def select_best_segments(
        segments_with_quality: List[Dict[str, Any]],
        max_segments: int = 5,
        min_segments: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Select the best segments for speaker enrollment.
        
        Args:
            segments_with_quality: List of segment dicts with quality scores
            max_segments: Maximum number of segments to select
            min_segments: Minimum number of segments required
            
        Returns:
            List of selected segments sorted by quality
        """
        # Filter out segments that don't pass quality filters
        valid_segments = [
            seg for seg in segments_with_quality 
            if seg['quality_assessment']['passes_filters']
        ]
        
        if len(valid_segments) < min_segments:
            logger.warning(f"Only {len(valid_segments)} valid segments, need at least {min_segments}")
            return []
        
        # Score segments for selection
        scored_segments = []
        for segment in valid_segments:
            quality = segment['quality_assessment']
            
            # Combined selection score
            selection_score = (
                quality['overall_score'] * 0.6 +        # Quality is most important
                min(quality['duration'] / 5.0, 1.0) * 0.3 +  # Longer is better (up to 5s)
                (1.0 if quality['snr_db'] > 20 else 0.5) * 0.1  # Bonus for high SNR
            )
            
            scored_segments.append((selection_score, segment))
        
        # Sort by score and take top segments
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        selected = [seg for score, seg in scored_segments[:max_segments]]
        
        logger.info(f"Selected {len(selected)} segments from {len(valid_segments)} valid candidates")
        
        return selected


class PersistentSpeakerManager:
    """
    Main manager for persistent speaker operations.

    Handles speaker enrollment from recordings, session-to-persistent mapping,
    and maintains the persistent speaker database.
    """

    def __init__(self, db_service: DatabaseService, speaker_identifier: SpeakerIdentifier):
        self.db = db_service
        self.speaker_identifier = speaker_identifier
        self.quality_assessor = SegmentQualityAssessor()
        self.segment_selector = SegmentSelector()

    def _get_persistent_speakers_sync(self, active_only: bool = True):
        """Synchronous method to get persistent speakers for Celery tasks."""
        from database.models import PersistentSpeaker
        from sqlalchemy.orm import Session

        if isinstance(self.db.db, Session):
            # Sync session (Celery context)
            query = self.db.db.query(PersistentSpeaker)
            if active_only:
                query = query.filter(PersistentSpeaker.is_active == True)
            return query.order_by(PersistentSpeaker.last_seen_at.desc()).all()
        else:
            # This shouldn't happen, but fall back to empty list
            logger.warning("Expected sync session but got async session")
            return []

    def _get_speaker_embeddings_sync(self, speaker_id: str):
        """Synchronous method to get speaker embeddings for Celery tasks."""
        from database.models import SpeakerEmbedding
        from sqlalchemy.orm import Session
        import io

        if isinstance(self.db.db, Session):
            # Sync session (Celery context)
            embeddings = self.db.db.query(SpeakerEmbedding).filter(
                SpeakerEmbedding.speaker_id == speaker_id
            ).all()

            if embeddings:
                # Properly decompress embeddings (they were saved with np.save, not raw bytes)
                decompressed = []
                for e in embeddings:
                    buffer = io.BytesIO(e.embedding)
                    embedding = np.load(buffer).astype(np.float32)
                    decompressed.append(embedding)
                return np.array(decompressed)
            return None
        else:
            logger.warning("Expected sync session but got async session")
            return None
    
    async def enroll_speaker_from_segments(
        self,
        session_speaker_label: str,
        segments: List[Dict[str, Any]],
        recording_id: int,
        audio_path: str,
        speaker_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enroll a speaker from recording segments with intelligent quality filtering.
        
        Args:
            session_speaker_label: Session speaker ID (e.g., "SPEAKER_00")
            segments: List of segment dictionaries with timing info
            recording_id: ID of the source recording
            audio_path: Path to the source audio file
            speaker_name: Optional name for the speaker
            
        Returns:
            Enrollment result dictionary
        """
        try:
            logger.info(f"Enrolling speaker from {len(segments)} segments: {session_speaker_label}")
            
            # Step 1: Assess quality of all segments
            segments_with_quality = []
            for segment in segments:
                quality_assessment = self.quality_assessor.assess_segment_quality(
                    audio_path,
                    segment['start_time'],
                    segment['end_time']
                )
                
                segments_with_quality.append({
                    **segment,
                    'quality_assessment': quality_assessment
                })
            
            # Step 2: Select best segments for enrollment
            selected_segments = self.segment_selector.select_best_segments(
                segments_with_quality,
                max_segments=5,
                min_segments=2
            )
            
            if not selected_segments:
                return {
                    'success': False,
                    'error': 'no_quality_segments',
                    'message': 'No segments passed quality filters for enrollment'
                }
            
            # Step 3: Extract embeddings from selected segments
            embeddings = []
            embedding_metadata = []
            
            for segment in selected_segments:
                try:
                    # Extract audio segment
                    audio = AudioSegment.from_file(audio_path)
                    start_ms = int(segment['start_time'] * 1000)
                    end_ms = int(segment['end_time'] * 1000)
                    segment_audio = audio[start_ms:end_ms]
                    
                    # Save to temporary file for embedding extraction
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        segment_audio.export(temp_file.name, format="wav")
                        
                        # Extract embedding
                        embedding, quality_metrics = self.speaker_identifier.extract_embedding(temp_file.name)
                        embeddings.append(embedding)
                        
                        # Store metadata
                        embedding_metadata.append({
                            'segment_start': segment['start_time'],
                            'segment_end': segment['end_time'],
                            'duration': segment['end_time'] - segment['start_time'],
                            'quality_score': segment['quality_assessment']['overall_score'],
                            'snr_db': segment['quality_assessment']['snr_db'],
                            'embedding_quality': quality_metrics
                        })
                        
                        # Clean up temp file
                        os.unlink(temp_file.name)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract embedding from segment: {e}")
                    continue
            
            if not embeddings:
                return {
                    'success': False,
                    'error': 'embedding_extraction_failed',
                    'message': 'Failed to extract embeddings from segments'
                }
            
            # Step 4: Create persistent speaker
            overall_quality = np.mean([meta['quality_score'] for meta in embedding_metadata])
            
            persistent_speaker = await self.db.persistent_speakers.create_persistent_speaker(
                name=speaker_name,
                embeddings=embeddings,
                embedding_metadata={
                    'enrollment_segments': len(selected_segments),
                    'overall_quality': overall_quality,
                    'source_recording_id': recording_id,
                    'segment_metadata': embedding_metadata
                },
                first_seen_recording_id=recording_id,
                enrollment_method="auto_segment"
            )
            
            # Step 5: Store individual embeddings
            for i, (embedding, metadata) in enumerate(zip(embeddings, embedding_metadata)):
                await self.db.speaker_embeddings.add_embedding(
                    speaker_id=persistent_speaker.id,
                    embedding=embedding,
                    quality_score=metadata['quality_score'],
                    snr_db=metadata['snr_db'],
                    duration=metadata['duration'],
                    source_recording_id=recording_id,
                    source_segment_start=metadata['segment_start'],
                    source_segment_end=metadata['segment_end'],
                    enrollment_method="auto_segment"
                )
            
            logger.info(f"✅ Successfully enrolled speaker {persistent_speaker.id} from {len(embeddings)} segments")
            
            return {
                'success': True,
                'persistent_speaker_id': persistent_speaker.id,
                'speaker_name': persistent_speaker.name,
                'embeddings_count': len(embeddings),
                'overall_quality': overall_quality,
                'selected_segments': len(selected_segments),
                'total_segments_analyzed': len(segments)
            }
            
        except Exception as e:
            logger.error(f"Speaker enrollment failed: {e}")
            return {
                'success': False,
                'error': 'enrollment_failed',
                'message': str(e)
            }
    
    def find_matching_persistent_speaker_sync(
        self,
        session_segments: List[Dict[str, Any]],
        audio_path: str,
        confidence_threshold: float = 0.45
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous version: Find if session speaker matches any existing persistent speakers.
        Used in Celery tasks.

        Args:
            session_segments: List of segments for this session speaker
            audio_path: Path to the source audio file
            confidence_threshold: Minimum confidence for matching

        Returns:
            Matching result dictionary or None
        """
        try:
            logger.info(f"🔍 Searching for matches against enrolled speakers...")

            # Get all active persistent speakers (sync)
            persistent_speakers = self._get_persistent_speakers_sync(active_only=True)

            logger.info(f"Found {len(persistent_speakers)} enrolled speakers to check against")

            if not persistent_speakers:
                return None

            # Find best segment for identification
            best_segment = max(session_segments, key=lambda s: s['end_time'] - s['start_time'])

            # Extract audio segment
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(best_segment['start_time'] * 1000)
            end_ms = int(best_segment['end_time'] * 1000)
            segment_audio = audio[start_ms:end_ms]

            # Save to temporary file for identification
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                segment_audio.export(temp_file.name, format="wav")

                # Prepare persistent speakers for identification
                identification_speakers = []
                for speaker in persistent_speakers:
                    embeddings = self._get_speaker_embeddings_sync(speaker.id)
                    if embeddings is not None and len(embeddings) > 0:
                        identification_speakers.append({
                            'speaker_id': speaker.id,
                            'speaker_name': speaker.name or speaker.id,
                            'embeddings': embeddings,
                            'enrollment_threshold': speaker.confidence_threshold
                        })

                # Run identification
                result = self.speaker_identifier.identify_speaker(
                    temp_file.name,
                    identification_speakers
                )

                # Clean up temp file
                os.unlink(temp_file.name)

                # Debug logging
                logger.info(f"🔍 Speaker identification result: identified={result['identified_speaker']}, confidence={result['confidence']:.3f}, threshold={confidence_threshold:.3f}")

                # Check if match meets threshold
                if (result['identified_speaker'] and
                    result['confidence'] >= confidence_threshold):

                    logger.info(f"✅ Match found: {result['identified_speaker']} with confidence {result['confidence']:.3f}")
                    return {
                        'persistent_speaker_id': result['identified_speaker'],
                        'confidence': result['confidence'],
                        'similarity_score': result['similarity_score'],
                        'method': 'auto_identification'
                    }

                logger.info(f"❌ No match: identified={result['identified_speaker']}, confidence={result['confidence']:.3f} < threshold={confidence_threshold:.3f}")
                return None

        except Exception as e:
            logger.error(f"Speaker matching failed: {e}")
            return None

    async def find_matching_persistent_speaker(
        self,
        session_segments: List[Dict[str, Any]],
        audio_path: str,
        confidence_threshold: float = 0.45
    ) -> Optional[Dict[str, Any]]:
        """
        Async version: Find if session speaker matches any existing persistent speakers.
        Used in FastAPI routes.

        Args:
            session_segments: List of segments for this session speaker
            audio_path: Path to the source audio file
            confidence_threshold: Minimum confidence for matching

        Returns:
            Matching result dictionary or None
        """
        try:
            logger.info(f"🔍 Searching for matches against enrolled speakers...")

            # Get all active persistent speakers (async)
            persistent_speakers = await self.db.persistent_speakers.get_all_persistent_speakers(
                active_only=True
            )
            
            logger.info(f"Found {len(persistent_speakers)} enrolled speakers to check against")
            
            if not persistent_speakers:
                return None
            
            # Find best segment for identification
            best_segment = max(session_segments, key=lambda s: s['end_time'] - s['start_time'])
            
            # Extract audio segment
            audio = AudioSegment.from_file(audio_path)
            start_ms = int(best_segment['start_time'] * 1000)
            end_ms = int(best_segment['end_time'] * 1000)
            segment_audio = audio[start_ms:end_ms]
            
            # Save to temporary file for identification
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                segment_audio.export(temp_file.name, format="wav")
                
                # Prepare persistent speakers for identification
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
                
                # Run identification
                result = self.speaker_identifier.identify_speaker(
                    temp_file.name,
                    identification_speakers
                )
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                # Debug logging
                logger.info(f"🔍 Speaker identification result: identified={result['identified_speaker']}, confidence={result['confidence']:.3f}, threshold={confidence_threshold:.3f}")
                
                # Check if match meets threshold
                if (result['identified_speaker'] and 
                    result['confidence'] >= confidence_threshold):
                    
                    logger.info(f"✅ Match found: {result['identified_speaker']} with confidence {result['confidence']:.3f}")
                    return {
                        'persistent_speaker_id': result['identified_speaker'],
                        'confidence': result['confidence'],
                        'similarity_score': result['similarity_score'],
                        'method': 'auto_identification'
                    }
                
                logger.info(f"❌ No match: identified={result['identified_speaker']}, confidence={result['confidence']:.3f} < threshold={confidence_threshold:.3f}")
                return None
                
        except Exception as e:
            logger.error(f"Speaker matching failed: {e}")
            return None
    
    def create_speaker_mapping_sync(
        self,
        recording_id: int,
        session_speaker_label: str,
        persistent_speaker_id: str,
        confidence: float,
        similarity_score: float,
        method: str = "auto"
    ) -> None:
        """Synchronous version: Create a session-to-persistent speaker mapping for Celery tasks."""
        from database.models import SpeakerMapping
        from sqlalchemy.orm import Session
        from datetime import datetime

        try:
            if isinstance(self.db.db, Session):
                # Sync session (Celery context)
                mapping = SpeakerMapping(
                    recording_id=recording_id,
                    session_speaker_label=session_speaker_label,
                    persistent_speaker_id=persistent_speaker_id,
                    assignment_confidence=confidence,
                    assignment_method=method,
                    similarity_score=similarity_score,
                    needs_review=(confidence < 0.85),
                    assigned_at=datetime.utcnow()
                )
                self.db.db.add(mapping)
                self.db.db.commit()

                logger.info(f"✅ Created speaker mapping: {session_speaker_label} -> {persistent_speaker_id} (confidence: {confidence:.3f})")
            else:
                logger.warning("Expected sync session but got async session")
        except Exception as e:
            logger.error(f"Failed to create speaker mapping: {e}")
            if isinstance(self.db.db, Session):
                self.db.db.rollback()
            raise

    async def create_speaker_mapping(
        self,
        recording_id: int,
        session_speaker_label: str,
        persistent_speaker_id: str,
        confidence: float,
        similarity_score: float,
        method: str = "auto"
    ) -> None:
        """Create a session-to-persistent speaker mapping."""
        try:
            await self.db.speaker_mappings.create_mapping(
                recording_id=recording_id,
                session_speaker_label=session_speaker_label,
                persistent_speaker_id=persistent_speaker_id,
                assignment_confidence=confidence,
                assignment_method=method,
                similarity_score=similarity_score,
                needs_review=(confidence < 0.85)  # Flag for review if medium confidence
            )

            logger.info(f"✅ Created speaker mapping: {session_speaker_label} -> {persistent_speaker_id} (confidence: {confidence:.3f})")

        except Exception as e:
            logger.error(f"Failed to create speaker mapping: {e}")
            raise
    
    async def enroll_speaker_from_files(
        self,
        speaker_name: str,
        audio_files: List[str]
    ) -> Dict[str, Any]:
        """
        Enroll a speaker from provided audio files.
        
        Args:
            speaker_name: Name of the speaker to enroll
            audio_files: List of audio file paths
            
        Returns:
            Enrollment result dictionary
        """
        try:
            logger.info(f"Enrolling speaker '{speaker_name}' from {len(audio_files)} files")
            
            # Extract embeddings from each file
            embeddings = []
            quality_scores = []
            
            for audio_file in audio_files:
                try:
                    # Generate embedding for this file
                    embedding, quality_info = self.speaker_identifier.extract_embedding(audio_file)
                    if embedding is not None:
                        embeddings.append(embedding)
                        # Use actual quality score from speaker identifier
                        quality_score = quality_info.get('overall_quality', 0.8)
                        quality_scores.append(quality_score)
                        logger.debug(f"Generated embedding for {audio_file}")
                    else:
                        logger.warning(f"Failed to generate embedding for {audio_file}")
                        
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    continue
            
            if len(embeddings) < 2:
                raise ValueError(f"Need at least 2 valid audio files, got {len(embeddings)}")
            
            # Convert to numpy arrays
            import numpy as np
            embeddings = [np.array(emb) for emb in embeddings]
            
            # Calculate enrollment quality
            avg_quality = np.mean(quality_scores)
            consistency_score = self._calculate_embedding_consistency(embeddings)
            
            # Create persistent speaker
            persistent_speaker = await self.db.persistent_speakers.create_persistent_speaker(
                name=speaker_name,
                embeddings=embeddings
            )
            
            # Save individual embeddings
            for i, embedding in enumerate(embeddings):
                try:
                    logger.info(f"Saving embedding {i+1}/{len(embeddings)} for speaker {persistent_speaker.id}")
                    embedding_record = await self.db.speaker_embeddings.add_embedding(
                        speaker_id=persistent_speaker.id,
                        embedding=embedding,
                        quality_score=quality_scores[i]
                    )
                    logger.info(f"✅ Saved embedding {embedding_record.id} successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to save embedding {i+1}: {e}")
                    raise
            
            logger.info(f"Successfully enrolled speaker {persistent_speaker.id} ({speaker_name}) with {len(embeddings)} embeddings")

            return {
                "status": "enrolled",
                "speaker_id": persistent_speaker.id,
                "speaker_name": speaker_name,
                "quality_score": avg_quality,
                "consistency_score": consistency_score,
                "embeddings_count": len(embeddings),
                "created_at": persistent_speaker.created_at.isoformat() if persistent_speaker.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Failed to enroll speaker from files: {e}")
            raise
    
    def _calculate_embedding_consistency(self, embeddings: List[np.ndarray]) -> float:
        """Calculate consistency score between embeddings."""
        if len(embeddings) < 2:
            return 1.0
        
        import numpy as np
        from scipy.spatial.distance import cosine
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities))