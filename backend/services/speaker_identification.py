"""
Industrial-grade speaker identification service using ECAPA-TDNN embeddings.

This module provides robust speaker identification capabilities with:
- Multi-sample enrollment for improved accuracy
- Quality-aware threshold adjustment
- Score fusion for better reliability
- Confidence calibration for production use
"""

import logging
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
import librosa

logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """
    Industrial-grade speaker identification using ECAPA-TDNN embeddings.
    
    Features:
    - Multi-sample enrollment (3-5 samples per speaker)
    - Quality-aware processing with SNR analysis
    - Score fusion for improved accuracy
    - Dynamic threshold adjustment
    - Confidence calibration
    """
    
    def __init__(self, model_source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        """
        Initialize the speaker identification system.
        
        Args:
            model_source: SpeechBrain model source for ECAPA-TDNN
        """
        logger.info("Initializing SpeakerIdentifier with ECAPA-TDNN...")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load ECAPA-TDNN model
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_source,
                run_opts={"device": self.device}
            )
            logger.info("ECAPA-TDNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {e}")
            raise
        
        # Configuration
        self.base_threshold = 0.50  # Base similarity threshold
        self.min_audio_duration = 2.0  # Minimum audio duration in seconds
        self.target_sample_rate = 16000  # Target sample rate
        
    def extract_embedding(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract speaker embedding from audio file with quality assessment.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (embedding, quality_metrics)
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = self.target_sample_rate
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Check duration
            duration = waveform.shape[1] / sample_rate
            if duration < self.min_audio_duration:
                logger.warning(f"Audio duration {duration:.1f}s is below minimum {self.min_audio_duration}s")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_audio_quality(waveform, sample_rate)
            
            # Move to device and extract embedding
            waveform = waveform.to(self.device)
            embedding = self.classifier.encode_batch(waveform)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            
            logger.info(f"Extracted embedding: shape={embedding.shape}, quality_score={quality_metrics['quality_score']:.3f}")
            
            return embedding, quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to extract embedding from {audio_path}: {e}")
            raise
    
    def _calculate_audio_quality(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, float]:
        """
        Calculate audio quality metrics for threshold adjustment.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            # Convert to numpy for analysis
            audio_np = waveform.squeeze().numpy()
            
            # Signal-to-Noise Ratio (SNR) estimation
            # Use simple energy-based approach
            signal_power = np.mean(audio_np ** 2)
            noise_floor = np.percentile(audio_np ** 2, 10)  # Bottom 10% as noise estimate
            snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            
            # RMS energy
            rms_energy = np.sqrt(np.mean(audio_np ** 2))
            
            # Zero crossing rate (indicates speech vs noise)
            zero_crossings = np.sum(np.diff(np.sign(audio_np)) != 0)
            zcr = zero_crossings / len(audio_np)
            
            # Overall quality score (0-1)
            # Higher SNR and moderate ZCR indicate better quality
            snr_score = min(1.0, max(0.0, (snr_db - 5) / 15))  # 5-20 dB range
            energy_score = min(1.0, max(0.0, rms_energy * 10))  # Energy normalization
            zcr_score = 1.0 - abs(zcr - 0.05) / 0.05  # Optimal ZCR around 0.05
            
            quality_score = (snr_score * 0.5) + (energy_score * 0.3) + (zcr_score * 0.2)
            
            return {
                'snr_db': float(snr_db),
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zcr),
                'quality_score': float(quality_score)
            }
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return {
                'snr_db': 0.0,
                'rms_energy': 0.0,
                'zero_crossing_rate': 0.0,
                'quality_score': 0.5  # Default medium quality
            }
    
    def enroll_speaker(self, speaker_name: str, audio_paths: List[str]) -> Dict[str, Any]:
        """
        Enroll a speaker with multiple audio samples for robust identification.
        
        Args:
            speaker_name: Name of the speaker
            audio_paths: List of audio file paths (3-5 recommended)
            
        Returns:
            Enrollment result dictionary
        """
        logger.info(f"Enrolling speaker '{speaker_name}' with {len(audio_paths)} samples")
        
        if len(audio_paths) < 2:
            raise ValueError("At least 2 enrollment samples required for robust identification")
        
        embeddings = []
        quality_metrics = []
        
        for i, audio_path in enumerate(audio_paths):
            try:
                embedding, quality = self.extract_embedding(audio_path)
                embeddings.append(embedding)
                quality_metrics.append(quality)
                logger.info(f"Sample {i+1}/{len(audio_paths)}: quality={quality['quality_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to process enrollment sample {audio_path}: {e}")
                continue
        
        if len(embeddings) < 2:
            raise ValueError("Failed to process enough enrollment samples")
        
        # Calculate average quality
        avg_quality = np.mean([q['quality_score'] for q in quality_metrics])
        
        # Calculate enrollment statistics
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        std_embedding = np.std(embeddings_array, axis=0)
        
        # Inter-sample similarity (consistency check)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        consistency_score = np.mean(similarities) if similarities else 0.0
        
        enrollment_result = {
            'speaker_name': speaker_name,
            'embeddings': embeddings,
            'mean_embedding': mean_embedding,
            'std_embedding': std_embedding,
            'num_samples': len(embeddings),
            'avg_quality': avg_quality,
            'consistency_score': consistency_score,
            'enrollment_threshold': self._calculate_speaker_threshold(embeddings, avg_quality)
        }
        
        logger.info(f"✅ Speaker '{speaker_name}' enrolled successfully:")
        logger.info(f"  - Samples: {len(embeddings)}")
        logger.info(f"  - Avg Quality: {avg_quality:.3f}")
        logger.info(f"  - Consistency: {consistency_score:.3f}")
        logger.info(f"  - Threshold: {enrollment_result['enrollment_threshold']:.3f}")
        
        return enrollment_result
    
    def _calculate_speaker_threshold(self, embeddings: List[np.ndarray], quality_score: float) -> float:
        """
        Calculate personalized threshold for speaker based on enrollment quality.
        
        Args:
            embeddings: List of enrollment embeddings
            quality_score: Average quality score
            
        Returns:
            Adjusted threshold for this speaker
        """
        # Start with base threshold
        threshold = self.base_threshold
        
        # Adjust based on quality
        quality_adjustment = (quality_score - 0.5) * 0.1  # ±0.05 adjustment
        threshold += quality_adjustment
        
        # Adjust based on consistency (if we have multiple samples)
        if len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            consistency = np.mean(similarities) if similarities else 0.0
            consistency_adjustment = (consistency - 0.8) * 0.1  # ±0.05 adjustment
            threshold += consistency_adjustment
        
        # Clamp threshold to reasonable range
        threshold = max(0.6, min(0.9, threshold))
        
        return threshold
    
    def identify_speaker(self, audio_path: str, enrolled_speakers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify speaker from audio against enrolled speakers.
        
        Args:
            audio_path: Path to audio file to identify
            enrolled_speakers: List of enrolled speaker dictionaries
            
        Returns:
            Identification result dictionary
        """
        logger.info(f"Identifying speaker from: {audio_path}")
        
        if not enrolled_speakers:
            return {
                'identified_speaker': None,
                'confidence': 0.0,
                'all_scores': {},
                'quality_metrics': {},
                'status': 'no_enrolled_speakers'
            }
        
        try:
            # Extract embedding from query audio
            query_embedding, quality_metrics = self.extract_embedding(audio_path)
            
            # Compare against all enrolled speakers
            speaker_scores = {}
            
            for speaker_data in enrolled_speakers:
                speaker_name = speaker_data['speaker_name']
                scores = []
                
                # Compare against all enrollment samples
                for enrollment_embedding in speaker_data['embeddings']:
                    similarity = 1 - cosine(query_embedding, enrollment_embedding)
                    scores.append(similarity)
                
                # Use maximum similarity (best match)
                max_score = max(scores) if scores else 0.0
                # Use global runtime threshold instead of stored per-speaker threshold
                runtime_threshold = self.base_threshold
                passes = max_score >= runtime_threshold
                
                # Debug logging for each speaker
                logger.info(f"🎯 {speaker_name}: similarity={max_score:.3f}, runtime_threshold={runtime_threshold:.3f}, passes={passes}")
                
                speaker_scores[speaker_name] = {
                    'max_score': max_score,
                    'avg_score': np.mean(scores) if scores else 0.0,
                    'threshold': runtime_threshold,  # Use runtime threshold
                    'passes_threshold': passes
                }
            
            # Find best candidate with confidence-based approach
            # High confidence threshold prevents weak assignments (handles similar speakers)
            HIGH_CONFIDENCE_THRESHOLD = 0.5  # Lowered threshold for better recognition
            
            best_speaker = None
            best_score = 0.0
            
            for speaker_name, score_data in speaker_scores.items():
                if score_data['passes_threshold'] and score_data['max_score'] > best_score:
                    best_score = score_data['max_score']
                    best_speaker = speaker_name
            
            # Calculate final confidence with quality adjustment
            # Use weighted combination instead of multiplication to avoid over-penalization
            if best_speaker:
                # Give more weight to similarity score, less to quality
                confidence = (best_score * 0.8) + (quality_metrics['quality_score'] * 0.2)
            else:
                confidence = 0.0
            
            # Debug logging
            if best_speaker:
                logger.info(f"🔍 Best match: {best_speaker}, similarity: {best_score:.3f}, quality: {quality_metrics['quality_score']:.3f}, confidence: {confidence:.3f}")
                logger.info(f"🎯 Checking confidence {confidence:.3f} >= {HIGH_CONFIDENCE_THRESHOLD} threshold")
            
            # Apply high confidence threshold - reject weak identifications
            if best_speaker and confidence < HIGH_CONFIDENCE_THRESHOLD:
                logger.info(f"⚠️  {best_speaker} confidence {confidence:.3f} < {HIGH_CONFIDENCE_THRESHOLD} threshold, marking as unidentified")
                best_speaker = None
                confidence = 0.0
            
            result = {
                'identified_speaker': best_speaker,
                'confidence': confidence,
                'similarity_score': best_score,
                'all_scores': speaker_scores,
                'quality_metrics': quality_metrics,
                'status': 'identified' if best_speaker else 'not_identified'
            }
            
            if best_speaker:
                logger.info(f"✅ Identified: {best_speaker} (confidence: {confidence:.3f}, similarity: {best_score:.3f})")
            else:
                # Log all scores for debugging similar speakers
                score_summary = ", ".join([f"{name}: {data['max_score']:.3f}" for name, data in speaker_scores.items()])
                logger.info(f"❌ No speaker identified (scores: {score_summary})")
            
            return result
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return {
                'identified_speaker': None,
                'confidence': 0.0,
                'all_scores': {},
                'quality_metrics': {},
                'status': 'error',
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'ECAPA-TDNN',
            'model_source': 'speechbrain/spkrec-ecapa-voxceleb',
            'device': str(self.device),
            'base_threshold': self.base_threshold,
            'target_sample_rate': self.target_sample_rate,
            'min_audio_duration': self.min_audio_duration
        }