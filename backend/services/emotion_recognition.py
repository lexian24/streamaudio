"""
Emotion Recognition Service
Lightweight emotion recognition using transformers-based models
"""
import logging
import librosa
import numpy as np
import torch
import tempfile
import soundfile as sf
import os
from typing import Tuple, Optional
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

logger = logging.getLogger(__name__)


class EmotionRecognizer:
    """
    Lightweight emotion recognition service using pretrained models
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize emotion recognition model
        
        Args:
            model_name: HuggingFace model name for emotion recognition
        """
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.audio_model = None
        self.audio_extractor = None
        
        # Emotion labels for different models
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
        
        self._load_models()
    
    def _load_models(self):
        """Load emotion recognition models"""
        try:
            logger.info("Loading emotion recognition models...")
            
            # Try to load audio-based emotion model first
            try:
                self.audio_model = AutoModelForAudioClassification.from_pretrained(
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
                self.audio_extractor = AutoFeatureExtractor.from_pretrained(
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
                logger.info("✅ Audio-based emotion model loaded")
            except Exception as e:
                logger.warning(f"Failed to load audio emotion model: {e}")
                logger.info("Will use text-based fallback only")
            
            # Load text-based emotion model as fallback
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.text_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                logger.info("✅ Text-based emotion model loaded as fallback")
            except Exception as e:
                logger.warning(f"Failed to load text emotion model: {e}")
            
            # Set models to evaluation mode
            if self.audio_model:
                self.audio_model.eval()
            if hasattr(self, 'text_model'):
                self.text_model.eval()
                
        except Exception as e:
            logger.error(f"Failed to load emotion models: {e}")
            raise
    
    def predict_emotion(self, audio_path: str, start_time: float, end_time: float, 
                       text: Optional[str] = None) -> Tuple[str, float]:
        """
        Predict emotion from audio segment and optional text
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            text: Optional transcribed text for context
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        try:
            # Try audio-based prediction first
            if self.audio_model and self.audio_extractor:
                emotion, confidence = self._predict_from_audio(audio_path, start_time, end_time)
                # Always return audio prediction if available (removed confidence filtering)
                if emotion != "neutral" or confidence > 0.3:  # Return unless very low confidence neutral
                    return emotion, confidence
            
            # Fallback to text-based prediction if available
            if text and hasattr(self, 'text_model'):
                text_emotion, text_confidence = self._predict_from_text(text)
                
                # If we have both predictions, combine them
                if 'emotion' in locals():
                    # Simple ensemble: average confidences, prefer audio if close
                    combined_confidence = (confidence + text_confidence) / 2
                    if abs(confidence - text_confidence) < 0.2:
                        return emotion, combined_confidence  # Prefer audio
                    else:
                        return (emotion if confidence > text_confidence else text_emotion), combined_confidence
                else:
                    return text_emotion, text_confidence
            
            # If audio prediction exists but low confidence, return it
            if 'emotion' in locals():
                return emotion, confidence
            
            # Default fallback - should rarely reach here
            return "neutral", 0.1
            
        except Exception as e:
            logger.error(f"Emotion prediction failed for segment [{start_time}-{end_time}]: {e}")
            return "neutral", 0.0
    
    def _predict_from_audio(self, audio_path: str, start_time: float, end_time: float) -> Tuple[str, float]:
        """Predict emotion from audio segment"""
        try:
            # Load audio segment
            audio, sr = librosa.load(
                audio_path,
                sr=16000,  # Most models expect 16kHz
                offset=start_time,
                duration=end_time - start_time
            )
            
            # Skip very short segments (but still try to predict)
            if len(audio) < 0.1 * 16000:  # Less than 0.1 seconds - too short
                return "neutral", 0.1
            
            # Process with feature extractor
            inputs = self.audio_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            # Get emotion label
            if hasattr(self.audio_model.config, 'id2label'):
                emotion = self.audio_model.config.id2label[predicted_id].lower()
            else:
                emotion = self.emotion_labels[predicted_id] if predicted_id < len(self.emotion_labels) else "neutral"
            
            # Clean up emotion label
            emotion = self._normalize_emotion_label(emotion)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Audio-based emotion prediction failed: {e}")
            return "neutral", 0.2
    
    def _predict_from_text(self, text: str) -> Tuple[str, float]:
        """Predict emotion from text"""
        try:
            if not text.strip():
                return "neutral", 0.3
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            # Get emotion label
            if hasattr(self.text_model.config, 'id2label'):
                emotion = self.text_model.config.id2label[predicted_id].lower()
            else:
                emotion = self.emotion_labels[predicted_id] if predicted_id < len(self.emotion_labels) else "neutral"
            
            emotion = self._normalize_emotion_label(emotion)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Text-based emotion prediction failed: {e}")
            return "neutral", 0.2
    
    def _normalize_emotion_label(self, emotion: str) -> str:
        """Normalize emotion labels to standard set"""
        emotion = emotion.lower().strip()
        
        # Mapping for various model outputs to standard labels
        emotion_mapping = {
            "anger": "angry",
            "joy": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "fear": "fearful",
            "disgust": "disgusted",
            "surprise": "surprised",
            "calm": "neutral",
            "excited": "happy"
        }
        
        return emotion_mapping.get(emotion, emotion)
    
    def get_info(self) -> dict:
        """Get emotion recognizer information"""
        return {
            "audio_model": "wav2vec2-speech-emotion" if self.audio_model else None,
            "text_model": self.model_name if hasattr(self, 'text_model') else None,
            "provider": "HuggingFace Transformers",
            "audio_loaded": self.audio_model is not None,
            "text_loaded": hasattr(self, 'text_model'),
            "supported_emotions": self.emotion_labels
        }