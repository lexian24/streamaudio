"""
Celery tasks for audio processing.

These tasks handle heavy audio processing operations asynchronously,
allowing the API to respond immediately while processing continues in the background.
"""

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from collections import Counter

from celery import Task
from celery_app import celery_app
from sqlalchemy.orm import Session

# Import services
from services.enhanced_audio_processor import EnhancedAudioProcessor
from database.config import SessionLocal
from database.models import ProcessingTask, Recording
from database.services import DatabaseService

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """
    Base task class with database session management and error handling.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
        # Update task status in database
        try:
            db = SessionLocal()
            task_record = db.query(ProcessingTask).filter(ProcessingTask.task_id == task_id).first()
            if task_record:
                task_record.status = 'failed'
                task_record.error_message = str(exc)
                task_record.traceback = str(einfo)
                task_record.completed_at = datetime.utcnow()
                db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Task {task_id} completed successfully")


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name='tasks.audio_tasks.process_audio_file',
    max_retries=3,
    default_retry_delay=60
)
def process_audio_file(
    self,
    recording_id: int,
    audio_path: str,
    auto_enroll_new_speakers: bool = True
) -> Dict[str, Any]:
    """
    Process uploaded audio file with full analysis pipeline.

    Args:
        self: Celery task instance
        recording_id: Database ID of the recording
        audio_path: Path to the audio file
        auto_enroll_new_speakers: Whether to auto-enroll new speakers

    Returns:
        Processing results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting audio processing task {task_id} for recording {recording_id}")

    # Create database session
    db = SessionLocal()
    db_service = DatabaseService(db)

    try:
        # Update task status to processing
        task_record = db.query(ProcessingTask).filter(ProcessingTask.task_id == task_id).first()
        if task_record:
            task_record.status = 'processing'
            task_record.started_at = datetime.utcnow()
            task_record.worker_name = self.request.hostname
            db.commit()

        # Initialize audio processor
        processor = EnhancedAudioProcessor(db_service=db_service)

        # Process audio with persistent speakers
        start_time = time.time()
        result = await_sync(
            processor.process_audio_with_persistent_speakers(
                audio_path=audio_path,
                recording_id=recording_id,
                db_service=db_service,
                auto_enroll_new_speakers=auto_enroll_new_speakers
            )
        )

        # Save processing results to database
        logger.info(f"Saving processing results to database for recording {recording_id}")

        # Prepare transcription
        transcription = ""
        if result.get('segments'):
            segment_texts = [seg.get('text', '').strip() for seg in result['segments'] if seg.get('text', '').strip()]
            transcription = " ".join(segment_texts).strip()

        # Calculate overall confidence
        confidence_score = None
        if result.get('segments'):
            confidences = [seg.get('confidence', 0) for seg in result['segments'] if seg.get('confidence')]
            if confidences:
                confidence_score = sum(confidences) / len(confidences)

        # Extract emotion data
        emotions_data = None
        dominant_emotion = None
        emotion_confidence = None
        if result.get('segments'):
            emotions = [seg.get('emotion', '') for seg in result['segments'] if seg.get('emotion')]
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]

                emotion_confidences = [
                    seg.get('emotion_confidence', 0)
                    for seg in result['segments']
                    if seg.get('emotion') == dominant_emotion
                ]
                if emotion_confidences:
                    emotion_confidence = sum(emotion_confidences) / len(emotion_confidences)

                emotions_data = {
                    'dominant_emotion': dominant_emotion,
                    'emotion_confidence': emotion_confidence,
                    'all_emotions': dict(emotion_counts)
                }

        # Prepare diarization data
        filtered_segments = [seg for seg in result.get('segments', []) if seg.get('text', '').strip()]
        diarization_data = {
            'num_speakers': len(result.get('speakers', [])),
            'speaker_segments': filtered_segments,
            'speaker_summary': result.get('speakers', [])
        }

        # Create processing result in database (synchronous)
        from database.models import ProcessingResult
        processing_result = ProcessingResult(
            recording_id=recording_id,
            transcription=transcription or None,
            confidence_score=confidence_score,
            diarization_json=diarization_data,
            num_speakers=diarization_data.get('num_speakers'),
            emotions_json=emotions_data,
            dominant_emotion=dominant_emotion,
            emotion_confidence=emotion_confidence,
            model_versions={
                "whisper": "openai/whisper-small",
                "pyannote": "pyannote/speaker-diarization-3.1",
                "emotion": "j-hartmann/emotion-english-distilroberta-base"
            },
            status="completed"
        )
        db.add(processing_result)
        db.commit()
        db.refresh(processing_result)

        # Save speaker segments
        if result.get('segments'):
            segments_to_save = []
            for segment in result['segments']:
                segment_text = segment.get('text', '').strip()
                if not segment_text:
                    continue

                segments_to_save.append({
                    'speaker_id': None,
                    'speaker_label': segment.get('persistent_speaker_name') or segment.get('speaker_id', 'Unknown'),
                    'start_time': segment.get('start_time', 0),
                    'end_time': segment.get('end_time', 0),
                    'confidence': segment.get('confidence', 0),
                    'text': segment_text
                })

            # Save speaker segments (synchronous)
            if segments_to_save:
                from database.models import SpeakerSegment
                for seg_data in segments_to_save:
                    start = seg_data.get('start_time', 0)
                    end = seg_data.get('end_time', 0)
                    speaker_segment = SpeakerSegment(
                        result_id=processing_result.id,
                        speaker_id=seg_data.get('speaker_id'),
                        speaker_label=seg_data.get('speaker_label', 'Unknown'),
                        start_time=start,
                        end_time=end,
                        duration=end - start,
                        confidence=seg_data.get('confidence', 0),
                        segment_text=seg_data.get('text', '')
                    )
                    db.add(speaker_segment)
                db.commit()

        # Add metadata to result
        result.update({
            "processing_time": time.time() - start_time,
            "status": "completed",
            "recording_id": recording_id
        })

        # Update task record with results
        if task_record:
            task_record.status = 'completed'
            task_record.result_data = result
            task_record.completed_at = datetime.utcnow()
            task_record.progress = 100
            db.commit()

        logger.info(f"✅ Task {task_id} completed successfully in {result['processing_time']:.2f}s")

        return result

    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update task status
        if task_record:
            task_record.status = 'failed'
            task_record.error_message = str(e)
            task_record.traceback = traceback.format_exc()
            task_record.completed_at = datetime.utcnow()
            db.commit()

        # Retry logic
        try:
            raise self.retry(exc=e, countdown=60)
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for task {task_id}")
            raise e

    finally:
        db.close()


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name='tasks.audio_tasks.process_vad_recording',
    max_retries=3,
    default_retry_delay=60
)
def process_vad_recording(
    self,
    recording_id: int,
    audio_path: str,
    auto_enroll_new_speakers: bool = True
) -> Dict[str, Any]:
    """
    Process VAD recording with full analysis pipeline.

    Same as process_audio_file but specifically for VAD recordings.

    Args:
        self: Celery task instance
        recording_id: Database ID of the recording
        audio_path: Path to the audio file
        auto_enroll_new_speakers: Whether to auto-enroll new speakers

    Returns:
        Processing results dictionary
    """
    # VAD recordings use the same processing pipeline as regular uploads
    task_id = self.request.id
    logger.info(f"Starting VAD recording task {task_id} for recording {recording_id}")

    # Create database session
    db = SessionLocal()
    db_service = DatabaseService(db)

    try:
        # Update task status to processing
        task_record = db.query(ProcessingTask).filter(ProcessingTask.task_id == task_id).first()
        if task_record:
            task_record.status = 'processing'
            task_record.started_at = datetime.utcnow()
            task_record.worker_name = self.request.hostname
            db.commit()

        # Initialize audio processor
        processor = EnhancedAudioProcessor(db_service=db_service)

        # Process audio with persistent speakers
        start_time = time.time()
        result = await_sync(
            processor.process_audio_with_persistent_speakers(
                audio_path=audio_path,
                recording_id=recording_id,
                db_service=db_service,
                auto_enroll_new_speakers=auto_enroll_new_speakers
            )
        )

        # Save processing results to database
        logger.info(f"Saving processing results to database for recording {recording_id}")

        # Prepare transcription
        transcription = ""
        if result.get('segments'):
            segment_texts = [seg.get('text', '').strip() for seg in result['segments'] if seg.get('text', '').strip()]
            transcription = " ".join(segment_texts).strip()

        # Calculate overall confidence
        confidence_score = None
        if result.get('segments'):
            confidences = [seg.get('confidence', 0) for seg in result['segments'] if seg.get('confidence')]
            if confidences:
                confidence_score = sum(confidences) / len(confidences)

        # Extract emotion data
        emotions_data = None
        dominant_emotion = None
        emotion_confidence = None
        if result.get('segments'):
            emotions = [seg.get('emotion', '') for seg in result['segments'] if seg.get('emotion')]
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]

                emotion_confidences = [
                    seg.get('emotion_confidence', 0)
                    for seg in result['segments']
                    if seg.get('emotion') == dominant_emotion
                ]
                if emotion_confidences:
                    emotion_confidence = sum(emotion_confidences) / len(emotion_confidences)

                emotions_data = {
                    'dominant_emotion': dominant_emotion,
                    'emotion_confidence': emotion_confidence,
                    'all_emotions': dict(emotion_counts)
                }

        # Prepare diarization data
        filtered_segments = [seg for seg in result.get('segments', []) if seg.get('text', '').strip()]
        diarization_data = {
            'num_speakers': len(result.get('speakers', [])),
            'speaker_segments': filtered_segments,
            'speaker_summary': result.get('speakers', [])
        }

        # Create processing result in database (synchronous)
        from database.models import ProcessingResult
        processing_result = ProcessingResult(
            recording_id=recording_id,
            transcription=transcription or None,
            confidence_score=confidence_score,
            diarization_json=diarization_data,
            num_speakers=diarization_data.get('num_speakers'),
            emotions_json=emotions_data,
            dominant_emotion=dominant_emotion,
            emotion_confidence=emotion_confidence,
            model_versions={
                "whisper": "openai/whisper-small",
                "pyannote": "pyannote/speaker-diarization-3.1",
                "emotion": "j-hartmann/emotion-english-distilroberta-base"
            },
            status="completed"
        )
        db.add(processing_result)
        db.commit()
        db.refresh(processing_result)

        # Save speaker segments
        if result.get('segments'):
            segments_to_save = []
            for segment in result['segments']:
                segment_text = segment.get('text', '').strip()
                if not segment_text:
                    continue

                segments_to_save.append({
                    'speaker_id': None,
                    'speaker_label': segment.get('persistent_speaker_name') or segment.get('speaker_id', 'Unknown'),
                    'start_time': segment.get('start_time', 0),
                    'end_time': segment.get('end_time', 0),
                    'confidence': segment.get('confidence', 0),
                    'text': segment_text
                })

            # Save speaker segments (synchronous)
            if segments_to_save:
                from database.models import SpeakerSegment
                for seg_data in segments_to_save:
                    start = seg_data.get('start_time', 0)
                    end = seg_data.get('end_time', 0)
                    speaker_segment = SpeakerSegment(
                        result_id=processing_result.id,
                        speaker_id=seg_data.get('speaker_id'),
                        speaker_label=seg_data.get('speaker_label', 'Unknown'),
                        start_time=start,
                        end_time=end,
                        duration=end - start,
                        confidence=seg_data.get('confidence', 0),
                        segment_text=seg_data.get('text', '')
                    )
                    db.add(speaker_segment)
                db.commit()

        # Add metadata to result
        result.update({
            "processing_time": time.time() - start_time,
            "status": "completed",
            "recording_id": recording_id
        })

        # Update task record with results
        if task_record:
            task_record.status = 'completed'
            task_record.result_data = result
            task_record.completed_at = datetime.utcnow()
            task_record.progress = 100
            db.commit()

        logger.info(f"✅ Task {task_id} completed successfully in {result['processing_time']:.2f}s")

        return result

    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update task status
        if task_record:
            task_record.status = 'failed'
            task_record.error_message = str(e)
            task_record.traceback = traceback.format_exc()
            task_record.completed_at = datetime.utcnow()
            db.commit()

        # Retry logic
        try:
            raise self.retry(exc=e, countdown=60)
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for task {task_id}")
            raise e

    finally:
        db.close()


def await_sync(coroutine):
    """
    Helper to run async functions in sync context.
    Required because Celery tasks are synchronous but our services use async.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coroutine)
