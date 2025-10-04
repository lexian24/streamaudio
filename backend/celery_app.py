"""
Celery application configuration for async audio processing tasks.

This module configures Celery with Redis as the message broker and result backend.
Workers process heavy audio tasks (transcription, speaker identification, emotion analysis)
in the background, allowing the FastAPI app to respond immediately to upload requests.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from celery import Celery
from kombu import Exchange, Queue

# Load environment variables from .env file (look in parent directory)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Redis connection URL from environment or default
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery application
celery_app = Celery(
    'fastaudio',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['tasks.audio_tasks']  # Auto-discover tasks
)

# Celery Configuration
celery_app.conf.update(
    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution settings
    task_track_started=True,  # Track when tasks start
    task_time_limit=600,  # Hard limit: 10 minutes per task
    task_soft_time_limit=540,  # Soft limit: 9 minutes (warning before kill)

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store additional metadata

    # Worker settings
    worker_prefetch_multiplier=1,  # Workers take one task at a time (for heavy tasks)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)

    # Retry policy
    task_acks_late=True,  # Only acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies

    # Task routing (optional: for multiple queues)
    task_default_queue='audio_processing',
    task_queues=(
        Queue('audio_processing', Exchange('audio_processing'), routing_key='audio.*'),
        Queue('priority', Exchange('priority'), routing_key='priority.*'),
    ),
    task_default_exchange='audio_processing',
    task_default_routing_key='audio.process',
)

# Task routes (assign tasks to specific queues)
celery_app.conf.task_routes = {
    'tasks.audio_tasks.process_audio_file': {'queue': 'audio_processing'},
    'tasks.audio_tasks.process_vad_recording': {'queue': 'audio_processing'},
}


# Periodic tasks configuration (optional - requires celery beat)
# celery_app.conf.beat_schedule = {
#     'cleanup-old-tasks': {
#         'task': 'tasks.maintenance.cleanup_old_tasks',
#         'schedule': 3600.0,  # Every hour
#     },
# }
