"""
FastAudio Configuration
Simple configuration for the FastAudio application
"""
import os
from pathlib import Path


class Settings:
    def __init__(self):
        # Model configuration
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny, base, small, medium, large
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", None)

        # Audio processing settings
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

        # Server settings
        self.upload_dir = "./uploads"
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", 8000))

        # Processing settings
        self.enable_gpu = os.getenv("ENABLE_GPU", "true").lower() == "true"
        self.batch_size = int(os.getenv("BATCH_SIZE", 1))

        # Redis and Celery settings
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.celery_broker_url = os.getenv("CELERY_BROKER_URL", self.redis_url)
        self.celery_result_backend = os.getenv("CELERY_RESULT_BACKEND", self.redis_url)
        self.task_result_expires = int(os.getenv("TASK_RESULT_EXPIRES", 3600))  # 1 hour

        # Ensure directories exist
        Path(self.upload_dir).mkdir(exist_ok=True)


# Global settings instance
settings = Settings()