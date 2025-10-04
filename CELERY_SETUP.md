# FastAudio Celery + Redis Setup Guide

This guide explains how to set up and test the asynchronous processing system using Celery and Redis.

## 🎯 Overview

The system now uses **Celery** workers with **Redis** as a message broker to process audio files asynchronously. This means:
- **Upload endpoint** returns immediately with a `task_id`
- **Processing happens in background** on Celery workers
- **Frontend polls** `/api/tasks/{task_id}` for status and results
- **Scalable** - add more workers for parallel processing

## 📋 Prerequisites

- Python 3.9+
- Redis server (via Docker or local installation)
- FFmpeg (for audio conversion)
- All existing dependencies

## 🚀 Quick Start

### 1. Install New Dependencies

```bash
cd streamaudio/backend
pip install -r requirements.txt
```

New packages:
- `celery==5.3.4`
- `redis==5.0.1`

### 2. Start Redis

**Option A: Using Docker (Recommended)**
```bash
cd streamaudio
docker-compose up -d redis
```

**Option B: Local Redis**
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Verify Redis is running
redis-cli ping  # Should return: PONG
```

### 3. Run Database Migration

```bash
cd streamaudio/backend
python -c "
from database.config import SessionLocal
from database.models import Base
from sqlalchemy import create_engine

# Run migration
engine = create_engine('sqlite:///./fastaudio.db')
Base.metadata.create_all(engine)
print('✅ Database migration completed')
"
```

Or manually run the SQL migration:
```bash
sqlite3 fastaudio.db < database/migrations/add_processing_tasks_table.sql
```

### 4. Start the System

You need **3 terminal windows**:

**Terminal 1: FastAPI Server**
```bash
cd streamaudio/backend
python main.py
```

**Terminal 2: Celery Worker**

**macOS (Recommended - avoids fork issues with PyTorch):**
```bash
cd streamaudio/backend
celery -A celery_app worker --loglevel=info --pool=solo
```

**Linux/Alternative macOS:**
```bash
cd streamaudio/backend
# Option 1: Disable fork safety check
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
celery -A celery_app worker --loglevel=info --concurrency=2

# Option 2: Use threads instead of processes
celery -A celery_app worker --loglevel=info --pool=threads --concurrency=2
```

**Terminal 3: Frontend (optional)**
```bash
cd streamaudio/frontend
npm start
```

## 📊 System Architecture

```
┌─────────────┐         ┌──────────────┐         ┌────────────┐
│   Frontend  │────────▶│  FastAPI     │────────▶│   Redis    │
│             │         │   Server     │         │  (Broker)  │
└─────────────┘         └──────────────┘         └────────────┘
      │                        │                         │
      │ Polls for results      │                         │
      │                        │                         ▼
      │                        │                  ┌────────────┐
      │                        │                  │   Celery   │
      └────────────────────────┼─────────────────│   Worker   │
                               │                  └────────────┘
                               ▼                         │
                        ┌──────────────┐                │
                        │   Database   │◀───────────────┘
                        │  (SQLite)    │
                        └──────────────┘
```

## 🔄 API Workflow

### Old Synchronous Flow:
```
POST /analyze
  ↓ (wait 10-30s)
Returns: Full analysis results
```

### New Async Flow:
```
POST /analyze
  ↓ (immediate)
Returns: {task_id, status: "queued", poll_url}

GET /api/tasks/{task_id}
  ↓
Returns: {status: "processing", progress: 45}

GET /api/tasks/{task_id}
  ↓
Returns: {status: "completed", result: {...}}
```

## 🧪 Testing

### 1. Test Redis Connection
```bash
redis-cli ping
# Expected output: PONG
```

### 2. Test Celery Worker
```bash
# Terminal 1: Start worker
celery -A celery_app worker --loglevel=info

# You should see:
# [tasks]
#   . tasks.audio_tasks.process_audio_file
#   . tasks.audio_tasks.process_vad_recording
```

### 3. Test API Upload

```bash
# Upload a test audio file
curl -X POST "http://localhost:8000/analyze" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@test.wav"

# Response (immediate):
{
  "status": "queued",
  "task_id": "abc123-def456-...",
  "recording_id": 42,
  "filename": "test.wav",
  "message": "Audio file uploaded successfully. Processing in background.",
  "poll_url": "/api/tasks/abc123-def456-..."
}
```

### 4. Check Task Status

```bash
# Poll for task status
curl "http://localhost:8000/api/tasks/abc123-def456-..."

# While processing:
{
  "task_id": "abc123-def456-...",
  "status": "processing",
  "progress": 50,
  "created_at": "2025-01-03T10:00:00",
  "started_at": "2025-01-03T10:00:01"
}

# When completed:
{
  "task_id": "abc123-def456-...",
  "status": "completed",
  "progress": 100,
  "result": {
    "segments": [...],
    "speakers": [...],
    "transcription": "...",
    ...
  },
  "created_at": "2025-01-03T10:00:00",
  "started_at": "2025-01-03T10:00:01",
  "completed_at": "2025-01-03T10:00:25"
}
```

### 5. Monitor with Redis CLI

```bash
# Check queued tasks
redis-cli LLEN celery

# Monitor real-time
redis-cli MONITOR
```

## 🛠️ Configuration

### Environment Variables

Create `.env` file in `backend/`:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Task Configuration
TASK_RESULT_EXPIRES=3600  # 1 hour

# Existing settings
WHISPER_MODEL_SIZE=base
ENABLE_GPU=false
```

### Celery Worker Options

```bash
# Basic worker
celery -A celery_app worker --loglevel=info

# Multiple workers (for parallel processing)
celery -A celery_app worker --loglevel=info --concurrency=4

# Background worker (daemon)
celery -A celery_app worker --loglevel=info --detach

# With auto-reload (development)
celery -A celery_app worker --loglevel=info --autoreload
```

## 📈 Monitoring

### Flower (Celery Monitoring Tool)

Install:
```bash
pip install flower
```

Run:
```bash
celery -A celery_app flower
```

Access: http://localhost:5555

## 🔧 Troubleshooting

### Issue: Redis connection refused
```bash
# Check if Redis is running
redis-cli ping

# If not, start it:
docker-compose up -d redis
# or
brew services start redis
```

### Issue: Celery worker not picking up tasks
```bash
# Check worker is running
celery -A celery_app inspect active

# Restart worker
# Ctrl+C to stop, then restart:
celery -A celery_app worker --loglevel=info
```

### Issue: Task stuck in "queued" status
```bash
# Check Celery worker logs
# Worker terminal should show task execution

# Purge queue (WARNING: clears all pending tasks)
celery -A celery_app purge
```

### Issue: Worker crashes with "SIGABRT" or fork errors (macOS)
```
Error: objc[xxxxx]: +[MPSGraphObject initialize] may have been in progress...
Worker exited prematurely: signal 6 (SIGABRT)
```

**Cause:** macOS doesn't handle fork() well with PyTorch/ML libraries

**Solution:**
```bash
# Use solo pool (single process, no forking)
celery -A celery_app worker --loglevel=info --pool=solo

# OR disable fork safety (if you need multiple workers)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
celery -A celery_app worker --loglevel=info --concurrency=2

# OR use threads instead of processes
celery -A celery_app worker --loglevel=info --pool=threads --concurrency=2
```

### Issue: Import errors in tasks
```bash
# Make sure you're in the backend directory
cd streamaudio/backend

# Check Python path
python -c "import sys; print(sys.path)"

# Task file should be importable
python -c "from tasks.audio_tasks import process_audio_file; print('OK')"
```

## 📝 Development Notes

### Adding New Tasks

1. Create task in `tasks/audio_tasks.py`:
```python
@celery_app.task(bind=True, name='tasks.my_new_task')
def my_new_task(self, param1, param2):
    # Task logic
    return result
```

2. Import and use in routes:
```python
from tasks.audio_tasks import my_new_task

task = my_new_task.delay(arg1, arg2)
```

### Task Retry Logic

Tasks automatically retry on failure (up to 3 times):
```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def my_task(self):
    try:
        # Task logic
        pass
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

## 🎓 Next Steps

1. **Frontend Integration**: Update frontend to poll `/api/tasks/{task_id}`
2. **Production Deployment**: Use Redis cluster, multiple workers
3. **Monitoring**: Set up Flower for production monitoring
4. **Task Cleanup**: Schedule periodic cleanup of old completed tasks

## 📚 Additional Resources

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Documentation](https://redis.io/documentation)
- [Flower Documentation](https://flower.readthedocs.io/)

---

**Status**: ✅ Backend implementation complete, frontend update pending
