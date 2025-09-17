# StreamAudio - Voice-Activated Recording & Analysis

An intelligent audio recording system that automatically detects speech and provides comprehensive audio analysis through AI models.

## Features

### Core Functionality
- **Real-time Voice Activity Detection** using Silero VAD
- **Automatic Recording** with speech-triggered start/stop
- **Continuous Audio Capture** via MediaRecorder API
- **Manual Audio Analysis** with AI models

### AI-Powered Analysis
- **Speech Transcription** using OpenAI Whisper
- **Speaker Diarization** with pyannote.audio  
- **Emotion Recognition** from audio signals
- **High-Quality Audio Processing** (WebM to WAV conversion)

### Technical Highlights
- **WebSocket Streaming** for real-time VAD communication
- **React TypeScript** frontend with modern hooks architecture
- **FastAPI Backend** with async processing
- **Separated Architecture** - VAD detection independent from recording

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- FFmpeg (for audio conversion)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo>
cd streamaudio
```

2. **Make scripts executable**
```bash
chmod +x *.sh
```

3. **Start the application**
```bash
./start_app.sh
```

This will:
- Set up Python virtual environment
- Install backend dependencies
- Install frontend dependencies  
- Start both backend and frontend servers

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Set up environment variables
cp ../.env.example ../.env
# Edit .env file with your HuggingFace token

python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

### Environment Setup

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Get HuggingFace token:**
   - Visit [HuggingFace Tokens](https://huggingface.co/settings/tokens)
   - Create a new token (read access is sufficient)
   - Add it to your `.env` file:
     ```
     HUGGINGFACE_TOKEN=hf_your_actual_token_here
     ```

## How It Works

### Workflow
1. **Start Monitoring** - Click "Start VAD Monitoring" 
2. **Speech Detection** - Silero VAD detects when you speak (green indicator)
3. **Automatic Recording** - MediaRecorder starts continuous recording
4. **Smart Stop** - Recording stops after 5 seconds of silence
5. **Manual Analysis** - Click "Analyze Recording" for AI processing

### Recording Process
- **Voice Activity Detection** runs in real-time via WebSocket
- **High-Quality Recording** captures audio separately using MediaRecorder
- **Automatic Conversion** from WebM to WAV for compatibility
- **No Chunking Artifacts** - smooth, continuous audio files

## API Endpoints

### VAD & Recording
- `POST /vad/start` - Start VAD monitoring
- `POST /vad/stop` - Stop VAD monitoring  
- `GET /vad/recordings` - List all recordings
- `POST /vad/upload-recording` - Store recording (auto-called)
- `POST /vad/process-recording/{filename}` - Analyze recording
- `DELETE /vad/recordings/{filename}` - Delete recording

### Legacy Analysis
- `POST /analyze` - Direct file upload and analysis
- `GET /health` - Health check

## Configuration

Environment variables:

```bash
# HuggingFace token for pyannote models
export HUGGINGFACE_TOKEN=your_token_here

# Server settings
export HOST=0.0.0.0
export PORT=8000

# VAD sensitivity (0.1-1.0, lower = more sensitive)
export VAD_THRESHOLD=0.3
```

## Supported Audio Formats

- **Input**: WebM (from browser), WAV, MP3, M4A, FLAC, OGG
- **Output**: WAV (16kHz, mono, 16-bit PCM)
- **Maximum file size**: 50MB

## Architecture

```
StreamAudio/
├── backend/
│   ├── main.py                     # FastAPI + WebSocket server
│   ├── config.py                   # Configuration
│   ├── requirements.txt            # Dependencies
│   └── services/
│       ├── audio_processor.py      # Post-processing pipeline
│       ├── auto_recorder.py        # VAD service
│       ├── voice_activity_detection.py # Silero VAD
│       ├── whisper_transcription.py # Whisper service
│       ├── diarization.py          # Speaker separation
│       └── emotion_recognition.py  # Emotion analysis
└── frontend/
    ├── src/
    │   ├── App.tsx                 # Main React app
    │   ├── components/
    │   │   ├── StreamingInterface.tsx # VAD recording UI
    │   │   ├── AudioUploader.tsx   # Manual upload UI
    │   │   └── TranscriptViewer.tsx # Results display
    │   ├── hooks/
    │   │   ├── useMicrophone.ts    # Audio capture
    │   │   ├── useVadWebSocket.ts  # VAD streaming
    │   │   └── useContinuousRecorder.ts # MediaRecorder
    │   └── types/                  # TypeScript definitions
    └── package.json
```

## System Flow

```
Browser Audio Input
       ↓
  MediaRecorder (High Quality Recording)
       ↓
  WebSocket Stream (Lightweight VAD)
       ↓
  Silero VAD Detection
       ↓
  Speech Trigger → Start/Stop Recording
       ↓
  WAV File Storage (FFmpeg Conversion)
       ↓
  Manual Analysis Trigger
       ↓
  AI Pipeline (Whisper → Diarization → Emotion)
       ↓
  Results Display
```

## AI Models Used

- **Voice Activity Detection**: Silero VAD (real-time, 16kHz)
- **Speech Transcription**: OpenAI Whisper (base model)
- **Speaker Diarization**: pyannote/speaker-diarization-3.1
- **Emotion Recognition**: 
  - Audio: wav2vec2-speech-emotion-recognition
  - Text: distilroberta-emotion-classification

## Development

Start in development mode:
```bash
# Backend with auto-reload
cd backend && uvicorn main:app --reload

# Frontend with hot reload  
cd frontend && npm start
```

## Troubleshooting

**Common issues:**

1. **pyannote model access**: You may need a HuggingFace token
2. **GPU support**: Install PyTorch with CUDA for GPU acceleration
3. **Memory issues**: Use smaller Whisper models (tiny/base)
4. **Port conflicts**: Change PORT environment variable

## License

MIT License - see LICENSE file for details.