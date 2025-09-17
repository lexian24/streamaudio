"""
Simple test server to check WebSocket connectivity
"""
import uvicorn
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="StreamAudio Test API", description="Test WebSocket connectivity")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: List[WebSocket] = []

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "service": "StreamAudio-Test",
        "timestamp": time.time()
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Test WebSocket endpoint"""
    await websocket.accept()
    active_connections.append(websocket)
    session_id = id(websocket)
    
    logger.info(f"WebSocket connection established: {session_id}")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "WebSocket connected successfully",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            message_type = data.get("type")
            logger.info(f"Received message: {message_type}")
            
            if message_type == "start_stream":
                await websocket.send_text(json.dumps({
                    "type": "stream_started",
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "message": "Streaming started (test mode)"
                }))
                
            elif message_type == "audio_chunk":
                await websocket.send_text(json.dumps({
                    "type": "chunk_processed",
                    "timestamp": time.time(),
                    "segments": [{
                        "start_time": 0,
                        "end_time": 3,
                        "speaker_id": "Speaker 1",
                        "text": "This is a test message",
                        "emotion": "neutral",
                        "emotion_confidence": 0.8,
                        "duration": 3.0
                    }],
                    "message": "Test audio chunk processed"
                }))
                
            elif message_type == "stop_stream":
                await websocket.send_text(json.dumps({
                    "type": "stream_stopped",
                    "timestamp": time.time(),
                    "session_summary": {
                        "total_segments": 1,
                        "total_speakers": 1,
                        "speakers": [{
                            "speaker_id": "Speaker 1",
                            "total_duration": 3.0,
                            "segment_count": 1,
                            "latest_emotion": "neutral",
                            "latest_text": "This is a test message"
                        }]
                    }
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e),
            "timestamp": time.time()
        }))
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed: {session_id}")

@app.get("/streaming/status")
async def get_streaming_status():
    """Get current streaming status"""
    return {
        "active_connections": len(active_connections),
        "test_mode": True
    }

if __name__ == "__main__":
    uvicorn.run("test_main:app", host="0.0.0.0", port=8000, reload=False, ws_max_size=16777216)