"""
WebSocket endpoints for real-time audio streaming.
"""

import asyncio
import base64
import json
import logging
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .dependencies import get_auto_recorder

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/vad-stream")
async def websocket_vad_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time voice activity detection and recording"""
    await websocket.accept()
    logger.info("WebSocket connection established for VAD streaming")
    
    auto_recorder = get_auto_recorder()
    if not auto_recorder:
        await websocket.close(code=1011, reason="Auto recorder not initialized")
        return
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming data
                audio_data = json.loads(data)
                
                if audio_data.get("type") == "audio":
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(audio_data["data"])
                    
                    # Convert to numpy array (assuming float32 format)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Process through VAD
                    result = auto_recorder.process_audio_stream(audio_array)
                    
                    # Send response back to client
                    response = {
                        "type": "vad_result",
                        "is_speech": result.get("is_speech", False),
                        "confidence": result.get("confidence", 0.0),
                        "recording_status": result.get("recording_status", "idle")
                    }
                    
                    await websocket.send_text(json.dumps(response))
                
                elif audio_data.get("type") == "control":
                    # Handle control messages (start/stop recording, etc.)
                    command = audio_data.get("command")
                    
                    if command == "start_monitoring":
                        auto_recorder.start_monitoring()
                        await websocket.send_text(json.dumps({
                            "type": "control_response",
                            "command": "start_monitoring",
                            "status": "started"
                        }))
                    
                    elif command == "stop_monitoring":
                        auto_recorder.stop_monitoring()
                        await websocket.send_text(json.dumps({
                            "type": "control_response", 
                            "command": "stop_monitoring",
                            "status": "stopped"
                        }))
                    
                    elif command == "get_status":
                        status = auto_recorder.get_status()
                        await websocket.send_text(json.dumps({
                            "type": "status_response",
                            "status": status
                        }))
                
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON data from WebSocket client")
                continue
            except Exception as e:
                logger.error(f"Error processing WebSocket data: {e}")
                continue
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")