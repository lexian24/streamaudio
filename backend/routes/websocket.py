"""
WebSocket endpoints for real-time audio streaming.
"""

import asyncio
import base64
import json
import logging
import numpy as np
import time

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
    logger.info(f"Auto recorder status: {auto_recorder is not None}")
    if not auto_recorder:
        logger.error("Auto recorder not initialized!")
        await websocket.close(code=1011, reason="Auto recorder not initialized")
        return
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming data
                audio_data = json.loads(data)
                message_type = audio_data.get('type')
                logger.info(f"📨 Received WebSocket message: type={message_type}")
                
                if message_type == "audio_chunk":
                    # Decode base64 audio data
                    audio_b64 = audio_data.get("audio_data", "")
                    if not audio_b64:
                        logger.warning("Received empty audio data")
                        continue
                        
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert to numpy array (assuming float32 format)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Get timestamp and convert from milliseconds to seconds if needed
                    timestamp = audio_data.get("timestamp", time.time())
                    if timestamp > 10000000000:  # If timestamp is in milliseconds
                        timestamp = timestamp / 1000.0
                    
                    # Calculate audio level for debugging
                    audio_level = float(np.sqrt(np.mean(audio_array ** 2)))
                    logger.info(f"📡 Audio level: {audio_level:.4f}")
                    
                    # Process through VAD
                    result = auto_recorder.process_audio_chunk(audio_array, timestamp)
                    
                    # Send response back to client
                    response = {
                        "type": "vad_status",
                        "result": result,
                        "timestamp": timestamp
                    }
                    
                    # Log the VAD result
                    if result.get("status") == "active":
                        logger.info(f"🎤 VAD Result: speech={result.get('is_speech')}, state={result.get('state')}, confidence={result.get('confidence', 0):.3f}")
                    
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
                
            except json.JSONDecodeError as e:
                logger.warning(f"Received invalid JSON data from WebSocket client: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing WebSocket data: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")