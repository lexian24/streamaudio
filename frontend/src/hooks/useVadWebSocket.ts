import { useState, useCallback, useRef, useEffect } from 'react';

export interface VadStatus {
  status: string;
  state: string;
  is_speech: boolean;
  confidence: number;
  silence_duration?: number;
  recording_duration?: number;
  buffer_size: number;
}

export interface UseVadWebSocketReturn {
  isConnected: boolean;
  isStreaming: boolean;
  error: string | null;
  vadStatus: VadStatus | null;
  connect: () => void;
  disconnect: () => void;
  startStreaming: () => void;
  stopStreaming: () => void;
  sendAudioChunk: (audioData: Float32Array, timestamp: number) => void;
}

export const useVadWebSocket = (url: string): UseVadWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vadStatus, setVadStatus] = useState<VadStatus | null>(null);

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      setError(null);
      console.log('Connecting to VAD WebSocket:', url);
      
      websocketRef.current = new WebSocket(url);

      websocketRef.current.onopen = () => {
        console.log('VAD WebSocket connected');
        setIsConnected(true);
        reconnectAttempts.current = 0;
      };

      websocketRef.current.onclose = (event) => {
        console.log('VAD WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setIsStreaming(false);
        
        // Attempt to reconnect if it wasn't a manual disconnect
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts.current})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };

      websocketRef.current.onerror = (error) => {
        console.error('VAD WebSocket error:', error);
        setError('WebSocket connection error');
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('VAD WebSocket message:', data.type);

          switch (data.type) {
            case 'vad_status':
              setVadStatus(data.result);
              break;
            case 'error':
              console.error('VAD processing error:', data.message);
              setError(data.message);
              break;
            case 'pong':
              // Heartbeat response
              break;
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setError(`Connection failed: ${error}`);
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'Manual disconnect');
      websocketRef.current = null;
    }

    setIsConnected(false);
    setIsStreaming(false);
    reconnectAttempts.current = 0;
  }, []);

  const startStreaming = useCallback(() => {
    if (!isConnected) {
      setError('Not connected to WebSocket');
      return;
    }
    setIsStreaming(true);
    console.log('Started VAD streaming');
  }, [isConnected]);

  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    setVadStatus(null); // Clear status immediately
    console.log('Stopped VAD streaming');
  }, []);

  const sendAudioChunk = useCallback((audioData: Float32Array, timestamp: number) => {
    if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN || !isStreaming) {
      return;
    }

    try {
      // Convert Float32Array to base64 for transmission (TypeScript compatible)
      const buffer = audioData.buffer.slice(audioData.byteOffset, audioData.byteOffset + audioData.byteLength);
      const bytes = new Uint8Array(buffer);
      
      // Convert bytes to string without spread operator for compatibility
      let binaryString = '';
      for (let i = 0; i < bytes.length; i++) {
        binaryString += String.fromCharCode(bytes[i]);
      }
      const base64 = btoa(binaryString);

      const message = {
        type: 'audio_chunk',
        audio_data: base64,
        timestamp: timestamp,
        sample_rate: 16000,
        channels: 1,
        samples: audioData.length
      };

      websocketRef.current.send(JSON.stringify(message));
      
    } catch (error) {
      console.error('Error sending audio chunk:', error);
      setError(`Failed to send audio: ${error}`);
    }
  }, [isStreaming]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [isConnected]);

  return {
    isConnected,
    isStreaming,
    error,
    vadStatus,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    sendAudioChunk,
  };
};