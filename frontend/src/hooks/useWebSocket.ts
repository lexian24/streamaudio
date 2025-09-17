import { useState, useCallback, useRef, useEffect } from 'react';

export interface TranscriptionResult {
  start_time: number;
  end_time: number;
  text: string;
  speaker_id: string;
  status: string;
  chunk_id: string;
  processing_time: number;
  emotion?: string;
  emotion_confidence?: number;
}

export interface SpeakerSegment {
  start_time: number;
  end_time: number;
  speaker_id: string;
  emotion: string;
  emotion_confidence: number;
  duration: number;
}

export interface StreamingResult {
  type: string;
  timestamp: number;
  // Legacy format (for backwards compatibility)
  segments?: Array<{
    start_time: number;
    end_time: number;
    speaker_id: string;
    text: string;
    emotion: string;
    emotion_confidence: number;
    duration: number;
  }>;
  // New dual processing formats
  result?: TranscriptionResult;
  speaker_segments?: SpeakerSegment[];
  updated_transcriptions?: TranscriptionResult[];
  results?: StreamingResult[];
  active_speakers?: Record<string, any>;
  session_summary?: {
    total_segments: number;
    total_speakers: number;
    speakers: Array<{
      speaker_id: string;
      total_duration: number;
      segment_count: number;
      latest_emotion: string;
      latest_text: string;
    }>;
  };
  error?: string;
  processing_time?: number;
}

export interface WebSocketState {
  isConnected: boolean;
  isStreaming: boolean;
  error: string | null;
  results: StreamingResult[];
}

export interface UseWebSocketReturn extends WebSocketState {
  connect: () => void;
  disconnect: () => void;
  startStreaming: () => void;
  stopStreaming: () => void;
  sendAudioChunk: (audioData: Float32Array, timestamp: number) => void;
  clearResults: () => void;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isStreaming: false,
    error: null,
    results: [],
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const maxReconnectAttempts = 5;
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setState(prev => ({ ...prev, error: null }));

    try {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setState(prev => ({ ...prev, isConnected: true, error: null }));
        reconnectAttempts.current = 0;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const result: StreamingResult = JSON.parse(event.data);
          
          // Handle multi_result format by flattening the results
          if (result.type === 'multi_result' && result.results && Array.isArray(result.results)) {
            setState(prev => ({
              ...prev,
              results: [...prev.results, ...(result.results || [])],
            }));
          } else {
            setState(prev => ({
              ...prev,
              results: [...prev.results, result],
            }));
          }

          // Handle specific message types
          if (result.type === 'stream_started') {
            setState(prev => ({ ...prev, isStreaming: true }));
          } else if (result.type === 'stream_stopped') {
            setState(prev => ({ ...prev, isStreaming: false }));
          } else if (result.type === 'error') {
            setState(prev => ({ ...prev, error: result.error || 'Unknown error' }));
          }

        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          setState(prev => ({ ...prev, error: 'Failed to parse server response' }));
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setState(prev => ({ ...prev, error: 'Connection error' }));
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setState(prev => ({ 
          ...prev, 
          isConnected: false, 
          isStreaming: false 
        }));

        // Attempt reconnection if it wasn't a clean close and we haven't exceeded max attempts
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          console.log(`Attempting reconnection in ${timeout}ms...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++;
            connect();
          }, timeout);
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({ ...prev, error: 'Failed to connect' }));
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    
    setState(prev => ({ 
      ...prev, 
      isConnected: false, 
      isStreaming: false 
    }));
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
      setState(prev => ({ ...prev, error: 'Not connected to server' }));
    }
  }, []);

  const startStreaming = useCallback(() => {
    sendMessage({ type: 'start_stream' });
  }, [sendMessage]);

  const stopStreaming = useCallback(() => {
    sendMessage({ type: 'stop_stream' });
  }, [sendMessage]);

  const sendAudioChunk = useCallback((audioData: Float32Array, timestamp: number) => {
    if (!state.isConnected || !state.isStreaming) {
      console.log('Not sending audio - not connected or streaming, connected:', state.isConnected, 'streaming:', state.isStreaming);
      return;
    }

    // Check WebSocket readyState before sending
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.log('WebSocket not ready, state:', wsRef.current?.readyState);
      return;
    }

    try {
      console.log('Sending audio chunk:', audioData.length, 'samples');
      
      // Convert Float32Array to base64 more efficiently
      const buffer = audioData.buffer;
      const bytes = new Uint8Array(buffer);
      
      // Use built-in btoa with proper chunking for large data
      const base64Audio = btoa(String.fromCharCode.apply(null, Array.from(bytes)));

      const message = {
        type: 'audio_chunk',
        audio_data: base64Audio,
        timestamp: timestamp,
        samples: audioData.length
      };

      wsRef.current.send(JSON.stringify(message));
      console.log('Audio chunk sent, base64 length:', base64Audio.length);
      
    } catch (error) {
      console.error('Error sending audio chunk:', error);
      // If sending fails, the connection might be broken
      setState(prev => ({ 
        ...prev, 
        error: `Failed to send audio data: ${error}`,
        isConnected: false 
      }));
    }
  }, [state.isConnected, state.isStreaming]);

  const clearResults = useCallback(() => {
    setState(prev => ({ ...prev, results: [] }));
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    sendAudioChunk,
    clearResults,
  };
};