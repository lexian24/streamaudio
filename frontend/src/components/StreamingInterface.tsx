import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useMicrophone } from '../hooks/useMicrophone';
import { useVadWebSocket } from '../hooks/useVadWebSocket';
import { useContinuousRecorder } from '../hooks/useContinuousRecorder';
import TranscriptViewer from './TranscriptViewer';
import '../styles/StreamingInterface.css';
import { AudioAnalysisResult } from '../types/audio';

interface StreamingInterfaceProps {
  onBack: () => void;
}



const StreamingInterface: React.FC<StreamingInterfaceProps> = ({ onBack }) => {
  const [processedResult, setProcessedResult] = useState<AudioAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const API_BASE = 'http://localhost:8000';
  
  // Initialize microphone, WebSocket, and continuous recorder hooks
  const microphone = useMicrophone();
  const vadWebSocket = useVadWebSocket('ws://localhost:8000/ws/vad-stream');
  const continuousRecorder = useContinuousRecorder();
  
  // Recording state management
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'processing'>('idle');
  const [isProcessingRecording, setIsProcessingRecording] = useState(false);


  // Poll for task completion
  const pollTaskStatus = useCallback(async (taskId: string) => {
    const maxAttempts = 120; // 2 minutes max
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const response = await fetch(`${API_BASE}/api/tasks/${taskId}`);
        const taskStatus = await response.json();

        if (taskStatus.status === 'completed') {
          if (taskStatus.result && taskStatus.result.speakers) {
            setProcessedResult(taskStatus.result);
            setLoading(false);
            console.log('✅ Recording analysis complete');
          } else {
            throw new Error('Task completed but no result data');
          }
          return;
        } else if (taskStatus.status === 'failed') {
          throw new Error(taskStatus.error || 'Processing failed');
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to check task status';
        setError(errorMessage);
        setLoading(false);
        console.error('❌ Error polling task:', err);
        return;
      }
    }

    setError('Processing timed out after 2 minutes');
    setLoading(false);
  }, [API_BASE]);

  // Upload recording to backend and poll for results
  const uploadRecording = useCallback(async (blob: Blob) => {
    try {
      setIsProcessingRecording(true);
      setError(null);

      const formData = new FormData();
      const filename = `recording_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
      formData.append('file', blob, filename);

      const response = await fetch(`${API_BASE}/vad/upload-recording`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const uploadResponse = await response.json();
        console.log('✅ Recording uploaded, task ID:', uploadResponse.task_id);

        // Poll for results
        if (uploadResponse.task_id) {
          await pollTaskStatus(uploadResponse.task_id);
        } else {
          throw new Error('No task ID returned from upload');
        }

      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload recording');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error uploading recording';
      setError(errorMessage);
      setIsProcessingRecording(false);
      console.error('❌ Error uploading recording:', error);
    } finally {
      setIsProcessingRecording(false);
    }
  }, [API_BASE, pollTaskStatus]);


  // Helper functions
  const isMonitoring = useCallback(() => {
    return microphone.isRecording && vadWebSocket.isStreaming && vadWebSocket.isConnected;
  }, [microphone.isRecording, vadWebSocket.isStreaming, vadWebSocket.isConnected]);

  const getCombinedError = () => {
    // Show microphone, processing, and recording errors
    return error || microphone.error || continuousRecorder.error;
  };

  // VAD-triggered recording logic
  useEffect(() => {
    if (!isMonitoring()) return;

    const vadStatus = vadWebSocket.vadStatus;
    if (!vadStatus) return;

    if (vadStatus.is_speech && vadStatus.confidence > 0.1) {
      // High-confidence speech detected
      console.log(`🟢 Speech detected (conf: ${vadStatus.confidence.toFixed(3)})`);
      
      if (recordingState === 'idle') {
        // Start recording
        setRecordingState('recording');
        continuousRecorder.startRecording();
        console.log('🎤 Started continuous recording (speech detected)');
      }
      
      // Clear any pending silence timer only for confident speech
      if (silenceTimerRef.current) {
        console.log('🔇 Clearing silence timer due to speech');
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    } else {
      // No speech or low-confidence speech
      if (recordingState === 'recording') {
        // Start silence timer if not already started
        if (!silenceTimerRef.current) {
          console.log(`🔇 Starting 5-second silence timer (conf: ${vadStatus.confidence.toFixed(3)})`);
          silenceTimerRef.current = setTimeout(async () => {
            // 5 seconds of silence - stop recording and upload
            console.log('🔇 5 seconds of silence reached - saving recording');
            setRecordingState('processing');

            const recordingBlob = await continuousRecorder.stopRecording();
            if (recordingBlob) {
              console.log('📁 Uploading recording blob:', recordingBlob.size, 'bytes');
              await uploadRecording(recordingBlob);
              // Results will be shown immediately via setProcessedResult in uploadRecording
            }

            setRecordingState('idle');
            silenceTimerRef.current = null;
          }, 5000); // 5 seconds
        }
      }
    }
  }, [vadWebSocket.vadStatus, recordingState, isMonitoring, continuousRecorder, uploadRecording]);

  // Setup audio data callback
  useEffect(() => {
    microphone.onAudioData(vadWebSocket.sendAudioChunk);
  }, [microphone, vadWebSocket.sendAudioChunk]);


  // Update VAD status from WebSocket (we use the WebSocket status directly)
  // No need to store it separately since we get it from vadWebSocket.vadStatus

  // WebSocket connection is now managed manually in start/stop functions

  // Poll for recordings list only when monitoring
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    const monitoring = isMonitoring();
    
    if (monitoring) {
      // Only poll when actively monitoring
      interval = setInterval(() => {
        }, 2000);
      // Initial fetch when starting
    }

    return () => {
      if (interval) clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [microphone.isRecording, vadWebSocket.isStreaming, vadWebSocket.isConnected]);

  // No initial fetch - recordings only loaded after starting monitoring or manual refresh

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
    };
  }, []);

  const startVadMonitoring = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Connect WebSocket first
      vadWebSocket.connect();
      
      // Start VAD monitoring on backend
      const response = await fetch(`${API_BASE}/vad/start`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start monitoring');
      }

      // Start microphone recording
      await microphone.startRecording();
      
      // Start WebSocket streaming
      vadWebSocket.startStreaming();
      
      setError(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start monitoring';
      setError(errorMessage);
      console.error('Failed to start VAD monitoring:', error);
      // Clean up on error
      vadWebSocket.disconnect();
      microphone.stopRecording();
    } finally {
      setLoading(false);
    }
  };

  const stopVadMonitoring = async () => {
    setLoading(true);
    
    try {
      // Stop continuous recording if active
      if (continuousRecorder.isRecording) {
        const recordingBlob = await continuousRecorder.stopRecording();
        if (recordingBlob) {
          await uploadRecording(recordingBlob);
          // Results will be shown immediately via setProcessedResult in uploadRecording
        }
      }
      
      // Clear silence timer
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      
      // Reset recording state
      setRecordingState('idle');
      
      // Stop VAD monitoring
      vadWebSocket.stopStreaming();
      vadWebSocket.disconnect();
      microphone.stopRecording();
      
      // Stop VAD monitoring on backend
      const response = await fetch(`${API_BASE}/vad/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to stop monitoring');
      }
      
      // Refresh recordings list once after stopping
      
      setError(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to stop monitoring';
      setError(errorMessage);
      console.error('Failed to stop VAD monitoring:', error);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="streaming-interface">
      <header className="streaming-header">
        <button className="back-button" onClick={onBack}>
          ← Back to File Upload
        </button>
        <h1>Live Monitoring</h1>
        <p>Automatic voice activity pickup for analysis</p>
      </header>

      {/* Simple Status */}
      <div className="vad-status">
        <div className={`status-indicator ${isMonitoring() ? 'monitoring' : 'idle'}`}>
          {isMonitoring() ? 'Monitoring' : 'Not Monitoring'}
        </div>
        {continuousRecorder.isRecording && (
          <div className="recording-timer">
            ⏱️ Recording: {Math.floor(continuousRecorder.recordingDuration / 60)}:{Math.floor(continuousRecorder.recordingDuration % 60).toString().padStart(2, '0')}
          </div>
        )}
        {isProcessingRecording && (
          <div className="processing-indicator">
            ⚙️ Processing recording...
          </div>
        )}
      </div>

      {/* Error Display */}
      {getCombinedError() && (
        <div className="error-banner">
          <span className="error-icon">❌</span>
          <span className="error-text">{getCombinedError()}</span>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* VAD Controls */}
      <div className="vad-controls">
        <div className="control-panel">
          {!isMonitoring() ? (
            <button 
              className="start-button"
              onClick={startVadMonitoring}
              disabled={loading}
            >
              {loading ? 'Starting...' : 'Start Monitoring'}
            </button>
          ) : (
            <button
              className="stop-button"
              onClick={stopVadMonitoring}
              disabled={loading && !isProcessingRecording}
            >
              {loading && !isProcessingRecording ? '⏳ Stopping...' : '⏹️ Stop Monitoring'}
            </button>
          )}
          
          {continuousRecorder.isRecording && (
            <div className="recording-status">
              🔴 Recording...
            </div>
          )}

          {recordingState === 'processing' && (
            <div className="recording-status">
              ⚙️ Saving recording...
            </div>
          )}
        </div>

        {/* Simple VAD Indicator - Only Green/Red Circle */}
        {isMonitoring() && (
          <div className="vad-indicator">
            <div className={`vad-light ${vadWebSocket.vadStatus?.is_speech && vadWebSocket.vadStatus?.confidence > 0.1 ? 'speech' : 'silence'}`}>
              {vadWebSocket.vadStatus?.is_speech && vadWebSocket.vadStatus?.confidence > 0.1 ? '🟢' : '🔴'}
            </div>
          </div>
        )}
      </div>




      {/* Analysis Results */}
      {processedResult && (
        <div className="results-section">
          <div className="results-header">
            <h2>📊 Recording Analysis Complete!</h2>
            <button className="reset-button" onClick={() => setProcessedResult(null)}>
              Close Results
            </button>
          </div>
          <TranscriptViewer result={processedResult} />
        </div>
      )}

      {/* No Latest Recording section - VAD recordings show results immediately like upload */}

    </div>
  );
};

export default StreamingInterface;