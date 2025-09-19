import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useMicrophone } from '../hooks/useMicrophone';
import { useVadWebSocket } from '../hooks/useVadWebSocket';
import { useContinuousRecorder } from '../hooks/useContinuousRecorder';
import TranscriptViewer from './TranscriptViewer';
import RecordingHistory from './RecordingHistory';
import '../styles/StreamingInterface.css';
import { AudioAnalysisResult } from '../types/audio';

interface StreamingInterfaceProps {
  onBack: () => void;
}


interface Recording {
  filename: string;
  path: string;
  timestamp: string;
  duration: number;
  size: number;
}

const StreamingInterface: React.FC<StreamingInterfaceProps> = ({ onBack }) => {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [processingFile, setProcessingFile] = useState<string | null>(null);
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
  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'stopping'>('idle');

  // Fetch recordings list
  const fetchRecordings = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/vad/recordings`);
      if (response.ok) {
        const data = await response.json();
        setRecordings(data.recordings);
      }
    } catch (error) {
      console.error('Failed to fetch recordings:', error);
    }
  }, []);

  // Upload recording to backend
  const uploadRecording = useCallback(async (blob: Blob) => {
    try {
      const formData = new FormData();
      const filename = `recording_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
      formData.append('file', blob, filename);

      const response = await fetch(`${API_BASE}/vad/upload-recording`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('✅ Recording uploaded successfully');
        // Refresh recordings list
        fetchRecordings();
      } else {
        console.error('❌ Failed to upload recording');
      }
    } catch (error) {
      console.error('❌ Error uploading recording:', error);
    }
  }, [fetchRecordings]);

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

    if (vadStatus.is_speech && vadStatus.confidence > 0.4) {
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
            // 5 seconds of silence - stop recording
            console.log('🔇 5 seconds of silence reached - stopping recording');
            setRecordingState('stopping');
            
            const recordingBlob = await continuousRecorder.stopRecording();
            if (recordingBlob) {
              console.log('📁 Uploading recording blob:', recordingBlob.size, 'bytes');
              await uploadRecording(recordingBlob);
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
        fetchRecordings();
      }, 2000);
      // Initial fetch when starting
      fetchRecordings();
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
      setTimeout(() => fetchRecordings(), 500);
      
      setError(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to stop monitoring';
      setError(errorMessage);
      console.error('Failed to stop VAD monitoring:', error);
    } finally {
      setLoading(false);
    }
  };

  const processRecording = async (filename: string) => {
    setProcessingFile(filename);
    setProcessedResult(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/vad/process-recording/${filename}`, {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        setProcessedResult(result);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to process recording');
      }
    } catch (error) {
      setError('Network error processing recording');
    } finally {
      setProcessingFile(null);
    }
  };

  const deleteRecording = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE}/vad/recordings/${filename}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchRecordings();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to delete recording');
      }
    } catch (error) {
      setError('Network error deleting recording');
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
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
            ⏱️ Recording: {formatDuration(continuousRecorder.recordingDuration)}
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
              disabled={loading}
            >
              {loading ? '⏳ Stopping...' : '⏹️ Stop Monitoring'}
            </button>
          )}
          
          {continuousRecorder.isRecording && (
            <div className="recording-status">
              🔴 Recording...
            </div>
          )}
          
          {recordingState === 'stopping' && (
            <div className="recording-status">
              ⏹️ Finishing recording...
            </div>
          )}
        </div>

        {/* Simple VAD Indicator - Only Green/Red Circle */}
        {isMonitoring() && (
          <div className="vad-indicator">
            <div className={`vad-light ${vadWebSocket.vadStatus?.is_speech && vadWebSocket.vadStatus?.confidence > 0.4 ? 'speech' : 'silence'}`}>
              {vadWebSocket.vadStatus?.is_speech && vadWebSocket.vadStatus?.confidence > 0.4 ? '🟢' : '🔴'}
            </div>
          </div>
        )}
      </div>


      {/* Recordings List */}
      <div className="recordings-section">
        <h2>📁 Recordings ({recordings.length})</h2>
        
        {recordings.length === 0 ? (
          <div className="no-recordings">
            <p>No recordings yet.</p>
            <p>Start monitoring to automatically record when speech is detected.</p>
          </div>
        ) : (
          <div className="recordings-list">
            {recordings.map((recording) => (
              <div key={recording.filename} className="recording-item">
                <div className="recording-info">
                  <div className="recording-name">{recording.filename}</div>
                  <div className="recording-meta">
                    <span>📅 {new Date(recording.timestamp).toLocaleString()}</span>
                    <span>⏱️ {formatDuration(recording.duration)}</span>
                    <span>💾 {formatFileSize(recording.size)}</span>
                  </div>
                </div>
                <div className="recording-actions">
                  <button 
                    className="analyze-button"
                    onClick={() => processRecording(recording.filename)}
                    disabled={processingFile !== null}
                    title="Analyze with transcription, speaker diarization, and emotion recognition"
                  >
                    {processingFile === recording.filename ? '🔄 Analyzing...' : '🔍 Analyze Recording'}
                  </button>
                  <button 
                    className="delete-button"
                    onClick={() => deleteRecording(recording.filename)}
                    title="Delete this recording permanently"
                  >
                    🗑️ Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Processing Status */}
      {processingFile && (
        <div className="processing-section">
          <div className="processing-container">
            <div className="spinner"></div>
            <h3>⚡ Processing {processingFile}</h3>
            <p>Running speaker diarization, transcription, and emotion analysis...</p>
          </div>
        </div>
      )}

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

      {/* Recording History Section */}
      <RecordingHistory 
        onRecordingSelect={(recording) => {
          console.log('Selected recording:', recording);
          // You can add logic here to display selected recording details
        }}
      />

    </div>
  );
};

export default StreamingInterface;