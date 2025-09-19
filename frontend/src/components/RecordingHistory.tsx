import React, { useState, useEffect } from 'react';
import './RecordingHistory.css';

interface Recording {
  id: number;
  filename: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  duration: number;
  sample_rate: number;
  channels: number;
  format: string;
  created_at: string;
  updated_at: string | null;
  processing_results: ProcessingResult[];
}

interface ProcessingResult {
  id: number;
  transcription: string | null;
  confidence_score: number | null;
  diarization_json: any;
  num_speakers: number | null;
  emotions_json: any;
  dominant_emotion: string | null;
  emotion_confidence: number | null;
  processing_duration: number | null;
  model_versions: any;
  status: string;
  processed_at: string;
  speaker_segments: SpeakerSegment[];
}

interface SpeakerSegment {
  id: number;
  start_time: number;
  end_time: number;
  duration: number;
  speaker_label: string;
  segment_text: string | null;
  confidence: number | null;
  speaker_name: string | null;
}

interface RecordingHistoryProps {
  onRecordingSelect?: (recording: Recording) => void;
}

const RecordingHistory: React.FC<RecordingHistoryProps> = ({ onRecordingSelect }) => {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedRecording, setExpandedRecording] = useState<number | null>(null);

  useEffect(() => {
    fetchRecordings();
  }, []);

  const fetchRecordings = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/recordings');
      if (!response.ok) {
        throw new Error('Failed to fetch recordings');
      }
      const data = await response.json();
      setRecordings(data.recordings);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recordings');
    } finally {
      setLoading(false);
    }
  };

  const fetchRecordingDetails = async (recordingId: number) => {
    try {
      const response = await fetch(`/api/recordings/${recordingId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch recording details');
      }
      const recording = await response.json();
      
      // Update the recording in our list with full details
      setRecordings(prev => 
        prev.map(r => r.id === recordingId ? recording : r)
      );
      
      if (onRecordingSelect) {
        onRecordingSelect(recording);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recording details');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDateTime = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  const toggleExpanded = async (recordingId: number) => {
    if (expandedRecording === recordingId) {
      setExpandedRecording(null);
    } else {
      setExpandedRecording(recordingId);
      await fetchRecordingDetails(recordingId);
    }
  };

  if (loading) {
    return (
      <div className="recording-history">
        <h3>Recording History</h3>
        <div className="loading">Loading recordings...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="recording-history">
        <h3>Recording History</h3>
        <div className="error">
          {error}
          <button onClick={fetchRecordings} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="recording-history">
      <div className="history-header">
        <h3>Recording History</h3>
        <button onClick={fetchRecordings} className="refresh-button">
          🔄 Refresh
        </button>
      </div>
      
      {recordings.length === 0 ? (
        <div className="no-recordings">
          No recordings yet. Start recording to see your history here!
        </div>
      ) : (
        <div className="recordings-list">
          {recordings.map((recording) => (
            <div 
              key={recording.id} 
              className={`recording-item ${expandedRecording === recording.id ? 'expanded' : ''}`}
            >
              <div 
                className="recording-summary"
                onClick={() => toggleExpanded(recording.id)}
              >
                <div className="recording-info">
                  <div className="recording-name">
                    {recording.original_filename || recording.filename}
                  </div>
                  <div className="recording-meta">
                    {formatDuration(recording.duration)} • {formatFileSize(recording.file_size)} • {formatDateTime(recording.created_at)}
                  </div>
                </div>
                <div className="recording-stats">
                  <span className="format-badge">{recording.format.toUpperCase()}</span>
                  {recording.processing_results.length > 0 && (
                    <span className="processed-badge">
                      {recording.processing_results.length} analysis
                    </span>
                  )}
                  <span className="expand-icon">
                    {expandedRecording === recording.id ? '▼' : '▶'}
                  </span>
                </div>
              </div>

              {expandedRecording === recording.id && (
                <div className="recording-details">
                  <div className="technical-details">
                    <h4>Technical Details</h4>
                    <div className="details-grid">
                      <div><strong>Sample Rate:</strong> {recording.sample_rate} Hz</div>
                      <div><strong>Channels:</strong> {recording.channels}</div>
                      <div><strong>File Path:</strong> {recording.file_path}</div>
                      <div><strong>ID:</strong> {recording.id}</div>
                    </div>
                  </div>

                  {recording.processing_results.length > 0 && (
                    <div className="processing-results">
                      <h4>Processing Results ({recording.processing_results.length})</h4>
                      {recording.processing_results.map((result) => (
                        <div key={result.id} className="processing-result">
                          <div className="result-header">
                            <span className={`status-badge ${result.status}`}>
                              {result.status}
                            </span>
                            <span className="processed-time">
                              {formatDateTime(result.processed_at)}
                            </span>
                          </div>
                          
                          {result.transcription && (
                            <div className="transcription">
                              <strong>Transcription:</strong>
                              <p>{result.transcription}</p>
                              {result.confidence_score && (
                                <small>Confidence: {(result.confidence_score * 100).toFixed(1)}%</small>
                              )}
                            </div>
                          )}

                          {result.num_speakers && (
                            <div className="diarization">
                              <strong>Speakers Detected:</strong> {result.num_speakers}
                            </div>
                          )}

                          {result.dominant_emotion && (
                            <div className="emotion">
                              <strong>Dominant Emotion:</strong> {result.dominant_emotion}
                              {result.emotion_confidence && (
                                <span> ({(result.emotion_confidence * 100).toFixed(1)}%)</span>
                              )}
                            </div>
                          )}

                          {result.speaker_segments.length > 0 && (
                            <div className="speaker-segments">
                              <strong>Speaker Segments:</strong>
                              <div className="segments-list">
                                {result.speaker_segments.map((segment) => (
                                  <div key={segment.id} className="segment">
                                    <span className="speaker">{segment.speaker_name || segment.speaker_label}</span>
                                    <span className="time">
                                      {formatDuration(segment.start_time)} - {formatDuration(segment.end_time)}
                                    </span>
                                    {segment.segment_text && (
                                      <span className="text">"{segment.segment_text}"</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RecordingHistory;