import React, { useState, useEffect } from 'react';
import './RecordingHistory.css';
import TranscriptViewer from './TranscriptViewer';
import { AudioAnalysisResult } from '../types/audio';

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
  const [selectedRecordings, setSelectedRecordings] = useState<Set<number>>(new Set());
  const [deleting, setDeleting] = useState<Set<number>>(new Set());
  const [modalRecording, setModalRecording] = useState<{recording: Recording, result: ProcessingResult} | null>(null);

  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    fetchRecordings();
  }, []);

  const fetchRecordings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/recordings`);
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
      const response = await fetch(`${API_BASE}/api/recordings/${recordingId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch recording details');
      }
      const recordingDetails = await response.json();
      console.log('Raw API response:', recordingDetails);
      
      // Debug processing results structure
      if (recordingDetails.processing_results) {
        recordingDetails.processing_results.forEach((result: any, idx: number) => {
          console.log(`Processing Result ${idx}:`, {
            status: result.status,
            has_speaker_segments: !!result.speaker_segments,
            speaker_segments_length: result.speaker_segments ? result.speaker_segments.length : 0,
            speaker_segments_sample: result.speaker_segments ? result.speaker_segments.slice(0, 2) : null
          });
        });
      }
      
      // The API returns {recording: {...}, processing_results: [...], speakers: [...]}
      // We need to flatten this structure for the frontend
      const flattenedRecording = {
        ...recordingDetails.recording,
        processing_results: recordingDetails.processing_results || [],
        speakers: recordingDetails.speakers || []
      };
      
      console.log('Flattened recording:', flattenedRecording);
      console.log('Processing results:', flattenedRecording.processing_results);
      
      // Update the recording in our list with full details
      setRecordings(prev => 
        prev.map(r => r.id === recordingId ? flattenedRecording : r)
      );
      
      if (onRecordingSelect) {
        onRecordingSelect(flattenedRecording);
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

  const convertToAnalysisResult = (recording: Recording, result: ProcessingResult): AudioAnalysisResult => {
    // Convert speaker segments to the format expected by TranscriptViewer
    const segments = result.speaker_segments.map(segment => ({
      start_time: segment.start_time,
      end_time: segment.end_time,
      speaker_id: segment.speaker_name || segment.speaker_label,
      text: segment.segment_text || '',
      emotion: 'neutral', // We'll extract from emotions_json if available
      emotion_confidence: segment.confidence || 0
    }));

    // Create speakers list from unique segment speakers
    const speakerMap = new Map();
    segments.forEach(segment => {
      if (!speakerMap.has(segment.speaker_id)) {
        speakerMap.set(segment.speaker_id, {
          speaker_id: segment.speaker_id,
          start_time: segment.start_time,
          end_time: segment.end_time
        });
      } else {
        const existing = speakerMap.get(segment.speaker_id);
        existing.start_time = Math.min(existing.start_time, segment.start_time);
        existing.end_time = Math.max(existing.end_time, segment.end_time);
      }
    });

    return {
      filename: recording.original_filename || recording.filename,
      status: result.status,
      processing_time: result.processing_duration || 0,
      speakers: Array.from(speakerMap.values()),
      segments: segments
    };
  };

  const toggleExpanded = async (recordingId: number) => {
    if (expandedRecording === recordingId) {
      setExpandedRecording(null);
    } else {
      setExpandedRecording(recordingId);
      await fetchRecordingDetails(recordingId);
    }
  };

  const deleteRecording = async (recordingId: number) => {
    if (!window.confirm('Are you sure you want to delete this recording?')) {
      return;
    }

    setDeleting(prev => new Set(prev).add(recordingId));
    
    try {
      const response = await fetch(`${API_BASE}/api/recordings/${recordingId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete recording');
      }
      
      setRecordings(prev => prev.filter(r => r.id !== recordingId));
      setSelectedRecordings(prev => {
        const newSet = new Set(prev);
        newSet.delete(recordingId);
        return newSet;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete recording');
    } finally {
      setDeleting(prev => {
        const newSet = new Set(prev);
        newSet.delete(recordingId);
        return newSet;
      });
    }
  };

  const bulkDeleteRecordings = async () => {
    if (selectedRecordings.size === 0) return;
    
    if (!window.confirm(`Are you sure you want to delete ${selectedRecordings.size} recording(s)?`)) {
      return;
    }

    const recordingIds = Array.from(selectedRecordings);
    setDeleting(prev => new Set([...Array.from(prev), ...recordingIds]));
    
    try {
      const response = await fetch(`${API_BASE}/api/recordings/bulk-delete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(recordingIds),
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete recordings');
      }
      
      const result = await response.json();
      
      if (result.failed_ids && result.failed_ids.length > 0) {
        setError(`Failed to delete ${result.failed_ids.length} recording(s). Successfully deleted ${result.deleted_count}.`);
      }
      
      setRecordings(prev => prev.filter(r => !selectedRecordings.has(r.id)));
      setSelectedRecordings(new Set());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete recordings');
    } finally {
      setDeleting(prev => {
        const newSet = new Set(prev);
        recordingIds.forEach(id => newSet.delete(id));
        return newSet;
      });
    }
  };

  const toggleRecordingSelection = (recordingId: number) => {
    setSelectedRecordings(prev => {
      const newSet = new Set(prev);
      if (newSet.has(recordingId)) {
        newSet.delete(recordingId);
      } else {
        newSet.add(recordingId);
      }
      return newSet;
    });
  };

  const selectAllRecordings = () => {
    if (selectedRecordings.size === recordings.length) {
      setSelectedRecordings(new Set());
    } else {
      setSelectedRecordings(new Set(recordings.map(r => r.id)));
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
        <div className="header-controls">
          {recordings.length > 0 && (
            <div className="bulk-controls">
              <label className="select-all">
                <input
                  type="checkbox"
                  checked={selectedRecordings.size === recordings.length && recordings.length > 0}
                  onChange={selectAllRecordings}
                />
                Select All
              </label>
              {selectedRecordings.size > 0 && (
                <button 
                  onClick={bulkDeleteRecordings} 
                  className="bulk-delete-button"
                  disabled={deleting.size > 0}
                >
                  🗑️ Delete Selected ({selectedRecordings.size})
                </button>
              )}
            </div>
          )}
          <button onClick={fetchRecordings} className="refresh-button">
            🔄 Refresh
          </button>
        </div>
      </div>
      
      {recordings.length === 0 ? (
        <div className="no-recordings">
          No recordings yet. Start recording to see your history here!
        </div>
      ) : (
        <div className="recordings-list">
          {recordings.map((recording, index) => (
            <div 
              key={recording.id || `recording-${index}`} 
              className={`recording-item ${expandedRecording === recording.id ? 'expanded' : ''} ${selectedRecordings.has(recording.id) ? 'selected' : ''}`}
            >
              <div className="recording-controls">
                <input
                  type="checkbox"
                  checked={selectedRecordings.has(recording.id)}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleRecordingSelection(recording.id);
                  }}
                  className="recording-checkbox"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteRecording(recording.id);
                  }}
                  className="delete-button"
                  disabled={deleting.has(recording.id)}
                  title="Delete recording"
                >
                  {deleting.has(recording.id) ? '⏳' : '🗑️'}
                </button>
              </div>
              <div 
                className="recording-summary"
                onClick={() => toggleExpanded(recording.id)}
              >
                <div className="recording-info">
                  <div className="recording-name">
                    {recording.original_filename || recording.filename}
                  </div>
                  <div className="recording-meta">
                    {recording.duration ? formatDuration(recording.duration) : 'Unknown duration'} • {recording.file_size ? formatFileSize(recording.file_size) : 'Unknown size'} • {recording.created_at ? formatDateTime(recording.created_at) : 'Unknown date'}
                  </div>
                </div>
                <div className="recording-stats">
                  <span className="format-badge">{recording.format?.toUpperCase() || 'UNKNOWN'}</span>
                  {recording.processing_results && recording.processing_results.length > 0 && (
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

                  <div className="processing-results">
                    <h4>Processing Results ({recording.processing_results ? recording.processing_results.length : 0})</h4>
                  {recording.processing_results && recording.processing_results.length > 0 ? (
                    <div className="processing-results-content">
                      {recording.processing_results.map((result, resultIndex) => (
                        <div key={result.id || `result-${resultIndex}`} className="processing-result">
                          <div className="result-header">
                            <span className={`status-badge ${result.status}`}>
                              {result.status}
                            </span>
                            <span className="processed-time">
                              {formatDateTime(result.processed_at)}
                            </span>
                          </div>
                          
                          {result.status === 'completed' && result.speaker_segments && result.speaker_segments.length > 0 && (
                            <div className="result-actions">
                              <div className="result-summary">
                                <div className="summary-stats">
                                  <span>📝 {result.transcription ? `${result.transcription.length} chars transcribed` : 'No transcription'}</span>
                                  <span>🗣️ {result.speaker_segments.length} speaker segments</span>
                                  <span>⏱️ {result.processing_duration ? `${result.processing_duration.toFixed(1)}s processing` : 'Processing time unknown'}</span>
                                </div>
                                <button 
                                  className="view-details-button"
                                  onClick={() => setModalRecording({recording, result})}
                                >
                                  📊 View Full Analysis
                                </button>
                              </div>
                            </div>
                          )}

                          {result.status === 'completed' && (!result.speaker_segments || result.speaker_segments.length === 0) && (
                            <div className="no-processing-results">
                              <p className="no-result">
                                📊 This recording was processed but no speaker segments with text were found.
                              </p>
                            </div>
                          )}

                          {result.status === 'pending' && (
                            <div className="processing-pending">
                              <p>⏳ Processing in progress...</p>
                            </div>
                          )}

                          {result.status === 'failed' && (
                            <div className="processing-failed">
                              <p>❌ Processing failed. Please try reprocessing this recording.</p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="no-processing-results">
                      <p className="no-result">
                        📊 This recording hasn't been processed yet. 
                        {recording.file_path && recording.file_path.includes('upload_') ? 
                          ' Upload it through the main page to trigger analysis.' : 
                          ' Processing may still be in progress or failed.'
                        }
                      </p>
                      <button 
                        className="process-button"
                        onClick={() => console.log('TODO: Implement reprocessing for recording', recording.id)}
                      >
                        🔄 Process This Recording
                      </button>
                    </div>
                  )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Analysis Details Modal */}
      {modalRecording && (
        <div className="analysis-modal-overlay" onClick={() => setModalRecording(null)}>
          <div className="analysis-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📊 Full Analysis Details</h2>
              <div className="modal-recording-info">
                <h3>{modalRecording.recording.original_filename || modalRecording.recording.filename}</h3>
                <p>Processed: {formatDateTime(modalRecording.result.processed_at)}</p>
              </div>
              <button className="modal-close" onClick={() => setModalRecording(null)}>
                ✕
              </button>
            </div>
            <div className="modal-content">
              <TranscriptViewer result={convertToAnalysisResult(modalRecording.recording, modalRecording.result)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RecordingHistory;