import React, { useMemo } from 'react';
import { StreamingResult } from '../hooks/useWebSocket';
import '../styles/StreamingTranscript.css';

interface StreamingTranscriptProps {
  results: StreamingResult[];
  isStreaming: boolean;
}

interface ProcessedSegment {
  start_time: number;
  end_time: number;
  speaker_id: string;
  text: string;
  emotion: string;
  emotion_confidence: number;
  duration: number;
  timestamp: number;
  status: string;
  chunk_id: string;
  isPending: boolean;
}

const StreamingTranscript: React.FC<StreamingTranscriptProps> = ({ results, isStreaming }) => {
  // Process results to concatenate transcriptions and handle speaker assignments
  const { concatenatedText, speakerSegments, sessionSummary, processingStats } = useMemo(() => {
    let pendingTranscriptions: Array<{text: string, start_time: number, end_time: number}> = [];
    const confirmedSegments: ProcessedSegment[] = [];
    let latestSessionSummary: any = null;
    let totalProcessingTime = 0;
    let processedChunks = 0;

    results.forEach(result => {
      // Handle legacy format (backwards compatibility)
      if (result.type === 'chunk_processed' && result.segments) {
        result.segments.forEach(segment => {
          const processedSegment: ProcessedSegment = {
            ...segment,
            timestamp: result.timestamp,
            status: 'speaker_assigned',
            chunk_id: `legacy_${segment.start_time}`,
            isPending: false
          };
          confirmedSegments.push(processedSegment);
        });
        
        if (result.processing_time) {
          totalProcessingTime += result.processing_time;
          processedChunks++;
        }
      }
      
      // Handle new transcription results (concatenate pending transcriptions)
      else if (result.type === 'transcription' && result.result) {
        const transcription = result.result;
        if (transcription.text.trim()) {
          pendingTranscriptions.push({
            text: transcription.text.trim(),
            start_time: transcription.start_time,
            end_time: transcription.end_time
          });
        }
        
        if (result.processing_time) {
          totalProcessingTime += result.processing_time;
          processedChunks++;
        }
      }
      
      // Handle diarization results (convert pending transcriptions to speaker segments)
      else if (result.type === 'diarization' && result.updated_transcriptions) {
        // Clear pending transcriptions and create speaker segments
        result.updated_transcriptions.forEach(updatedTranscription => {
          const processedSegment: ProcessedSegment = {
            start_time: updatedTranscription.start_time,
            end_time: updatedTranscription.end_time,
            speaker_id: updatedTranscription.speaker_id,
            text: updatedTranscription.text,
            emotion: updatedTranscription.emotion || 'neutral',
            emotion_confidence: updatedTranscription.emotion_confidence || 0,
            duration: updatedTranscription.end_time - updatedTranscription.start_time,
            timestamp: result.timestamp,
            status: updatedTranscription.status,
            chunk_id: updatedTranscription.chunk_id,
            isPending: false
          };
          confirmedSegments.push(processedSegment);
        });
        
        // Remove processed transcriptions from pending
        pendingTranscriptions = [];
        
        if (result.processing_time) {
          totalProcessingTime += result.processing_time;
          processedChunks++;
        }
      }
      
      // Handle session summary
      else if (result.type === 'stream_stopped' && result.session_summary) {
        latestSessionSummary = result.session_summary;
      }
    });

    // Create concatenated text from pending transcriptions
    const concatenatedText = pendingTranscriptions.map(t => t.text).join(' ');
    
    // Sort confirmed segments by start time
    confirmedSegments.sort((a, b) => a.start_time - b.start_time);

    return {
      concatenatedText,
      speakerSegments: confirmedSegments,
      sessionSummary: latestSessionSummary,
      processingStats: {
        totalProcessingTime,
        processedChunks,
        averageProcessingTime: processedChunks > 0 ? totalProcessingTime / processedChunks : 0,
      }
    };
  }, [results]);

  // Group consecutive segments by the same speaker for display
  const conversationSegments = useMemo(() => {
    if (speakerSegments.length === 0) return [];
    
    const grouped: ProcessedSegment[] = [];
    let currentGroup: ProcessedSegment | null = null;
    
    speakerSegments.forEach(segment => {
      if (!currentGroup || currentGroup.speaker_id !== segment.speaker_id) {
        // Start a new group for different speaker
        if (currentGroup) {
          grouped.push(currentGroup);
        }
        currentGroup = {
          ...segment,
          text: segment.text,
          end_time: segment.end_time,
          duration: segment.duration
        };
      } else {
        // Same speaker - concatenate text and extend duration
        currentGroup.text = (currentGroup.text + ' ' + segment.text).trim();
        currentGroup.end_time = segment.end_time;
        currentGroup.duration = currentGroup.end_time - currentGroup.start_time;
        // Keep the latest emotion and confidence
        currentGroup.emotion = segment.emotion;
        currentGroup.emotion_confidence = segment.emotion_confidence;
      }
    });
    
    // Don't forget the last group
    if (currentGroup) {
      grouped.push(currentGroup);
    }
    
    return grouped;
  }, [speakerSegments]);

  // Group confirmed speaker segments by speaker for summary display
  const groupedSpeakerSegments = useMemo(() => {
    const grouped: Record<string, ProcessedSegment[]> = {};
    
    speakerSegments.forEach(segment => {
      if (!grouped[segment.speaker_id]) {
        grouped[segment.speaker_id] = [];
      }
      grouped[segment.speaker_id].push(segment);
    });

    return grouped;
  }, [speakerSegments]);

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getEmotionEmoji = (emotion: string) => {
    const emotionMap: Record<string, string> = {
      happy: '😊',
      sad: '😢',
      angry: '😠',
      fear: '😨',
      surprise: '😲',
      disgust: '🤢',
      neutral: '😐',
      joy: '😄',
      love: '❤️',
      excitement: '🤩',
    };
    return emotionMap[emotion.toLowerCase()] || '😐';
  };

  const getEmotionColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      happy: '#10b981',
      joy: '#10b981',
      love: '#ef4444',
      excitement: '#f59e0b',
      neutral: '#6b7280',
      sad: '#3b82f6',
      angry: '#ef4444',
      fear: '#8b5cf6',
      surprise: '#f59e0b',
      disgust: '#84cc16',
    };
    return colorMap[emotion.toLowerCase()] || '#6b7280';
  };

  if (!concatenatedText && conversationSegments.length === 0 && !isStreaming) {
    return (
      <div className="streaming-transcript empty">
        <div className="empty-state">
          <div className="empty-icon">🎤</div>
          <h3>No audio processed yet</h3>
          <p>Start recording to see real-time transcription and analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className="streaming-transcript">
      <div className="transcript-header">
        <h2>Live Transcript</h2>
        {isStreaming && (
          <div className="streaming-indicator">
            <span className="pulse-dot"></span>
            Live
          </div>
        )}
      </div>

      {/* Session Summary */}
      {(sessionSummary || concatenatedText || conversationSegments.length > 0) && (
        <div className="session-summary">
          <div className="summary-stats">
            <div className="stat">
              <span className="stat-value">{sessionSummary?.total_speakers || Object.keys(groupedSpeakerSegments).length}</span>
              <span className="stat-label">Speakers</span>
            </div>
            <div className="stat">
              <span className="stat-value">{speakerSegments.length}</span>
              <span className="stat-label">Segments</span>
            </div>
            {processingStats.averageProcessingTime > 0 && (
              <div className="stat">
                <span className="stat-value">{processingStats.averageProcessingTime.toFixed(2)}s</span>
                <span className="stat-label">Avg Processing</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Transcript Content */}
      <div className="transcript-content">
        {/* Show concatenated pending transcription */}
        {concatenatedText && (
          <div className="segment-item pending">
            <div className="segment-header">
              <div className="speaker-info">
                <span className="speaker-id pending-speaker">
                  Speaker Unknown
                  <span className="pending-indicator">⏳</span>
                </span>
              </div>
              <div className="emotion-info">
                <span className="pending-emotion">
                  ⏳ Analyzing...
                </span>
              </div>
            </div>
            <div className="segment-text">
              {concatenatedText}
            </div>
            <div className="pending-notice">
              Real-time transcription • Speaker identification in progress...
            </div>
          </div>
        )}

        {/* Show confirmed speaker segments grouped by consecutive speech */}
        {conversationSegments.length > 0 && (
          <div className="segments-container">
            {conversationSegments.map((segment, index) => (
              <div 
                key={segment.chunk_id || `${segment.speaker_id}-${segment.start_time}-${index}`}
                className="segment-item confirmed"
              >
                <div className="segment-header">
                  <div className="speaker-info">
                    <span className="speaker-id">
                      {segment.speaker_id}
                    </span>
                    <span className="segment-time">
                      {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                    </span>
                  </div>
                  <div className="emotion-info">
                    <span 
                      className="emotion-badge"
                      style={{ color: getEmotionColor(segment.emotion) }}
                    >
                      {getEmotionEmoji(segment.emotion)} {segment.emotion}
                    </span>
                    <span className="confidence">
                      {Math.round(segment.emotion_confidence * 100)}%
                    </span>
                  </div>
                </div>
                <div className="segment-text">
                  {segment.text}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Show waiting state */}
        {!concatenatedText && conversationSegments.length === 0 && isStreaming && (
          <div className="waiting-for-audio">
            <div className="waiting-animation">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <p>Listening for speech...</p>
          </div>
        )}
      </div>

      {/* Speaker Summary */}
      {Object.keys(groupedSpeakerSegments).length > 0 && (
        <div className="speaker-summary">
          <h3>Speaker Summary</h3>
          <div className="speakers-grid">
            {Object.entries(groupedSpeakerSegments).map(([speakerId, speakerSegs]) => {
              const totalDuration = speakerSegs.reduce((sum, seg) => sum + seg.duration, 0);
              const lastSegment = speakerSegs[speakerSegs.length - 1];
              
              return (
                <div key={speakerId} className="speaker-card">
                  <div className="speaker-header">
                    <span className="speaker-name">{speakerId}</span>
                    <span className="speaker-duration">{formatTime(totalDuration)}</span>
                  </div>
                  <div className="speaker-stats">
                    <span className="segment-count">{speakerSegs.length} segments</span>
                    {lastSegment && (
                      <span 
                        className="last-emotion"
                        style={{ color: getEmotionColor(lastSegment.emotion) }}
                      >
                        {getEmotionEmoji(lastSegment.emotion)} {lastSegment.emotion}
                      </span>
                    )}
                  </div>
                  {lastSegment && (
                    <div className="last-text">
                      "{lastSegment.text.slice(0, 60)}{lastSegment.text.length > 60 ? '...' : ''}"
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default StreamingTranscript;