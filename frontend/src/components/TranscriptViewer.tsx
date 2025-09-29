import React, { useMemo } from 'react';
import { AudioAnalysisResult } from '../types/audio';
import './TranscriptViewer.css';

interface TranscriptViewerProps {
  result: AudioAnalysisResult;
}

const TranscriptViewer: React.FC<TranscriptViewerProps> = ({ result }) => {
  const speakerColors = useMemo(() => {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];
    const colorMap: Record<string, string> = {};
    result.speakers.forEach((speaker, index) => {
      colorMap[speaker.speaker_id] = colors[index % colors.length];
    });
    return colorMap;
  }, [result.speakers]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`;
  };

  const getSpeakerName = (speakerId: string) => {
    return speakerId.replace('SPEAKER_', 'Speaker ');
  };


  return (
    <div className="transcript-viewer">
      {/* FastAudio Analysis Summary */}
      <div className="analysis-summary">
        <div className="summary-header">
          <h3>⚡ FastAudio Analysis Results</h3>
          <p>Whisper transcription with speaker diarization and emotion analysis</p>
        </div>
        <div className="summary-grid">
          <div className="summary-item">
            <div className="summary-value">{result.speakers.length}</div>
            <div className="summary-label">Speakers Detected</div>
          </div>
          <div className="summary-item">
            <div className="summary-value">{result.segments.filter(s => s.text && s.text.trim().length > 0).length}</div>
            <div className="summary-label">Speech Segments</div>
          </div>
          <div className="summary-item">
            <div className="summary-value">
              {result.processing_time ? `${result.processing_time.toFixed(1)}s` : 'Processing...'}
            </div>
            <div className="summary-label">Processing Time</div>
          </div>
        </div>
      </div>

      {/* Speaker Legend */}
      <div className="speaker-legend">
        <h3>Speakers</h3>
        <div className="speaker-list">
          {result.speakers.map((speaker) => (
            <div key={speaker.speaker_id} className="speaker-item">
              <div 
                className="speaker-color" 
                style={{ backgroundColor: speakerColors[speaker.speaker_id] }}
              ></div>
              <span className="speaker-name">{getSpeakerName(speaker.speaker_id)}</span>
              <span className="speaker-duration">
                {formatTime(speaker.start_time)} - {formatTime(speaker.end_time)}
              </span>
            </div>
          ))}
        </div>
      </div>


      {/* Full Transcript */}
      <div className="transcript-section">
        <h3>Full Transcript</h3>
        <div className="transcript-content">
          {result.segments
            .filter(segment => segment.text && segment.text.trim().length > 0)
            .map((segment, index) => (
            <div key={index} className="transcript-segment">
              <div className="segment-header">
                <div className="segment-info">
                  <span 
                    className="speaker-badge"
                    style={{ 
                      backgroundColor: speakerColors[segment.speaker_id],
                      color: 'white'
                    }}
                  >
                    {getSpeakerName(segment.speaker_id)}
                  </span>
                  <span className="timestamp">
                    {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                  </span>
                  <span className="duration">
                    ({formatTime(segment.end_time - segment.start_time)})
                  </span>
                </div>
                <div className="emotion-info">
                  <span 
                    className="emotion-badge"
                  >
                    {segment.emotion}
                  </span>
                  <span className="confidence">
                    {formatConfidence(segment.emotion_confidence)}
                  </span>
                </div>
              </div>
              
              <div className="segment-text">
                {segment.text}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline Visualization */}
      <div className="timeline-section">
        <h3>Timeline</h3>
        <div className="timeline-container">
          <div className="timeline">
            {result.segments
              .filter(segment => segment.text && segment.text.trim().length > 0)
              .map((segment, index) => {
              const duration = result.speakers.reduce((max, speaker) => 
                Math.max(max, speaker.end_time), 0);
              const left = (segment.start_time / duration) * 100;
              const width = ((segment.end_time - segment.start_time) / duration) * 100;
              
              return (
                <div
                  key={index}
                  className="timeline-segment"
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    backgroundColor: speakerColors[segment.speaker_id]
                  }}
                  title={`${getSpeakerName(segment.speaker_id)}: ${segment.emotion} (${formatConfidence(segment.emotion_confidence)})`}
                >
                </div>
              );
            })}
          </div>
          <div className="timeline-labels">
            <span>0:00</span>
            <span>{formatTime(result.speakers.reduce((max, speaker) => 
              Math.max(max, speaker.end_time), 0))}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranscriptViewer;