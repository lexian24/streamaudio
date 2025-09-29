import React, { useState } from 'react';
import './App.css';
import AudioUploader from './components/AudioUploader';
import TranscriptViewer from './components/TranscriptViewer';
import StreamingInterface from './components/StreamingInterface';
import SpeakerManagement from './components/SpeakerManagement';
import { AudioAnalysisResult } from './types/audio';

type ViewMode = 'upload' | 'streaming' | 'speakers';

function App() {
  const [analysisResult, setAnalysisResult] = useState<AudioAnalysisResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('upload');

  const handleAnalysisComplete = (result: AudioAnalysisResult) => {
    setAnalysisResult(result);
    setIsProcessing(false);
    setError(null);
  };

  const handleAnalysisStart = () => {
    setIsProcessing(true);
    setAnalysisResult(null);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setIsProcessing(false);
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setIsProcessing(false);
    setError(null);
  };

  const switchToStreaming = () => {
    setViewMode('streaming');
    handleReset();
  };

  const switchToUpload = () => {
    setViewMode('upload');
    handleReset();
  };

  const switchToSpeakers = () => {
    setViewMode('speakers');
    handleReset();
  };

  if (viewMode === 'streaming') {
    return <StreamingInterface onBack={switchToUpload} />;
  }

  if (viewMode === 'speakers') {
    return <SpeakerManagement onBack={switchToUpload} />;
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>⚡ FastAudio Analysis</h1>
        <p>Simple and fast audio processing with transcription, speaker identification, and emotion recognition</p>
        <div className="mode-switcher">
          <button 
            className="mode-button active"
            onClick={switchToUpload}
          >
            📁 Upload File
          </button>
          <button 
            className="mode-button"
            onClick={switchToStreaming}
          >
            🎤 Auto Recording
          </button>
          <button 
            className="mode-button"
            onClick={switchToSpeakers}
          >
            👥 Speaker Management
          </button>
        </div>
      </header>
      
      <main className="app-main">
        {error && (
          <div className="error-banner">
            <span className="error-icon">❌</span>
            <span className="error-text">{error}</span>
            <button className="error-close" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {!analysisResult && !isProcessing && (
          <>
            <div className="feature-highlights">
              <div className="feature-card">
                <span className="feature-icon">⚡</span>
                <h3>Fast Processing</h3>
                <p>Lightweight pipeline using for quick audio analysis</p>
              </div>
              <div className="feature-card">
                <span className="feature-icon">🎭</span>
                <h3>Emotion Detection</h3>
                <p>Audio based emotion recognition</p>
              </div>
              <div className="feature-card">
                <span className="feature-icon">🗣️</span>
                <h3>Speaker Diarization</h3>
                <p>Accurate speaker separation and identification</p>
              </div>
            </div>
            <div className="upload-section">
              <AudioUploader 
                onAnalysisStart={handleAnalysisStart}
                onAnalysisComplete={handleAnalysisComplete}
                onError={handleError}
                isProcessing={isProcessing}
              />
            </div>
          </>
        )}

        {isProcessing && (
          <div className="processing-section">
            <div className="processing-container">
              <div className="spinner"></div>
              <h3>⚡ FastAudio Processing</h3>
              <p>Analyzing with Whisper transcription, pyannote speaker diarization, and emotion recognition models.</p>
            </div>
            <div className="action-buttons">
              <button className="reset-button" onClick={handleReset}>
                Cancel & Upload New File
              </button>
            </div>
          </div>
        )}

        {analysisResult && (
          <div className="results-section">
            <div className="results-header">
              <h2>Analysis Complete!</h2>
              <button className="reset-button" onClick={handleReset}>
                Analyze Another File
              </button>
            </div>
            <TranscriptViewer result={analysisResult} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by Whisper, pyannote.audio, and Transformers</p>
      </footer>
    </div>
  );
}

export default App;
