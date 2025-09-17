import React, { useState, useCallback } from 'react';
import { AudioAnalysisResult } from '../types/audio';
import './AudioUploader.css';

interface AudioUploaderProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (result: AudioAnalysisResult) => void;
  onError: (error: string) => void;
  isProcessing: boolean;
}

const AudioUploader: React.FC<AudioUploaderProps> = ({ 
  onAnalysisStart, 
  onAnalysisComplete, 
  onError, 
  isProcessing 
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): string | null => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/flac', 'audio/ogg'];
    const allowedExtensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg'];

    if (file.size > maxSize) {
      return 'File size must be less than 50MB';
    }

    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      return 'Please select a valid audio file (.wav, .mp3, .m4a, .flac, .ogg)';
    }

    return null;
  };

  const uploadAndAnalyzeFile = useCallback(async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      onError(validationError);
      return;
    }

    setError(null);
    onAnalysisStart();
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const result: AudioAnalysisResult = await response.json();
      onAnalysisComplete(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      onError(errorMessage);
    }
  }, [onAnalysisStart, onAnalysisComplete, onError]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && !isProcessing) {
      uploadAndAnalyzeFile(files[0]);
    }
  }, [isProcessing, uploadAndAnalyzeFile]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0 && !isProcessing) {
      uploadAndAnalyzeFile(files[0]);
    }
  };

  return (
    <div className="audio-uploader">
      <div
        className={`upload-area ${dragOver ? 'drag-over' : ''} ${isProcessing ? 'uploading' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-content">
          {isProcessing ? (
            <div className="upload-loading">
              <div className="spinner"></div>
              <p>Processing with MERaLiON...</p>
            </div>
          ) : (
            <>
              <div className="upload-icon">🎤</div>
              <h3>Upload Audio for FastAudio Analysis</h3>
              <p>Drop your audio file for processing</p>
              <p className="file-types">Supported: .wav, .mp3, .m4a, .flac, .ogg (max 50MB)</p>
              <input
                type="file"
                accept=".wav,.mp3,.m4a,.flac,.ogg,audio/*"
                onChange={handleFileSelect}
                className="file-input"
                disabled={isProcessing}
              />
            </>
          )}
        </div>
      </div>
      
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}
    </div>
  );
};

export default AudioUploader;