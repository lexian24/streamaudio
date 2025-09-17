import { useState, useCallback, useRef } from 'react';

export interface ContinuousRecorderState {
  isRecording: boolean;
  isSupported: boolean;
  error: string | null;
  recordingDuration: number;
}

export interface UseContinuousRecorderReturn extends ContinuousRecorderState {
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<Blob | null>;
}

export const useContinuousRecorder = (): UseContinuousRecorderReturn => {
  const [state, setState] = useState<ContinuousRecorderState>({
    isRecording: false,
    isSupported: typeof navigator !== 'undefined' && !!navigator.mediaDevices && !!window.MediaRecorder,
    error: null,
    recordingDuration: 0,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTimeRef = useRef<number>(0);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = useCallback(async () => {
    if (!state.isSupported) {
      setState(prev => ({ ...prev, error: 'Recording not supported' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, error: null }));

      // Get high-quality audio stream for recording
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 44100,  // High quality for final recording
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Find best available format
      let mimeType = 'audio/webm;codecs=opus';
      const supportedTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/wav'
      ];
      
      for (const type of supportedTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }

      // Create MediaRecorder for continuous recording
      mediaRecorderRef.current = new MediaRecorder(stream, { 
        mimeType,
        audioBitsPerSecond: 128000  // High quality audio
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setState(prev => ({ ...prev, error: 'Recording error occurred' }));
      };

      // Start continuous recording
      mediaRecorderRef.current.start(1000); // 1 second chunks for progress updates
      startTimeRef.current = Date.now();

      // Update duration every second
      durationIntervalRef.current = setInterval(() => {
        if (startTimeRef.current > 0) {
          const duration = (Date.now() - startTimeRef.current) / 1000;
          setState(prev => ({ ...prev, recordingDuration: duration }));
        }
      }, 1000);

      setState(prev => ({ ...prev, isRecording: true, recordingDuration: 0 }));
      console.log('🎤 Continuous recording started with', mimeType);

    } catch (error) {
      console.error('Error starting recording:', error);
      setState(prev => ({
        ...prev,
        error: `Failed to start recording: ${error}`,
        isRecording: false,
      }));
    }
  }, [state.isSupported]);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
        resolve(null);
        return;
      }

      mediaRecorderRef.current.onstop = () => {
        console.log('🎤 Continuous recording stopped, creating blob...');
        
        // Create final blob from all chunks
        const blob = new Blob(chunksRef.current, { 
          type: mediaRecorderRef.current?.mimeType || 'audio/webm' 
        });
        
        console.log('📁 Recording blob created:', blob.size, 'bytes');
        resolve(blob);
      };

      // Stop recording
      mediaRecorderRef.current.stop();

      // Clean up stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      // Clear duration interval
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
        durationIntervalRef.current = null;
      }

      // Reset refs
      mediaRecorderRef.current = null;
      startTimeRef.current = 0;
      
      setState(prev => ({ 
        ...prev, 
        isRecording: false, 
        recordingDuration: 0 
      }));
    });
  }, []);

  return {
    ...state,
    startRecording,
    stopRecording,
  };
};