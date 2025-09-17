import { useState, useCallback, useRef, useEffect } from 'react';

export interface MicrophoneState {
  isRecording: boolean;
  isSupported: boolean;
  audioLevel: number;
  error: string | null;
}

export interface UseMicrophoneReturn extends MicrophoneState {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  onAudioData: (callback: (audioData: Float32Array, timestamp: number) => void) => void;
}

export const useMicrophone = (): UseMicrophoneReturn => {
  const [state, setState] = useState<MicrophoneState>({
    isRecording: false,
    isSupported: typeof navigator !== 'undefined' && !!navigator.mediaDevices,
    audioLevel: 0,
    error: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioDataCallbackRef = useRef<((audioData: Float32Array, timestamp: number) => void) | null>(null);
  const animationFrameRef = useRef<number | undefined>(undefined);

  // Audio level monitoring
  const updateAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteFrequencyData(dataArray);

    // Calculate RMS level
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);
    const level = Math.min(rms / 128, 1); // Normalize to 0-1

    setState(prev => ({ ...prev, audioLevel: level }));

    if (state.isRecording) {
      animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
    }
  }, [state.isRecording]);

  // Setup audio processing
  const setupAudioProcessing = useCallback(async (stream: MediaStream) => {
    try {
      // Create audio context
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000, // Target sample rate for Whisper
      });

      // Create source and analyser
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.smoothingTimeConstant = 0.8;

      sourceRef.current.connect(analyserRef.current);

      // Safari compatibility: Use ScriptProcessorNode for direct audio capture
      const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
      
      if (isSafari || !window.MediaRecorder) {
        
        // Use ScriptProcessorNode for Safari with immediate streaming
        const bufferSize = 4096;
        const processor = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);
        
        sourceRef.current.connect(processor);
        processor.connect(audioContextRef.current.destination);
        
        processor.onaudioprocess = (event) => {
          if (audioDataCallbackRef.current) {
            const inputBuffer = event.inputBuffer;
            const audioData = inputBuffer.getChannelData(0);
            
            // Send immediately without buffering to avoid gaps and overlaps
            const timestamp = Date.now();
            audioDataCallbackRef.current(new Float32Array(audioData), timestamp);
          }
        };
        
        // Store processor reference for cleanup
        (sourceRef.current as any).processor = processor;
        
      } else {
        
        // Try different MIME types for better compatibility
        let mimeType = 'audio/webm';
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

        console.log('Using MIME type:', mimeType);
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });

        // Handle recorded data
        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0 && audioDataCallbackRef.current) {
            
            // Convert blob to array buffer
            event.data.arrayBuffer().then(buffer => {
              console.log('Sending audio buffer, size:', buffer.byteLength);
              // Create a simple Float32Array representation
              const audioData = new Float32Array(buffer.byteLength / 4);
              const view = new DataView(buffer);
              
              // Convert bytes to float32 values (simplified)
              for (let i = 0; i < audioData.length && i * 4 < buffer.byteLength - 3; i++) {
                try {
                  audioData[i] = view.getFloat32(i * 4, true);
                } catch (e) {
                  audioData[i] = 0; // Fallback for out-of-bounds
                }
              }
              
              audioDataCallbackRef.current!(audioData, Date.now());
            }).catch(error => {
              console.error('Error processing audio data:', error);
            });
          }
        };

        // Start recording with smaller, consistent chunks for better quality
        mediaRecorderRef.current.start(100); // 100ms chunks for smoother audio
      }

      // Start audio level monitoring
      updateAudioLevel();

    } catch (error) {
      console.error('Error setting up audio processing:', error);
      setState(prev => ({ ...prev, error: `Audio setup failed: ${error}` }));
    }
  }, [updateAudioLevel]);

  const startRecording = useCallback(async () => {
    if (!state.isSupported) {
      setState(prev => ({ ...prev, error: 'Microphone not supported' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, error: null }));
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,  // Disable for more natural VAD detection
          noiseSuppression: false,  // Let VAD handle natural speech patterns
          autoGainControl: false,   // Preserve original audio levels for VAD
        },
      });

      streamRef.current = stream;
      await setupAudioProcessing(stream);

      setState(prev => ({ ...prev, isRecording: true }));
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setState(prev => ({
        ...prev,
        error: `Failed to access microphone: ${error}`,
        isRecording: false,
      }));
    }
  }, [state.isSupported, setupAudioProcessing]);

  const stopRecording = useCallback(() => {
    // Stop MediaRecorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Clean up ScriptProcessorNode for Safari
    if (sourceRef.current && (sourceRef.current as any).processor) {
      const processor = (sourceRef.current as any).processor;
      processor.disconnect();
      sourceRef.current.disconnect(processor);
      (sourceRef.current as any).processor = null;
    }

    // Stop audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Cancel animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    // Reset refs
    mediaRecorderRef.current = null;
    analyserRef.current = null;
    sourceRef.current = null;

    setState(prev => ({ 
      ...prev, 
      isRecording: false, 
      audioLevel: 0 
    }));
  }, []);

  const onAudioData = useCallback((callback: (audioData: Float32Array, timestamp: number) => void) => {
    audioDataCallbackRef.current = callback;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    ...state,
    startRecording,
    stopRecording,
    onAudioData,
  };
};