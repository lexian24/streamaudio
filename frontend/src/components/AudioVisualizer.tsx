import React, { useEffect, useRef } from 'react';
import '../styles/AudioVisualizer.css';

interface AudioVisualizerProps {
  audioLevel: number;
  isRecording: boolean;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ audioLevel, isRecording }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      if (isRecording) {
        // Draw background circle
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 40, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
        ctx.fill();

        // Draw pulsing circle based on audio level
        const radius = 20 + (audioLevel * 20);
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, radius, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(59, 130, 246, ${0.3 + audioLevel * 0.7})`;
        ctx.fill();

        // Draw center dot
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();

        // Draw audio bars around the circle
        const numBars = 12;
        const barHeight = audioLevel * 15 + 5;
        
        for (let i = 0; i < numBars; i++) {
          const angle = (i / numBars) * 2 * Math.PI;
          const x1 = width / 2 + Math.cos(angle) * 50;
          const y1 = height / 2 + Math.sin(angle) * 50;
          const x2 = width / 2 + Math.cos(angle) * (50 + barHeight);
          const y2 = height / 2 + Math.sin(angle) * (50 + barHeight);

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.strokeStyle = `rgba(59, 130, 246, ${0.4 + audioLevel * 0.6})`;
          ctx.lineWidth = 3;
          ctx.lineCap = 'round';
          ctx.stroke();
        }
      } else {
        // Draw idle state
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 30, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(156, 163, 175, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw microphone icon (simplified)
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 12, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
        ctx.fill();
      }

      if (isRecording) {
        animationRef.current = requestAnimationFrame(draw);
      }
    };

    if (isRecording) {
      draw();
    } else {
      draw(); // Draw once for idle state
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [audioLevel, isRecording]);

  return (
    <div className="audio-visualizer">
      <canvas
        ref={canvasRef}
        width={120}
        height={120}
        className="visualizer-canvas"
      />
      <div className="audio-level-indicator">
        <div className="level-bar">
          <div 
            className="level-fill"
            style={{ width: `${audioLevel * 100}%` }}
          />
        </div>
        <span className="level-text">
          {isRecording ? `${Math.round(audioLevel * 100)}%` : 'Ready'}
        </span>
      </div>
    </div>
  );
};

export default AudioVisualizer;