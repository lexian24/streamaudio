export interface Speaker {
  speaker_id: string;
  start_time: number;
  end_time: number;
}

export interface Segment {
  start_time: number;
  end_time: number;
  speaker_id: string;
  text: string;
  emotion: string;
  emotion_confidence: number;
}

export interface AudioAnalysisResult {
  filename: string;
  status: string;
  processing_time: number;
  speakers: Speaker[];
  segments: Segment[];
  error_message?: string;
}

export type EmotionType = 'angry' | 'disgusted' | 'fearful' | 'happy' | 'neutral' | 'sad' | 'surprised' | 'excited' | 'calm';

export const EMOTION_COLORS: Record<EmotionType, string> = {
  happy: '#4CAF50',      // Green
  sad: '#2196F3',        // Blue  
  angry: '#F44336',      // Red
  fearful: '#FF9800',    // Orange
  surprised: '#9C27B0',  // Purple
  disgusted: '#795548',  // Brown
  neutral: '#607D8B',    // Blue Grey
  excited: '#FF5722',    // Deep Orange
  calm: '#00BCD4'        // Cyan
};

export const EMOTION_EMOJIS: Record<EmotionType, string> = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  fearful: '😨',
  surprised: '😲',
  disgusted: '🤢',
  neutral: '😐',
  excited: '🤩',
  calm: '😌'
};