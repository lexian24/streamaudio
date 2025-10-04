-- Migration: Add processing_tasks table for Celery task tracking
-- Date: 2025-01-03
-- Description: Adds ProcessingTask model for tracking background Celery tasks

CREATE TABLE IF NOT EXISTS processing_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(255) NOT NULL UNIQUE,
    recording_id INTEGER,
    task_type VARCHAR(50) NOT NULL,
    task_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'queued' NOT NULL,
    progress INTEGER DEFAULT 0,
    result_data JSON,
    error_message TEXT,
    traceback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    worker_name VARCHAR(100),
    retries INTEGER DEFAULT 0,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_processing_tasks_task_id ON processing_tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_status ON processing_tasks(status);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_status_created ON processing_tasks(status, created_at);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_recording_id ON processing_tasks(recording_id);

-- Add comments (SQLite doesn't support comments, this is for documentation)
-- task_id: Unique Celery task ID
-- recording_id: Foreign key to recordings table
-- status: queued, processing, completed, failed, cancelled
-- progress: 0-100 percentage
-- result_data: Full processing results (JSON)
-- error_message: Error message if failed
-- traceback: Full traceback for debugging
