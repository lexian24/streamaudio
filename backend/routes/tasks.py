"""
Task management API endpoints.

Provides REST API for checking status of background processing tasks,
retrieving results, and managing task lifecycle.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from database import get_database_service
from database.services import DatabaseService
from celery_app import celery_app

router = APIRouter()
logger = logging.getLogger(__name__)


# Response models
class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: str  # queued, processing, completed, failed, cancelled
    progress: int  # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class TaskListResponse(BaseModel):
    """Response model for task list"""
    tasks: List[TaskStatusResponse]
    total: int


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get status and results of a processing task.

    Args:
        task_id: Celery task ID

    Returns:
        Task status and results if completed
    """
    try:
        # Get task from database
        task = await db_service.processing_tasks.get_task(task_id)

        if not task:
            # Check if task exists in Celery
            celery_task = celery_app.AsyncResult(task_id)
            if celery_task.state == 'PENDING':
                raise HTTPException(status_code=404, detail="Task not found")

            # Task exists in Celery but not in DB (shouldn't happen)
            return TaskStatusResponse(
                task_id=task_id,
                status=celery_task.state.lower(),
                progress=0,
                result=celery_task.result if celery_task.successful() else None,
                error=str(celery_task.info) if celery_task.failed() else None,
                created_at="",
                started_at=None,
                completed_at=None
            )

        # Return task from database
        return TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            result=task.result_data,
            error=task.error_message,
            created_at=task.created_at.isoformat() if task.created_at else "",
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, regex="^(queued|processing|completed|failed|cancelled)$"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    List recent processing tasks.

    Args:
        limit: Maximum number of tasks to return
        offset: Number of tasks to skip
        status: Filter by task status (optional)

    Returns:
        List of tasks with pagination
    """
    try:
        tasks = await db_service.processing_tasks.get_recent_tasks(limit=limit, offset=offset)

        # Filter by status if provided
        if status:
            tasks = [t for t in tasks if t.status == status]

        task_responses = [
            TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result=task.result_data,
                error=task.error_message,
                created_at=task.created_at.isoformat() if task.created_at else "",
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=task.completed_at.isoformat() if task.completed_at else None
            )
            for task in tasks
        ]

        return TaskListResponse(
            tasks=task_responses,
            total=len(task_responses)
        )

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Cancel a pending or processing task.

    Args:
        task_id: Celery task ID

    Returns:
        Cancellation confirmation
    """
    try:
        task = await db_service.processing_tasks.cancel_task(task_id)

        return {
            "status": "cancelled",
            "task_id": task.task_id,
            "message": "Task cancelled successfully"
        }

    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_tasks(
    days: int = Query(7, ge=1, le=90, description="Delete tasks older than this many days"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Clean up old completed/failed tasks.

    Args:
        days: Delete tasks older than this many days (default: 7)

    Returns:
        Number of tasks deleted
    """
    try:
        deleted_count = await db_service.processing_tasks.delete_old_tasks(days=days)

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} tasks older than {days} days"
        }

    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup tasks: {str(e)}")


@router.get("/recording/{recording_id}", response_model=TaskListResponse)
async def get_recording_tasks(
    recording_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get all tasks associated with a specific recording.

    Args:
        recording_id: Database ID of the recording

    Returns:
        List of tasks for the recording
    """
    try:
        tasks = await db_service.processing_tasks.get_tasks_by_recording(recording_id)

        task_responses = [
            TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result=task.result_data,
                error=task.error_message,
                created_at=task.created_at.isoformat() if task.created_at else "",
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=task.completed_at.isoformat() if task.completed_at else None
            )
            for task in tasks
        ]

        return TaskListResponse(
            tasks=task_responses,
            total=len(task_responses)
        )

    except Exception as e:
        logger.error(f"Failed to get recording tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording tasks: {str(e)}")
