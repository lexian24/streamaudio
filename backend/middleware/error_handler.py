"""
Global error handling middleware.
"""

import logging
import traceback
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and handle all exceptions globally.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip middleware for WebSocket connections
        if request.url.path.startswith("/ws/") or "websocket" in request.headers.get("upgrade", "").lower():
            return await call_next(request)

        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Validation Error",
                    "details": str(e),
                    "code": "VALIDATION_ERROR"
                }
            )

        except FileNotFoundError as e:
            # Handle file not found errors
            logger.warning(f"File not found: {e}")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "error": "Resource Not Found",
                    "details": str(e),
                    "code": "NOT_FOUND"
                }
            )

        except PermissionError as e:
            # Handle permission errors
            logger.warning(f"Permission denied: {e}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "success": False,
                    "error": "Permission Denied",
                    "details": str(e),
                    "code": "FORBIDDEN"
                }
            )

        except Exception as e:
            # Handle all other unexpected errors
            logger.error(f"Unhandled exception: {e}")
            logger.error(traceback.format_exc())

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Internal Server Error",
                    "details": "An unexpected error occurred. Please contact support.",
                    "code": "INTERNAL_ERROR"
                }
            )


def error_handler_middleware(app):
    """Add error handler middleware to the app."""
    app.add_middleware(ErrorHandlerMiddleware)
