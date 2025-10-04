"""
Request/response logging middleware with correlation IDs.
"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses with correlation IDs.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip middleware for WebSocket connections
        if request.url.path.startswith("/ws/") or "websocket" in request.headers.get("upgrade", "").lower():
            return await call_next(request)

        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # Log request
        start_time = time.time()
        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} - Started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_host": request.client.host if request.client else None,
            }
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} - "
            f"Completed {response.status_code} in {duration:.3f}s",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": duration,
            }
        )

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


def logging_middleware(app):
    """Add logging middleware to the app."""
    app.add_middleware(LoggingMiddleware)
