"""FastAPI application factory for the backend.

Provides create_app() that wires configuration, routers and basic error handling.
"""
import logging
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.app.api.routes import router as api_router
from src.app.api.websocket import router as ws_router
from src.app.core.settings import Settings
from src.app.core.settings import settings as default_settings
from src.app.endpoints.health import router as health_router
from src.inference.runtime_logging import (
    RuntimeLogContext,
    configure_runtime_logging,
    get_build_metadata,
    log_event,
)

logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional pre-built Settings instance (useful for tests).

    Returns:
        Configured FastAPI instance with routers mounted.
    """
    configure_runtime_logging(log_file="backend.log")
    app_settings = settings or default_settings
    build_metadata = get_build_metadata()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        debug=app_settings.debug,
    )
    app.state.settings = app_settings

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    log_event(
        logger,
        logging.INFO,
        "api_started",
        "API application initialized.",
        RuntimeLogContext(session_id=uuid.uuid4().hex),
        app_name=app_settings.app_name,
        app_version=app_settings.app_version,
        debug=app_settings.debug,
        **build_metadata,
    )

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        import time
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id

        log_context = RuntimeLogContext(session_id=request_id)
        log_event(
            logger,
            logging.INFO,
            "http_request_start",
            f"HTTP {request.method} {request.url.path} started.",
            log_context,
            http_method=request.method,
            http_path=request.url.path,
        )

        start_time = time.monotonic()
        try:
            response = await call_next(request)
            duration = round(time.monotonic() - start_time, 4)
            log_event(
                logger,
                logging.INFO,
                "http_request_completed",
                f"HTTP {request.method} {request.url.path} completed with {response.status_code}.",
                log_context,
                http_method=request.method,
                http_path=request.url.path,
                status_code=response.status_code,
                duration_s=duration,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            duration = round(time.monotonic() - start_time, 4)
            log_event(
                logger,
                logging.ERROR,
                "http_request_failed",
                f"HTTP {request.method} {request.url.path} failed: {exc}",
                log_context,
                exc_info=True,
                http_method=request.method,
                http_path=request.url.path,
                error_type=type(exc).__name__,
                duration_s=duration,
            )
            raise

    @app.on_event("startup")
    def on_startup() -> None:
        from src.app.db.session import init_db
        init_db()

    app.include_router(health_router)
    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws", tags=["websocket"])
    app.include_router(ws_router, prefix="/api/websocket", tags=["websocket"])

    # Globalna obsługa błędów
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions and return a generic 500 response."""
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        log_context = RuntimeLogContext(session_id=request_id)
        log_event(
            logger,
            logging.ERROR,
            "http_unhandled_exception",
            "Unhandled exception in request handler.",
            log_context,
            exc_info=True,
            http_method=request.method,
            http_path=request.url.path,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
            headers={"X-Request-ID": request_id},
        )

    return app
