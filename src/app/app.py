"""FastAPI application factory for the backend.

Provides create_app() that wires configuration, routers and basic error handling.
"""
import logging
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

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
    configure_runtime_logging()
    app_settings = settings or default_settings
    build_metadata = get_build_metadata()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        debug=app_settings.debug,
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
        request_id = getattr(request.state, "request_id", None)
        log_context = RuntimeLogContext(
            session_id=request_id or uuid.uuid4().hex)
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
        )

    return app
