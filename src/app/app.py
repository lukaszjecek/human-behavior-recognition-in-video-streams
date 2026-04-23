"""FastAPI application factory for the backend.

Provides create_app() that wires configuration, routers and basic error handling.
"""
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Application imports (settings and routers)
from src.app.core.settings import Settings, settings as default_settings
from src.app.endpoints.health import router as health_router
from src.app.api.routes import router as api_router
from src.app.api.websocket import router as ws_router


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        settings: Optional pre-built Settings instance (useful for tests).
    Returns:
        Configured FastAPI instance with routers mounted.
    """
    app_settings = settings or default_settings

    app = FastAPI(
        title=app_settings.app_name, 
        version=app_settings.app_version, 
        debug=app_settings.debug
    )

    app.include_router(health_router)
    app.include_router(api_router, prefix="/api", tags=["rest"])
    app.include_router(ws_router, prefix="/ws", tags=["websocket"])

    # Globalna obsługa błędów
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500, 
            content={"detail": "Internal server error"}
        )

    return app