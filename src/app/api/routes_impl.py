"""Implementation of API routes."""

from fastapi import APIRouter

from src.app.endpoints.events import router as events_router
from src.app.endpoints.sessions import router as sessions_router

router = APIRouter()

@router.get("/", summary="API root")
async def api_root() -> dict[str, object]:
    """Placeholder for the API root endpoint."""
    return {"message": "API root"}

router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
router.include_router(events_router, prefix="/events", tags=["events"])