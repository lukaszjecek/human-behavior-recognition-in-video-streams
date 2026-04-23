"""Implementation of API routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="API root")
async def api_root() -> dict[str, object]:
    """Placeholder for the API root endpoint."""
    return {"message": "API root"}