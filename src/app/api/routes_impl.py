"""Implementation of API routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="API root")
async def api_root():
    """Placeholder for the API root endpoint."""
    return {"message": "API root"}