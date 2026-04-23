"""Main REST API router wiring."""
from fastapi import APIRouter

# Musisz importować z routes_impl, nie z routes!
from src.app.api.routes_impl import router as placeholder_router

router = APIRouter()
router.include_router(placeholder_router)