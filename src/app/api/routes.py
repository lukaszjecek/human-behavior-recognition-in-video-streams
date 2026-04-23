"""Main REST API router wiring."""

from fastapi import APIRouter
from src.app.api.routes_impl import router as placeholder_router

router = APIRouter()

# Podłączamy konkretne implementacje tras
router.include_router(placeholder_router, tags=["placeholder"])