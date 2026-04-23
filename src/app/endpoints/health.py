"""Health and readiness endpoints for the backend (implementation).

This module provides simple liveness and readiness probes under /health and /readiness.
"""
from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health check")
async def health() -> dict[str, object]:
    """Simple liveness/health endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get("/readiness", summary="Readiness probe")
async def readiness() -> dict[str, object]:
    """Readiness endpoint - placeholder for future checks (DB, queues, etc.)."""
    return {"ready": True, "status": "ok"}