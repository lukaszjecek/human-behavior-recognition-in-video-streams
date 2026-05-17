"""Implementation of session REST endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from src.app.schemas.session import SessionResponse, SessionStartRequest, SessionStatus
from src.app.services.session_manager import manager

router = APIRouter()


@router.post(
    "/",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Inference Session",
)
async def start_session(request: SessionStartRequest) -> SessionResponse:
    """Start a new asynchronous offline inference session."""
    try:
        return manager.create_session(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc)
        )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get Session Status",
)
async def get_session(session_id: UUID) -> SessionResponse:
    """Retrieve the current status and metadata of an inference session."""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    return session


@router.post(
    "/{session_id}/stop",
    response_model=SessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stop Inference Session",
)
async def stop_session(session_id: UUID) -> SessionResponse:
    """Stop an ongoing inference session."""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.status not in (SessionStatus.PENDING, SessionStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot stop a session that is not running"
        )

    # Calling stop_session on the manager will gracefully set the stop_event
    stopped_session = manager.stop_session(session_id)
    if not stopped_session:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    
    return stopped_session
