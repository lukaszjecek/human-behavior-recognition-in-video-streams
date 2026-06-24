"""Implementation of session REST endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from src.app.schemas.session import SessionResponse, SessionStartRequest, SessionStatus
from src.app.services.session_manager import manager
from src.inference.runtime_logging import (
    RuntimeLogContext,
    log_event,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _video_source_label(request_body: SessionStartRequest) -> str:
    """Return a log-safe label for the requested video source."""
    if request_body.video_id is not None:
        return f"uploaded:{request_body.video_id}"
    return str(request_body.video_path)


@router.post(
    "/",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Inference Session",
)
async def start_session(
    request_body: SessionStartRequest,
    request: Request,
) -> SessionResponse:
    """Start a new asynchronous offline inference session."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    video_source = _video_source_label(request_body)
    log_event(
        logger,
        logging.INFO,
        "session_start_requested",
        f"Request to start session for video: {video_source}",
        log_context,
        video_source=video_source,
    )
    try:
        app_settings = getattr(request.app.state, "settings", None)
        response = manager.create_session(
            request_body,
            upload_dir=getattr(app_settings, "upload_dir", None),
            data_dir=getattr(app_settings, "data_dir", None),
        )
        log_event(
            logger,
            logging.INFO,
            "session_created",
            f"Successfully created session {response.id} for video: {video_source}",
            RuntimeLogContext(session_id=str(response.id)),
            video_source=video_source,
            video_path=str(request_body.video_path),
        )
        return response
    except FileNotFoundError as exc:
        log_event(
            logger,
            logging.WARNING,
            "session_start_failed",
            f"Failed to start session: {exc}",
            log_context,
            video_source=video_source,
            error_type="FileNotFoundError",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        status_code = (
            status.HTTP_409_CONFLICT
            if "already being processed" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )
        log_event(
            logger,
            logging.WARNING,
            "session_start_failed",
            f"Failed to start session: {exc}",
            log_context,
            video_source=video_source,
            error_type="ValueError",
        )
        raise HTTPException(
            status_code=status_code,
            detail=str(exc)
        )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get Session Status",
)
async def get_session(session_id: UUID, request: Request) -> SessionResponse:
    """Retrieve the current status and metadata of an inference session."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.DEBUG,
        "session_status_requested",
        f"Retrieving status for session {session_id}.",
        log_context,
        target_session_id=str(session_id),
    )
    session = manager.get_session(session_id)
    if not session:
        log_event(
            logger,
            logging.WARNING,
            "session_not_found",
            f"Session {session_id} not found.",
            log_context,
            target_session_id=str(session_id),
        )
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
async def stop_session(session_id: UUID, request: Request) -> SessionResponse:
    """Stop an ongoing inference session."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.INFO,
        "session_stop_requested",
        f"Request to stop session {session_id}.",
        log_context,
        target_session_id=str(session_id),
    )
    session = manager.get_session(session_id)
    if not session:
        log_event(
            logger,
            logging.WARNING,
            "session_stop_failed",
            f"Stop failed: session {session_id} not found.",
            log_context,
            target_session_id=str(session_id),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.status not in (SessionStatus.PENDING, SessionStatus.RUNNING):
        log_event(
            logger,
            logging.WARNING,
            "session_stop_failed",
            f"Stop failed: session {session_id} is in status {session.status.value}.",
            log_context,
            target_session_id=str(session_id),
            session_status=session.status.value,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot stop a session that is not running"
        )

    stopped_session = manager.stop_session(session_id)
    if not stopped_session:
        log_event(
            logger,
            logging.WARNING,
            "session_stop_failed",
            f"Stop failed: session {session_id} not found during stop operation.",
            log_context,
            target_session_id=str(session_id),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    log_event(
        logger,
        logging.INFO,
        "session_stopped",
        f"Successfully signaled session {session_id} to stop.",
        log_context,
        target_session_id=str(session_id),
    )
    return stopped_session
