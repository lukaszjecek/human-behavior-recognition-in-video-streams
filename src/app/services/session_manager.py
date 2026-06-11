"""Service layer for managing inference sessions."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

from src.app.db.repository import save_event
from src.app.db.session import SessionLocal
from src.app.schemas.action_event import EventPayload
from src.app.schemas.session import SessionResponse, SessionStartRequest, SessionStatus
from src.app.services.websocket_manager import websocket_manager
from src.inference.runtime_logging import (
    RuntimeLogContext,
    log_audit_event,
    log_event,
)
from src.inference.service import InferenceServiceRequest, run_offline_mp4_inference

logger = logging.getLogger(__name__)


def _resolve_video_path(path: Path) -> Path:
    """Resolve video path by checking if it exists directly or searching subdirectories."""
    if path.is_file():
        return path

    # Check recursively in data/raw or data folders inside container
    filename = path.name
    search_roots = [Path("data/raw"), Path("data"), Path("/app/data/raw"), Path("/app/data")]
    for root in search_roots:
        if root.is_dir():
            for found_path in root.rglob(filename):
                if found_path.is_file():
                    return found_path.resolve()

    return path


class SessionData:
    """Internal model for tracking a session."""

    def __init__(self, request: SessionStartRequest) -> None:
        """Initialize session data with the original request."""
        self.id = uuid4()
        self.status = SessionStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.error: str | None = None
        self.request = request
        self.stop_event = Event()
        self.task: asyncio.Task | None = None

    def update_status(self, status: SessionStatus, error: str | None = None) -> None:
        """Update session status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)
        if error is not None:
            self.error = error

    def to_response(self) -> SessionResponse:
        """Convert to external response schema."""
        return SessionResponse(
            id=self.id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            error=self.error,
        )


class InferenceSessionManager:
    """Manages lifecycle of inference sessions."""

    def __init__(self) -> None:
        """Initialize the inference session manager."""
        self._sessions: dict[UUID, SessionData] = {}

    def create_session(self, request: SessionStartRequest) -> SessionResponse:
        """Create and start a new inference session."""
        # Resolve the video path recursively if it doesn't exist at the given path
        resolved_path = _resolve_video_path(request.video_path)
        if not resolved_path.is_file():
            raise ValueError(f"Video file not found: {request.video_path}")
        request.video_path = resolved_path

        # Check for duplicates
        for existing_session in self._sessions.values():
            if existing_session.status in (SessionStatus.PENDING, SessionStatus.RUNNING):
                if existing_session.request.video_path == request.video_path:
                    raise ValueError(
                        f"Video {request.video_path} is already being "
                        f"processed in session {existing_session.id}"
                    )

        session = SessionData(request)
        self._sessions[session.id] = session

        session.update_status(SessionStatus.RUNNING)
        session.task = asyncio.create_task(self._run_session_task(session))

        return session.to_response()

    def get_session(self, session_id: UUID) -> SessionResponse | None:
        """Retrieve a session by its ID."""
        session = self._sessions.get(session_id)
        return session.to_response() if session else None

    def stop_session(self, session_id: UUID) -> SessionResponse | None:
        """Attempt to stop a running session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        if session.status in (SessionStatus.PENDING, SessionStatus.RUNNING):
            session.stop_event.set()
            session.update_status(SessionStatus.STOPPED)
            # The running task will naturally exit on the next frame due to stop_event
            # being checked inside the inference loop.

        return session.to_response()

    async def _run_session_task(self, session: SessionData) -> None:
        """Background task that runs the blocking inference."""
        log_event(
            logger,
            logging.INFO,
            "session_task_started",
            f"Background inference task started for session {session.id}.",
            RuntimeLogContext(session_id=str(session.id)),
            video_path=str(session.request.video_path),
        )
        try:
            # Map API request to internal request
            inference_request = InferenceServiceRequest(
                video_path=session.request.video_path,
                checkpoint_path=session.request.checkpoint_path,
                config_path=session.request.config_path,
                device=session.request.device,
            )

            def on_event(payload: EventPayload) -> None:
                # 1. Broadcast to websocket clients
                websocket_manager.broadcast_sync(payload)
                # 2. Persist to database
                try:
                    with SessionLocal() as db:
                        save_event(db, payload)
                except Exception as db_err:
                    log_event(
                        logger,
                        logging.ERROR,
                        "database_write_failed",
                        f"Database write-path failure for event {payload.event_id} "
                        f"in background session {session.id}: {db_err}",
                        RuntimeLogContext(session_id=str(session.id)),
                        exc_info=True,
                        event_id=str(payload.event_id),
                        error_type=type(db_err).__name__,
                    )
                    logger.error(
                        "Database write-path failure for event %s in background session %s: %s",
                        payload.event_id,
                        session.id,
                        db_err,
                    )

                # 3. Write to file-based audit trail
                try:
                    log_audit_event(payload)
                except Exception as audit_err:
                    log_event(
                        logger,
                        logging.ERROR,
                        "audit_file_write_failed",
                        f"Failed to write event {payload.event_id} to audit log: {audit_err}",
                        RuntimeLogContext(session_id=str(session.id)),
                        exc_info=True,
                        event_id=str(payload.event_id),
                        error_type=type(audit_err).__name__,
                    )

                # 4. Log structured audit event
                is_detection = payload.event_type.value == "DETECTION"
                event_name = (
                    "audit_detection_published"
                    if is_detection
                    else "audit_alert_triggered"
                )
                log_event(
                    logger,
                    logging.INFO,
                    event_name,
                    f"Published {payload.event_type.value} event {payload.event_id} "
                    f"for session {session.id}.",
                    RuntimeLogContext(session_id=str(session.id)),
                    event_id=str(payload.event_id),
                    camera_id=payload.camera_id,
                    event_type=payload.event_type.value,
                )

            # Run blocking call in a background thread
            await asyncio.to_thread(
                run_offline_mp4_inference,
                inference_request,
                session.stop_event,
                on_event,
                str(session.id),
            )

            # Only update to COMPLETED if not stopped manually
            if session.status == SessionStatus.RUNNING:
                session.update_status(SessionStatus.COMPLETED)
                log_event(
                    logger,
                    logging.INFO,
                    "session_task_completed",
                    f"Background inference task completed successfully for session {session.id}.",
                    RuntimeLogContext(session_id=str(session.id)),
                    video_path=str(session.request.video_path),
                )

        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "session_task_failed",
                f"Background inference task failed for session {session.id}: {exc}",
                RuntimeLogContext(session_id=str(session.id)),
                exc_info=True,
                video_path=str(session.request.video_path),
                error_type=type(exc).__name__,
            )
            logger.exception("Session %s execution failed", session.id)
            import sys
            import traceback

            print(
                f"ERROR: Session {session.id} execution failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            # If the session wasn't explicitly stopped, mark as FAILED
            if session.status != SessionStatus.STOPPED:
                session.update_status(SessionStatus.FAILED, str(exc))


# Global instance to be used by endpoints
manager = InferenceSessionManager()
