"""Service layer for managing inference sessions."""

import asyncio
from datetime import datetime, timezone
from threading import Event
from uuid import UUID, uuid4

from src.app.schemas.session import SessionResponse, SessionStartRequest, SessionStatus
from src.app.services.websocket_manager import websocket_manager
from src.inference.service import InferenceServiceRequest, run_offline_mp4_inference


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
        try:
            # Map API request to internal request
            inference_request = InferenceServiceRequest(
                video_path=session.request.video_path,
                checkpoint_path=session.request.checkpoint_path,
                config_path=session.request.config_path,
                device=session.request.device,
            )

            # Run blocking call in a background thread
            await asyncio.to_thread(
                run_offline_mp4_inference,
                inference_request,
                session.stop_event,
                websocket_manager.broadcast_sync,
            )

            # Only update to COMPLETED if not stopped manually
            if session.status == SessionStatus.RUNNING:
                session.update_status(SessionStatus.COMPLETED)

        except Exception as exc:
            # If the session wasn't explicitly stopped, mark as FAILED
            if session.status != SessionStatus.STOPPED:
                session.update_status(SessionStatus.FAILED, str(exc))


# Global instance to be used by endpoints
manager = InferenceSessionManager()
