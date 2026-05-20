"""Runtime session lifecycle hooks for inference service."""

from enum import Enum
import logging
import threading
from typing import Optional
import uuid

from src.inference.runtime_logging import (
    RuntimeLogContext,
    configure_runtime_logging,
    log_event,
)
from src.inference.service import (
    InferenceServiceRequest,
    InferenceServiceResult,
    run_inference,
    supports_session_id,
)

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Lifecycle states of an inference session."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class InferenceSession:
    """Explicit lifecycle object for managing background inference tasks."""

    def __init__(self, request: InferenceServiceRequest) -> None:
        """Initialize an idle session with the given request."""
        if not isinstance(request, InferenceServiceRequest):
            raise TypeError(
                "request must be an InferenceServiceRequest instance")

        self._request = request
        self._session_id = uuid.uuid4().hex
        self._log_context = RuntimeLogContext(
            session_id=self._session_id,
            source_type=(
                request.source_type if isinstance(
                    request.source_type, str) else None
            ),
            source_ref=_resolve_source_ref(request),
        )
        self._status = SessionStatus.IDLE
        self._stop_event = threading.Event()
        self._result: Optional[InferenceServiceResult] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the inference session in a background thread."""
        configure_runtime_logging()
        with self._lock:
            if self._status != SessionStatus.IDLE:
                raise RuntimeError(
                    f"Cannot start session in state {self._status.name}")

            self._status = SessionStatus.RUNNING
            self._thread = threading.Thread(
                target=self._run_worker, daemon=True)
            self._thread.start()
            log_event(
                logger,
                logging.INFO,
                "session_started",
                "Inference session thread started.",
                self._log_context,
            )

    def stop(self) -> None:
        """Signal the session to stop and wait for it to clean up."""
        with self._lock:
            if self._status in (
                SessionStatus.IDLE,
                SessionStatus.FINISHED,
                SessionStatus.ERROR,
                SessionStatus.STOPPED,
            ):
                return

        self._stop_event.set()
        log_event(
            logger,
            logging.INFO,
            "session_stop_requested",
            "Stop requested for inference session.",
            self._log_context,
        )

        if self._thread is not None and self._thread.is_alive():
            # Don't join the thread if we are calling stop() from the thread itself
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=5.0)

    def status(self) -> SessionStatus:
        """Get the current lifecycle status of the session."""
        with self._lock:
            return self._status

    def result(self) -> Optional[InferenceServiceResult]:
        """Return the completed result, or None if still running or errored."""
        with self._lock:
            return self._result

    def _run_worker(self) -> None:
        """Internal thread worker to run inference."""
        try:
            if supports_session_id(run_inference):
                result = run_inference(
                    self._request,
                    stop_event=self._stop_event,
                    session_id=self._session_id,
                )
            else:
                result = run_inference(
                    self._request, stop_event=self._stop_event)
            with self._lock:
                if self._stop_event.is_set():
                    self._status = SessionStatus.STOPPED
                    log_event(
                        logger,
                        logging.INFO,
                        "session_stopped",
                        "Inference session stopped before completion.",
                        self._log_context,
                    )
                else:
                    self._result = result
                    self._status = SessionStatus.FINISHED
                    log_event(
                        logger,
                        logging.INFO,
                        "session_finished",
                        "Inference session finished successfully.",
                        self._log_context,
                    )
        except Exception:
            with self._lock:
                self._status = SessionStatus.ERROR
            log_event(
                logger,
                logging.ERROR,
                "session_failed",
                "Inference session failed with an exception.",
                self._log_context,
                exc_info=True,
            )


def _resolve_source_ref(request: InferenceServiceRequest) -> str | None:
    """Resolve a string reference for runtime logging."""
    if request.video_path is not None:
        return str(request.video_path)
    if request.source_uri is not None:
        return request.source_uri
    return None
