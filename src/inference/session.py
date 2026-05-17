"""Runtime session lifecycle hooks for inference service."""

import threading
from enum import Enum
from typing import Optional

from src.inference.service import (
    InferenceServiceRequest,
    InferenceServiceResult,
    run_inference,
)


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
        self._status = SessionStatus.IDLE
        self._stop_event = threading.Event()
        self._result: Optional[InferenceServiceResult] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the inference session in a background thread."""
        with self._lock:
            if self._status != SessionStatus.IDLE:
                raise RuntimeError(
                    f"Cannot start session in state {self._status.name}")

            self._status = SessionStatus.RUNNING
            self._thread = threading.Thread(
                target=self._run_worker, daemon=True)
            self._thread.start()

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
            result = run_inference(self._request, stop_event=self._stop_event)
            with self._lock:
                if self._stop_event.is_set():
                    self._status = SessionStatus.STOPPED
                else:
                    self._result = result
                    self._status = SessionStatus.FINISHED
        except Exception:
            with self._lock:
                self._status = SessionStatus.ERROR
