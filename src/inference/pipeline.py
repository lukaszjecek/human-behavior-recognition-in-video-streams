"""Shared frame-to-event runtime pipeline for MP4, RTSP, and browser-camera sources.

Implements a push-based, stream-oriented processing pipeline that accepts raw
video frames one at a time and emits canonical ``EventPayload`` objects whenever
a temporal inference window fires.  The pipeline is source-agnostic: it does not
care whether the frame came from a file, an RTSP camera, or a browser WebSocket.

Conceptual flow::

    push_frame(frame)
        → InferenceEngine.process_frame()       # sliding window / stride logic
        → InferenceResult → ActionEventWriter   # logit → label/confidence
        → _maybe_enrich_context(window)         # ContextModule every N windows
        → bbox_hook(action_event)               # optional, issue #119
        → ContextAwareAlertProcessor.process()  # alert state machine
        → EventPayload(DETECTION)
        → EventPayload(ALERT)  # when alert fires
        → return list[EventPayload]
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4

import numpy as np

from src.app.schemas.action_event import (
    ActionEvent,
    AlertData,
    ContextData,
    EventPayload,
    EventType,
)
from src.inference.alert_state_machine import AlertRaisedEvent
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter

logger = logging.getLogger(__name__)



# Sentinel returned by ContextModule.get_context() when context is unavailable.
_UNKNOWN_CONTEXT: ContextData | None = None

# Type alias for the optional bbox enrichment hook supplied by issue #119.
BBoxHook = Callable[[ActionEvent], ActionEvent]

# Protocols for injected components.
class AlertProcessor(Protocol):
    def process_event(self, event: ActionEvent) -> AlertRaisedEvent | None: ...
    def reset_all(self) -> None: ...

class ContextProvider(Protocol):
    def get_context(self, frame: Any) -> dict: ...  # noqa: ANN401


class InferenceEventPipeline:
    """Shared, push-based frame-to-event runtime for all inference sources.

    Callers push raw BGR ``numpy`` frames one at a time via :meth:`push_frame`.
    The pipeline internally manages the sliding inference window, context
    enrichment, optional bounding-box hook, and alert state transitions, then
    returns zero or more :class:`~src.app.schemas.action_event.EventPayload`
    objects per call.

    This design deliberately avoids coupling to a frame *source*.  The pipeline
    works identically whether the caller loops over an MP4 file, reads from an
    RTSP connection, or receives frames over a browser WebSocket.

    Thread safety
    -------------
    :meth:`push_frame` acquires ``self._lock`` before every call.  The
    underlying :class:`~src.inference.engine.InferenceEngine` has its own
    internal lock; the pipeline lock serialises access to all *other* mutable
    state (writer, alert processor, context cache, window counter).  Concurrent
    callers are therefore safe to share a single pipeline instance.

    Context evaluation cadence
    --------------------------
    Running the context model (MobileNetV2) on every inference window may be
    too expensive for high-throughput RTSP streams.  ``context_eval_every_n_windows``
    controls how often the context module is queried:

    - ``1`` — evaluated on every inference window (most accurate).
    - ``N`` — evaluated once per *N* windows; the last known result is reused
      in between.  Scene context changes slowly (order of seconds), so
      caching is safe.

    Default is ``5`` — a good balance for 25 fps / stride-8 deployments
    (≈ one context update every 2.5 s).

    Bounding-box enrichment hook
    ----------------------------
    Pass a callable as ``bbox_hook`` to attach bounding boxes before an event
    is forwarded to the alert processor.  The hook receives the :class:`ActionEvent`
    (with context already attached) and must return an :class:`ActionEvent`
    (usually the same object with ``bboxes`` populated).  Issue #119 will
    implement a concrete provider; this parameter is the stable integration point.

    Parameters
    ----------
    engine:
        Configured :class:`~src.inference.engine.InferenceEngine` instance
        (window size, stride, and model adapter already set).
    writer:
        :class:`~src.inference.json_writer.ActionEventWriter` for converting
        ``InferenceResult`` objects into ``ActionEvent`` records.
    alert_processor:
        Either an :class:`~src.inference.alert_state_machine.AlertStateMachine`
        or a :class:`~src.inference.context_policy.ContextAwareAlertProcessor`.
        The pipeline calls ``process_event(event)`` on both.
    context_module:
        Optional object exposing ``get_context(frame) -> dict``.  When *None*
        the pipeline always falls back to ``scene_tag="unknown"``.
    context_eval_every_n_windows:
        How many inference windows to skip between context evaluations.
        Must be >= 1.  Defaults to ``5``.
    bbox_hook:
        Optional callable ``(ActionEvent) -> ActionEvent`` invoked after context
        enrichment and before alert processing.  Intended for issue #119.
    camera_id:
        Forwarded verbatim into every emitted ``EventPayload.camera_id``.
    session_id:
        Forwarded verbatim into every emitted ``EventPayload.session_id``.
        A random UUID is generated when *None*.
    track_id:
        Default track identifier attached to every ``ActionEvent``.  Mirrors
        ``InferenceRuntimeSettings.default_track_id``.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        writer: ActionEventWriter,
        alert_processor: AlertProcessor,
        *,
        context_module: ContextProvider | None = None,
        context_eval_every_n_windows: int = 5,
        bbox_hook: BBoxHook | None = None,
        camera_id: str | None = None,
        session_id: UUID | None = None,
        track_id: int | None = None,
    ) -> None:
        """Initialise the pipeline with injected components."""
        if not isinstance(engine, InferenceEngine):
            raise TypeError("engine must be an InferenceEngine instance")
        if not isinstance(writer, ActionEventWriter):
            raise TypeError("writer must be an ActionEventWriter instance")
        if (
            not isinstance(context_eval_every_n_windows, int)
            or isinstance(context_eval_every_n_windows, bool)
        ):
            raise TypeError("context_eval_every_n_windows must be an integer")
        if context_eval_every_n_windows < 1:
            raise ValueError("context_eval_every_n_windows must be >= 1")
        if bbox_hook is not None and not callable(bbox_hook):
            raise TypeError("bbox_hook must be callable or None")
        if camera_id is not None and not isinstance(camera_id, str):
            raise TypeError("camera_id must be a string or None")
        if session_id is not None and not isinstance(session_id, UUID):
            raise TypeError("session_id must be a UUID or None")
        if track_id is not None and (
            not isinstance(track_id, int) or isinstance(track_id, bool)
        ):
            raise TypeError("track_id must be an integer or None")
        if isinstance(track_id, int) and track_id < 0:
            raise ValueError("track_id must be >= 0")

        self._engine = engine
        self._writer = writer
        self._alert_processor = alert_processor
        self._context_module = context_module
        self._context_eval_every_n_windows = context_eval_every_n_windows
        self._bbox_hook = bbox_hook
        self._camera_id = camera_id
        self._session_id = session_id if session_id is not None else uuid4()
        self._track_id = track_id

        # Mutable pipeline state — protected by _lock.
        self._lock = threading.RLock()
        self._windows_since_context: int = 0  # counts up to next evaluation
        self._cached_context: ContextData | None = _UNKNOWN_CONTEXT
        self._total_windows_processed: int = 0

        logger.debug(
            "InferenceEventPipeline initialised "
            "(window=%d stride=%d context_every_n=%d camera_id=%r session_id=%s)",
            engine.window_size,
            engine.stride,
            context_eval_every_n_windows,
            camera_id,
            self._session_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> list[EventPayload]:
        """Push one raw BGR frame through the pipeline.

        This is the primary streaming API.  Call it once per decoded frame,
        regardless of the frame source.  Returns an empty list for most frames
        (while the sliding window fills up or stride keeps inference silent);
        returns one or more :class:`EventPayload` objects whenever an inference
        window fires and/or an alert transitions to ACTIVE.

        The call is thread-safe: a single :class:`InferenceEventPipeline`
        instance may be shared across threads.

        Parameters
        ----------
        frame:
            Raw BGR ``numpy`` array of shape ``(H, W, 3)`` and dtype
            ``np.uint8``, exactly as delivered by ``cv2.VideoCapture.read()``,
            a WebSocket payload decoder, or any equivalent source.
        timestamp:
            Optional wall-clock timestamp (seconds since epoch) for this frame.
            When *None* the engine records the current time automatically.

        Returns:
        -------
        list[EventPayload]
            Zero or more events emitted by this frame.  The list contains at
            most one ``DETECTION`` and one ``ALERT`` per inference window.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        if frame.dtype != np.uint8:
            raise TypeError("frame must have dtype uint8")

        result = self._engine.process_frame(frame, timestamp=timestamp)
        if result is None:
            return []

        with self._lock:
            return self._process_result(result)

    def reset(self) -> None:
        """Reset all pipeline state, including the engine, context cache, and alert processor.

        Use this when switching sources mid-session (e.g. camera re-connect)
        without destroying the pipeline object.
        """
        with self._lock:
            self._engine.reset()
            self._windows_since_context = 0
            self._cached_context = _UNKNOWN_CONTEXT
            self._total_windows_processed = 0
            self._alert_processor.reset_all()
            logger.info(
                "InferenceEventPipeline reset (session_id=%s)", self._session_id
            )

    def get_metrics(self) -> dict:
        """Return a snapshot of pipeline-level counters.

        Returns a plain ``dict`` suitable for logging or health-check endpoints.
        """
        with self._lock:
            engine_metrics = self._engine.get_metrics()
            return {
                **engine_metrics,
                "total_windows_processed": self._total_windows_processed,
                "cached_context_scene_tag": (
                    self._cached_context.scene_tag if self._cached_context else None
                ),
                "cached_context_confidence": (
                    self._cached_context.confidence if self._cached_context else None
                ),
                "context_eval_every_n_windows": self._context_eval_every_n_windows,
            }

    # ------------------------------------------------------------------
    # Internal helpers (not part of the public contract)
    # ------------------------------------------------------------------

    def _process_result(self, result: InferenceResult) -> list[EventPayload]:
        """Convert one ``InferenceResult`` into zero or more ``EventPayload`` objects.

        Called while ``self._lock`` is held.  Steps:

        1. Convert ``InferenceResult`` → ``ActionEvent`` via ``ActionEventWriter``.
        2. Enrich with context (conditionally, every N windows).
        3. Run optional bbox hook.
        4. Forward to alert processor → emit DETECTION + optional ALERT.

        Parameters
        ----------
        result:
            Fresh ``InferenceResult`` from the inference engine.

        Returns:
        -------
        list[EventPayload]
            DETECTION payload always included when ``ActionEvent`` is valid;
            ALERT payload appended when the alert state machine fires.
        """
        self._total_windows_processed += 1

        action_event = self._writer.process_inference_result(
            result, track_id=self._track_id
        )
        if action_event is None:
            # Prediction was None or unparseable — skip silently.
            logger.debug(
                "InferenceEventPipeline: window %d produced no ActionEvent (None prediction)",
                self._total_windows_processed,
            )
            return []

        # ---- Context enrichment ----
        action_event = self._enrich_context(action_event, result)

        # ---- BBox hook (integration point for issue #119) ----
        action_event = self._run_bbox_hook(action_event)

        # Save to writer's log for session accumulation
        self._writer.get_log().add_event(action_event)

        # ---- Emit DETECTION EventPayload ----
        detection_payload = EventPayload(
            event_type=EventType.DETECTION,
            data=action_event,
            camera_id=self._camera_id,
            session_id=self._session_id,
        )
        payloads: list[EventPayload] = [detection_payload]

        logger.debug(
            "InferenceEventPipeline: DETECTION emitted "
            "(label=%r confidence=%.3f frames=%d-%d session=%s)",
            action_event.label,
            action_event.confidence,
            result.start_frame_index,
            result.end_frame_index,
            self._session_id,
        )

        # ---- Alert processing ----
        alert_raised = self._run_alert_processor(action_event)
        if alert_raised is not None:
            alert_data = AlertData(
                severity="HIGH",
                message=f"Alert triggered for label: {alert_raised.label}",
                action_event=alert_raised.triggering_event,
            )
            alert_payload = EventPayload(
                event_type=EventType.ALERT,
                data=alert_data,
                camera_id=self._camera_id,
                session_id=self._session_id,
            )
            payloads.append(alert_payload)
            logger.info(
                "InferenceEventPipeline: ALERT emitted "
                "(label=%r track_id=%r hits=%d session=%s)",
                alert_raised.label,
                alert_raised.track_id,
                alert_raised.consecutive_hits,
                self._session_id,
            )

        return payloads

    def _enrich_context(
        self,
        event: ActionEvent,
        result: InferenceResult,
    ) -> ActionEvent:
        """Attach context data to *event*, evaluating the context module when due.

        Context is re-evaluated every ``_context_eval_every_n_windows`` windows.
        Between evaluations the cached result is reused.  Any exception from the
        context module is caught and results in ``scene_tag="unknown"`` with a
        WARNING log — inference is never interrupted by a context failure.

        Uses the *first* frame of the inference window as the context input,
        matching the existing ``mp4_cli.py`` convention.

        Parameters
        ----------
        event:
            ``ActionEvent`` to enrich (returned as a new object).
        result:
            ``InferenceResult`` whose ``window`` provides raw frames for the
            context module.

        Returns:
        -------
        ActionEvent
            New ``ActionEvent`` with ``context`` field populated.
        """
        self._windows_since_context += 1

        if self._windows_since_context >= self._context_eval_every_n_windows:
            self._windows_since_context = 0
            self._cached_context = self._evaluate_context(result)

        return event.model_copy(update={"context": self._cached_context})

    def _evaluate_context(self, result: InferenceResult) -> ContextData | None:
        """Query the context module for the current inference window.

        Falls back to ``_UNKNOWN_CONTEXT`` when:

        - ``_context_module`` is *None* (not configured),
        - the context module raises any exception (missing weights, missing
          torchvision, shape mismatch, etc.).

        Parameters
        ----------
        result:
            ``InferenceResult`` from which the representative frame is taken.

        Returns:
        -------
        ContextData | None
            Enriched context or the sentinel ``_UNKNOWN_CONTEXT``.
        """
        if self._context_module is None:
            return _UNKNOWN_CONTEXT

        try:
            # Use the first frame of the window as context representative.
            # Frames are raw numpy BGR arrays stored by InferenceEngine.
            representative_frame = result.window[0]

            # ContextModule.get_context() expects a PIL Image.
            # Convert BGR numpy array → PIL Image here so that ContextModule
            # stays unchanged and callers who pass PIL images directly still work.
            if isinstance(representative_frame, np.ndarray):
                try:
                    from PIL import Image as PilImage
                except ImportError:
                    raise RuntimeError("PIL not installed")
                rgb_frame = representative_frame[:, :, ::-1]  # BGR → RGB
                pil_image = PilImage.fromarray(rgb_frame)
            else:
                # Assume it is already a PIL Image (legacy callers).
                pil_image = representative_frame

            raw = self._context_module.get_context(pil_image)

            scene_tag = str(raw.get("scene_tag", "unknown"))
            if scene_tag == "unknown":
                return _UNKNOWN_CONTEXT

            confidence = float(raw.get("confidence", 0.0))
            context = ContextData(scene_tag=scene_tag, confidence=confidence)

            logger.debug(
                "InferenceEventPipeline: context evaluated "
                "(scene_tag=%r confidence=%.3f)", scene_tag, confidence
            )
            return context

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "InferenceEventPipeline: context evaluation failed, "
                "using fallback (scene_tag='unknown'): %s",
                exc,
            )
            return _UNKNOWN_CONTEXT

    def _run_bbox_hook(self, event: ActionEvent) -> ActionEvent:
        """Run the optional bounding-box enrichment hook.

        This is the stable integration point for issue #119.  When no hook is
        configured the event is returned unchanged.

        Parameters
        ----------
        event:
            ``ActionEvent`` to enrich with bounding boxes.

        Returns:
        -------
        ActionEvent
            The event returned by the hook, or the original event on failure.
        """
        if self._bbox_hook is None:
            return event

        try:
            enriched = self._bbox_hook(event)
            if not isinstance(enriched, ActionEvent):
                logger.warning(
                    "InferenceEventPipeline: bbox_hook returned %r instead of ActionEvent; "
                    "ignoring hook result",
                    type(enriched).__name__,
                )
                return event
            return enriched
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "InferenceEventPipeline: bbox_hook raised an exception, "
                "proceeding without bbox enrichment: %s",
                exc,
            )
            return event

    def _run_alert_processor(self, event: ActionEvent) -> AlertRaisedEvent | None:
        """Forward *event* to whichever alert processor was injected.

        Supports any processor adhering to the ``AlertProcessor`` protocol.

        Parameters
        ----------
        event:
            ``ActionEvent`` to forward.

        Returns:
        -------
        AlertRaisedEvent | None
            Alert event when a track transitions to ACTIVE, ``None`` otherwise.
        """
        try:
            return self._alert_processor.process_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "InferenceEventPipeline: alert processor raised an exception: %s",
                exc,
                exc_info=True,
            )
            return None


__all__ = ["InferenceEventPipeline"]
