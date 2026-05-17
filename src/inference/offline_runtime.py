"""Offline producer-consumer runtime for adapter-based inference inputs."""
import logging
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Optional

from src.inference.engine import InferenceEngine
from src.inference.json_writer import ActionEventWriter
from src.inference.source_adapters import FileSourceAdapter, InferenceSourceAdapter
from src.inference.tracker import BaseTracker, SingleTrackTracker

logger = logging.getLogger(__name__)

EOF_SENTINEL = object()

# ---------------------------------------------------------------------------
# Public exception and failure-state types
# ---------------------------------------------------------------------------

_RTSP_SOURCE_TYPES = {"rtsp"}


class SourceInterruptedError(RuntimeError):
    """Raised when an active source connection is lost mid-stream.

    Attributes:
        source_ref: The URI / path of the interrupted source.
        frames_read: Number of frames successfully read before interruption.
    """

    def __init__(self, source_ref: str, frames_read: int = 0) -> None:
        """Initialise with source reference and frames-read count."""
        super().__init__(
            f"Source interrupted after {frames_read} frame(s): {source_ref}"
        )
        self.source_ref = source_ref
        self.frames_read = frames_read


class RuntimeFailureState:
    """Carries structured failure information from a failed runtime session.

    This is the controlled form in which lower-level errors are surfaced to
    higher layers (e.g. FastAPI backend) instead of letting raw exceptions
    propagate unchecked.

    Attributes:
        error: The original exception that caused the failure.
        phase: Which phase failed – ``"producer"``, ``"consumer"``, or
            ``"unknown"``.
        frames_before_failure: Frames processed successfully before failure.
    """

    def __init__(
        self,
        error: BaseException,
        phase: str = "unknown",
        frames_before_failure: int = 0,
    ) -> None:
        """Initialise with error, phase, and frame count."""
        if phase not in {"producer", "consumer", "unknown"}:
            raise ValueError("phase must be 'producer', 'consumer', or 'unknown'")
        self.error = error
        self.phase = phase
        self.frames_before_failure = frames_before_failure

    def __repr__(self) -> str:
        """Return developer-readable representation."""
        return (
            f"RuntimeFailureState(phase={self.phase!r}, "
            f"frames_before_failure={self.frames_before_failure}, "
            f"error={self.error!r})"
        )


# ---------------------------------------------------------------------------
# Producer helpers
# ---------------------------------------------------------------------------


def produce_frames_from_source(
    source_adapter: InferenceSourceAdapter,
    frame_queue: Queue,
    stop_event: Optional[Event] = None,
) -> None:
    """Read frames from a source adapter and push them to a queue.

    Stops early when *stop_event* is set.  The EOF sentinel is always pushed
    to the queue so the consumer can terminate cleanly.

    Args:
        source_adapter (InferenceSourceAdapter): Source adapter to open and
            read.
        frame_queue (Queue): Queue used to pass frames to the consumer.
        stop_event (Optional[Event]): When set, the producer stops reading
            new frames and exits.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    try:
        cap = source_adapter.open_capture()

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open "
                f"{source_adapter.source_type} source: {source_adapter.source_ref}",
            )

        frames_read = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.debug(
                        "Producer stopping early on stop_event (read %d frames).",
                        frames_read,
                    )
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frames_read += 1
                frame_queue.put(frame)
        finally:
            cap.release()
    finally:
        frame_queue.put(EOF_SENTINEL)


def produce_frames(video_path: str, frame_queue: Queue) -> None:
    """Backward-compatible file-source producer."""
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string")

    source_adapter = FileSourceAdapter(video_path=Path(video_path))
    produce_frames_from_source(source_adapter, frame_queue)


def produce_frames_safe(
    source_adapter: InferenceSourceAdapter,
    frame_queue: Queue,
    stats: dict,
    stop_event: Optional[Event] = None,
) -> None:
    """Run the frame producer and store any raised exception in stats."""
    try:
        produce_frames_from_source(source_adapter, frame_queue, stop_event=stop_event)
    except Exception as exc:
        stats["producer_error"] = exc


# ---------------------------------------------------------------------------
# RTSP reconnect producer
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAY = 2.0  # seconds
_DEFAULT_BACKOFF_FACTOR = 2.0


def produce_frames_with_reconnect(
    source_adapter: InferenceSourceAdapter,
    frame_queue: Queue,
    stats: dict,
    stop_event: Optional[Event] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
) -> None:
    """Frame producer with exponential back-off reconnect for stream sources.

    On a ``cap.read()`` failure the producer waits *retry_delay* seconds (
    doubling each attempt up to *max_retries*) before re-opening the source.
    For non-stream sources (``source_type != 'rtsp'``) it falls back to
    the standard single-attempt producer.

    Args:
        source_adapter (InferenceSourceAdapter): Source adapter to open and
            read.
        frame_queue (Queue): Queue used to pass frames to the consumer.
        stats (dict): Mutable dict; ``producer_error`` is set on terminal
            failure.
        stop_event (Optional[Event]): When set the producer exits immediately.
        max_retries (int): Maximum reconnect attempts before giving up.
        retry_delay (float): Initial seconds to wait before the first retry.
        backoff_factor (float): Multiplier applied to *retry_delay* on each
            successive attempt.
    """
    if source_adapter.source_type not in _RTSP_SOURCE_TYPES:
        # Non-stream sources do not need reconnect logic.
        produce_frames_safe(source_adapter, frame_queue, stats, stop_event=stop_event)
        return

    frames_read = 0
    attempts = 0
    current_delay = retry_delay

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                logger.debug("Reconnect producer: stop_event set before attempt.")
                break

            try:
                cap = source_adapter.open_capture()
            except Exception as open_exc:
                logger.warning(
                    "Reconnect producer: open_capture() raised: %s", open_exc
                )
                cap = None

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()

                if attempts >= max_retries:
                    err = SourceInterruptedError(
                        source_ref=source_adapter.source_ref,
                        frames_read=frames_read,
                    )
                    logger.error(
                        "Reconnect producer: max retries (%d) reached for %s.",
                        max_retries,
                        source_adapter.source_ref,
                    )
                    stats["producer_error"] = err
                    return

                logger.warning(
                    "Reconnect producer: could not open source %s "
                    "(attempt %d/%d), retrying in %.1fs.",
                    source_adapter.source_ref,
                    attempts + 1,
                    max_retries,
                    current_delay,
                )
                _interruptible_sleep(current_delay, stop_event)
                current_delay *= backoff_factor
                attempts += 1
                continue

            # Successfully opened - reset retry counter for this connection.
            attempts = 0
            current_delay = retry_delay
            stop_requested = False
            connection_dropped = False

            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        logger.debug(
                            "Reconnect producer: stop_event set (read %d frames).",
                            frames_read,
                        )
                        stop_requested = True
                        break

                    ret, frame = cap.read()
                    if not ret:
                        # Real read failure - reconnect logic will handle it.
                        connection_dropped = True
                        logger.warning(
                            "Reconnect producer: read() returned False for %s "
                            "(frames read so far: %d). Will retry.",
                            source_adapter.source_ref,
                            frames_read,
                        )
                        break

                    frames_read += 1
                    frame_queue.put(frame)
            finally:
                cap.release()

            if stop_requested:
                # Clean shutdown requested - do not reconnect.
                break

            if not connection_dropped:
                # Source ended cleanly (stream finished).
                break

            # Real connection drop - attempt reconnect.
            if attempts >= max_retries:
                err = SourceInterruptedError(
                    source_ref=source_adapter.source_ref,
                    frames_read=frames_read,
                )
                logger.error(
                    "Reconnect producer: max retries (%d) exhausted for %s.",
                    max_retries,
                    source_adapter.source_ref,
                )
                stats["producer_error"] = err
                return

            logger.warning(
                "Reconnect producer: scheduling reconnect for %s in %.1fs "
                "(attempt %d/%d).",
                source_adapter.source_ref,
                current_delay,
                attempts + 1,
                max_retries,
            )
            _interruptible_sleep(current_delay, stop_event)
            current_delay *= backoff_factor
            attempts += 1

    finally:
        frame_queue.put(EOF_SENTINEL)


def _interruptible_sleep(duration: float, stop_event: Optional[Event]) -> None:
    """Sleep for *duration* seconds, waking early if *stop_event* is set."""
    if stop_event is None:
        time.sleep(duration)
        return
    stop_event.wait(timeout=duration)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


def consume_frame_queue(
    frame_queue: Queue,
    engine: InferenceEngine,
    stats: dict,
    stop_event: Optional[Event] = None,
) -> None:
    """Consume frames from a queue with an inference engine and update stats.

    Args:
        frame_queue (Queue): Queue providing video frames.
        engine (InferenceEngine): Engine used to process frames.
        stats (dict): Mutable stats dictionary with frame and inference counts.
        stop_event (Optional[Event]): When set the consumer discards remaining
            queued frames and terminates as soon as the current item is
            drained.
    """
    frame_count = 0
    inference_results = []

    while True:
        if stop_event is not None and stop_event.is_set():
            # Drain the queue until the sentinel arrives.
            while True:
                try:
                    # get_nowait() would exit early on a momentary Empty
                    # and leave the sentinel unconsumed, blocking 
                    # the producer on a bounded Queue.
                    item = frame_queue.get(timeout=0.05) 
                except Empty:
                    continue
                if item is EOF_SENTINEL:
                    break
            break

        try:
            frame = frame_queue.get(timeout=0.05)
        except Empty:
            continue

        if frame is EOF_SENTINEL:
            break

        frame_count += 1
        result = engine.process_frame(frame)

        if result is not None:
            inference_results.append(result)

    stats["frame_count"] = frame_count
    stats["inference_count"] = len(inference_results)
    stats["inference_results"] = inference_results


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------


def run_source(
    source_adapter: InferenceSourceAdapter,
    engine: Optional[InferenceEngine] = None,
    tracker: Optional[BaseTracker] = None,
    emit_runtime_summary: bool = True,
    stop_event: Optional[Event] = None,
) -> tuple[int, int, list[Any], list[Any]]:
    """Run offline inference on a generic source adapter.

    Args:
        source_adapter: Adapter that provides inference input frames.
        engine: Optional inference engine instance. If None, a default
            InferenceEngine is created.
        tracker: Optional tracker used to assign track IDs to inference
            results.
        emit_runtime_summary: Whether to print processed frame/window/event
            stats.
        stop_event: Optional threading.Event; when set the producer and
            consumer will stop early and the session will end gracefully.

    Returns:
        tuple[int, int, list[Any], list[Any]]: Number of processed frames,
        number of inference windows, collected inference results, and output
        action events.
    """
    if not isinstance(source_adapter, InferenceSourceAdapter):
        raise TypeError("source_adapter must be an InferenceSourceAdapter instance")

    runtime_engine = engine  # engine initialization moved to mp4_cli.py
    if runtime_engine is None:
        runtime_engine = InferenceEngine()
    elif not isinstance(runtime_engine, InferenceEngine):
        raise TypeError("engine must be an InferenceEngine instance or None")

    frame_queue: Queue = Queue()
    stats: dict = {
        "frame_count": 0,
        "inference_count": 0,
        "inference_results": [],
        "producer_error": None,
    }

    producer = Thread(
        target=produce_frames_safe,
        args=(source_adapter, frame_queue, stats),
        kwargs={"stop_event": stop_event},
    )
    consumer = Thread(
        target=consume_frame_queue,
        args=(frame_queue, runtime_engine, stats),
        kwargs={"stop_event": stop_event},
    )

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    if stats["producer_error"] is not None:
        raise stats["producer_error"]

    frame_count = stats["frame_count"]
    inference_count = stats["inference_count"]
    inference_results = stats["inference_results"]

    tracker = tracker or SingleTrackTracker()
    track_ids = tracker.assign_track_ids(inference_results)

    writer = ActionEventWriter()
    writer.add_results(inference_results, track_ids=track_ids)
    action_events = writer.get_log().events

    if emit_runtime_summary:
        print(f"Processed {frame_count} frames")
        print(f"Generated {inference_count} inference windows")
        print(f"Generated {len(action_events)} action events")

    return frame_count, inference_count, inference_results, action_events


def run_source_with_reconnect(
    source_adapter: InferenceSourceAdapter,
    engine: Optional[InferenceEngine] = None,
    tracker: Optional[BaseTracker] = None,
    emit_runtime_summary: bool = True,
    stop_event: Optional[Event] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
) -> tuple[int, int, list[Any], list[Any]]:
    """Run inference with automatic reconnect for stream (RTSP) sources.

    Behaves identically to :func:`run_source` for file sources.  For RTSP
    sources the producer attempts to reconnect on read failures using
    exponential back-off before giving up.

    Args:
        source_adapter: Adapter that provides inference input frames.
        engine: Optional inference engine instance. If None, a default
            InferenceEngine is created.
        tracker: Optional tracker used to assign track IDs to inference
            results.
        emit_runtime_summary: Whether to print processed frame/window/event
            stats.
        stop_event: When set the session ends gracefully.
        max_retries: Maximum reconnect attempts before raising
            :exc:`SourceInterruptedError`.
        retry_delay: Initial back-off delay in seconds.
        backoff_factor: Multiplier applied to *retry_delay* on each attempt.

    Returns:
        tuple[int, int, list[Any], list[Any]]: Processed frames, inference
        windows, inference results, and action events.
    """
    if not isinstance(source_adapter, InferenceSourceAdapter):
        raise TypeError("source_adapter must be an InferenceSourceAdapter instance")

    runtime_engine = engine
    if runtime_engine is None:
        runtime_engine = InferenceEngine()
    elif not isinstance(runtime_engine, InferenceEngine):
        raise TypeError("engine must be an InferenceEngine instance or None")

    frame_queue: Queue = Queue()
    stats: dict = {
        "frame_count": 0,
        "inference_count": 0,
        "inference_results": [],
        "producer_error": None,
    }

    producer = Thread(
        target=produce_frames_with_reconnect,
        args=(source_adapter, frame_queue, stats),
        kwargs={
            "stop_event": stop_event,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "backoff_factor": backoff_factor,
        },
    )
    consumer = Thread(
        target=consume_frame_queue,
        args=(frame_queue, runtime_engine, stats),
        kwargs={"stop_event": stop_event},
    )

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    if stats["producer_error"] is not None:
        raise stats["producer_error"]

    frame_count = stats["frame_count"]
    inference_count = stats["inference_count"]
    inference_results = stats["inference_results"]

    tracker = tracker or SingleTrackTracker()
    track_ids = tracker.assign_track_ids(inference_results)

    writer = ActionEventWriter()
    writer.add_results(inference_results, track_ids=track_ids)
    action_events = writer.get_log().events

    if emit_runtime_summary:
        print(f"Processed {frame_count} frames")
        print(f"Generated {inference_count} inference windows")
        print(f"Generated {len(action_events)} action events")

    return frame_count, inference_count, inference_results, action_events


def run_video(
    video_path: str,
    engine: Optional[InferenceEngine] = None,
    tracker: Optional[BaseTracker] = None,
    emit_runtime_summary: bool = True,
    stop_event: Optional[Event] = None,
) -> tuple[int, int, list[Any], list[Any]]:
    """Run offline inference on a single local video file."""
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string")

    source_adapter = FileSourceAdapter(video_path=Path(video_path))
    return run_source(
        source_adapter=source_adapter,
        engine=engine,
        tracker=tracker,
        emit_runtime_summary=emit_runtime_summary,
        stop_event=stop_event,
    )


__all__ = [
    "EOF_SENTINEL",
    "RuntimeFailureState",
    "SourceInterruptedError",
    "consume_frame_queue",
    "produce_frames",
    "produce_frames_from_source",
    "produce_frames_safe",
    "produce_frames_with_reconnect",
    "run_source",
    "run_source_with_reconnect",
    "run_video",
]
