"""Offline producer-consumer runtime for adapter-based inference inputs."""
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Optional

from src.inference.engine import InferenceEngine
from src.inference.json_writer import ActionEventWriter
from src.inference.source_adapters import FileSourceAdapter, InferenceSourceAdapter
from src.inference.tracker import BaseTracker, SingleTrackTracker

EOF_SENTINEL = object()


def produce_frames_from_source(
    source_adapter: InferenceSourceAdapter,
    frame_queue: Queue,
    stop_event: Optional[Event] = None,
) -> None:
    """Reads frames from a source adapter in source order and pushes them to a queue.

    Args:
        source_adapter (InferenceSourceAdapter): Source adapter to open and read.
        frame_queue (Queue): Queue used to pass frames to the consumer.
        stop_event (Optional[Event]): Event to signal early termination.

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

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                ret, frame = cap.read()

                if not ret:
                    break

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
    """Runs the frame producer and stores any raised exception in stats."""
    try:
        produce_frames_from_source(source_adapter, frame_queue, stop_event)
    except Exception as exc:
        stats["producer_error"] = exc


def consume_frame_queue(
    frame_queue: Queue,
    engine: InferenceEngine,
    stats: dict,
    stop_event: Optional[Event] = None,
) -> None:
    """Consumes frames from a queue with an inference engine and updates runtime stats.

    Args:
        frame_queue (Queue): Queue providing video frames.
        engine (InferenceEngine): Engine used to process frames.
        stats (dict): Mutable stats dictionary with frame and inference counts.
        stop_event (Optional[Event]): Event to signal early termination.
    """
    frame_count = 0
    inference_results = []

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        frame = frame_queue.get()

        if frame is EOF_SENTINEL:
            break

        frame_count += 1
        result = engine.process_frame(frame)

        if result is not None:
            inference_results.append(result)

    stats["frame_count"] = frame_count
    stats["inference_count"] = len(inference_results)
    stats["inference_results"] = inference_results


def run_source(
    source_adapter: InferenceSourceAdapter,
    engine: Optional[InferenceEngine] = None,
    tracker: Optional[BaseTracker] = None,
    emit_runtime_summary: bool = True,
    stop_event: Optional[Event] = None,
) -> tuple[int, int, list[Any], list[Any]]:
    """Runs offline inference on a generic source adapter.

    Args:
        source_adapter: Adapter that provides inference input frames.
        engine: Optional inference engine instance. If None, a default
            InferenceEngine is created.
        tracker: Optional tracker used to assign track IDs to inference results.
        emit_runtime_summary: Whether to print processed frame/window/event stats.

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

    frame_queue = Queue()
    stats = {
        "frame_count": 0,
        "inference_count": 0,
        "inference_results": [],
        "producer_error": None,
    }

    producer = Thread(target=produce_frames_safe,
                      args=(source_adapter, frame_queue, stats, stop_event))
    consumer = Thread(target=consume_frame_queue,
                      args=(frame_queue, runtime_engine, stats, stop_event))

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
) -> tuple[int, int, list[Any], list[Any]]:
    """Runs offline inference on a single local video file."""
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string")

    source_adapter = FileSourceAdapter(video_path=Path(video_path))
    return run_source(
        source_adapter=source_adapter,
        engine=engine,
        tracker=tracker,
        emit_runtime_summary=emit_runtime_summary,
    )
