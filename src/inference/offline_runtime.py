"""Offline producer-consumer runtime for MP4 inference."""
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2

from src.inference.engine import InferenceEngine

EOF_SENTINEL = object()


def produce_frames(video_path: str, frame_queue: Queue) -> None:
    """Reads frames from a video file in source order and pushes them to a queue.

    Args:
        video_path (str): Path to the input video file.
        frame_queue (Queue): Queue used to pass frames to the consumer.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    try:
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_queue.put(frame)
        finally:
            cap.release()
    finally:
        frame_queue.put(EOF_SENTINEL)

def produce_frames_safe(video_path: str, frame_queue: Queue, stats: dict) -> None:
    """Runs the frame producer and stores any raised exception in stats."""
    try:
        produce_frames(video_path, frame_queue)
    except Exception as exc:
        stats["producer_error"] = exc

def consume_frame_queue(frame_queue: Queue, engine: InferenceEngine, stats: dict) -> None:
    """Consumes frames from a queue with an inference engine and updates runtime stats.

    Args:
        frame_queue (Queue): Queue providing video frames.
        engine (InferenceEngine): Engine used to process frames.
        stats (dict): Mutable stats dictionary with frame and inference counts.
    """
    frame_count = 0
    inference_results = []

    while True:
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

def run_video(video_path: str) -> tuple[int, int, list]:
    """Runs offline inference on a single video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        tuple[int, int, list]: Number of processed frames, generated inference results and collected
        inferencemetadata/results.
    """
    engine = InferenceEngine()
    frame_queue = Queue()
    stats = {
        "frame_count": 0,
        "inference_count": 0,
        "inference_results": [],
        "producer_error": None,
    }

    producer = Thread(target=produce_frames_safe, args=(video_path, frame_queue, stats))
    consumer = Thread(target=consume_frame_queue, args=(frame_queue, engine, stats))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    if stats["producer_error"] is not None:
        raise stats["producer_error"]

    frame_count = stats["frame_count"]
    inference_count = stats["inference_count"]
    inference_results = stats["inference_results"]

    print(f"Processed {frame_count} frames")
    print(f"Generated {inference_count} inference windows")

    return frame_count, inference_count, inference_results