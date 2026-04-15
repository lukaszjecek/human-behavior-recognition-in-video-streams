from pathlib import Path
import cv2

from src.inference.engine import InferenceEngine

def read_video_frames(video_path: str):
    """
    Yields frames from a video file in source order.

    Args:
        video_path (str): Path to the input video file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
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

            yield frame
    finally:
        cap.release()

def consume_frames(frames, engine: InferenceEngine) -> tuple[int, int]:
    """
    Consumes frames with an inference engine and counts processed frames and results.

    Args:
        frames: Iterable of video frames.
        engine (InferenceEngine): Engine used to process frames.

    Returns:
        tuple[int, int]: Number of processed frames and generated inference results.
    """
    frame_count = 0
    inference_count = 0

    for frame in frames:
        frame_count += 1
        result = engine.process_frame(frame)

        if result is not None:
            inference_count += 1

    return frame_count, inference_count

def run_video(video_path: str) -> tuple[int, int]:
    """
    Runs offline inference on a single video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        tuple[int, int]: Number of processed frames and generated inference results.
    """
    engine = InferenceEngine()
    frames = read_video_frames(video_path)
    frame_count, inference_count = consume_frames(frames, engine)

    print(f"Processed {frame_count} frames")
    print(f"Generated {inference_count} inference windows")

    return frame_count, inference_count