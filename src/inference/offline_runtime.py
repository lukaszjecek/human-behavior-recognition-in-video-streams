from pathlib import Path
import cv2

def run_video(video_path: str) -> None:
    """
    Runs offline inference on a single video file.

    Opens the video file, validates access, and prepares it for
    frame-by-frame processing in the offline runtime pipeline.

    Args:
        video_path (str): Path to the input .mp4 video file.

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

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

    cap.release()

    print(f"Processed {frame_count} frames")