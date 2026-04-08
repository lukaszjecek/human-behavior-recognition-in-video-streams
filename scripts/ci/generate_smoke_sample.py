"""Generate a tiny deterministic MP4 sample set for Docker smoke tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SMOKE_ROOT = Path("data/smoke_raw")
CLASSES: tuple[str, ...] = ("walking", "running", "falling")
FRAME_SIZE: tuple[int, int] = (160, 120)
FPS = 8
FRAME_COUNT = 16


def make_frame(class_name: str, frame_idx: int, width: int, height: int) -> np.ndarray:
    """Create a synthetic frame for a given class and frame index.

    Args:
        class_name: Logical class name used only for coloring and text.
        frame_idx: Zero-based frame index.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A BGR image compatible with OpenCV VideoWriter.
    """
    color_map: dict[str, tuple[int, int, int]] = {
        "walking": (40, 180, 40),
        "running": (180, 120, 40),
        "falling": (40, 80, 200),
    }
    base_color = color_map[class_name]

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = base_color

    x_pos = 10 + (frame_idx * 7) % max(width - 40, 1)
    y_pos = 20 + (frame_idx * 3) % max(height - 30, 1)

    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 25, y_pos + 25), (255, 255, 255), -1)
    cv2.putText(
        frame,
        class_name,
        (8, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"f={frame_idx:02d}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return frame


def write_clip(
    output_path: Path,
    class_name: str,
    frame_count: int = FRAME_COUNT,
    fps: int = FPS,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> None:
    """Write a small MP4 clip for the smoke dataset.

    Args:
        output_path: Target path of the generated MP4 file.
        class_name: Logical class name used for synthetic content.
        frame_count: Number of frames to generate.
        fps: Output frames per second.
        frame_size: Output frame size as ``(width, height)``.
    """
    width, height = frame_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    try:
        for frame_idx in range(frame_count):
            frame = make_frame(class_name, frame_idx, width, height)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    """Generate the smoke sample under ``data/smoke_raw``."""
    for class_name in CLASSES:
        target_path = SMOKE_ROOT / class_name / "sample_01.mp4"
        write_clip(target_path, class_name)

    print(f"[OK] Generated smoke sample in: {SMOKE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())