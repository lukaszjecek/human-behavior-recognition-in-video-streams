"""Source adapters for inference frame ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2

SourceType = Literal["file", "rtsp"]


class InferenceSourceAdapter(ABC):
    """Common adapter contract for inference input sources."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return normalized source type identifier."""

    @property
    @abstractmethod
    def source_ref(self) -> str:
        """Return source path/URI in string form."""

    @abstractmethod
    def open_capture(self) -> cv2.VideoCapture:
        """Create and return a cv2 capture for this source."""


@dataclass(frozen=True)
class FileSourceAdapter(InferenceSourceAdapter):
    """Adapter for local video-file inference sources."""

    video_path: Path

    def __post_init__(self) -> None:
        """Validate file-based source metadata."""
        if not isinstance(self.video_path, Path):
            raise TypeError("video_path must be a pathlib.Path instance")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.video_path.is_file():
            raise ValueError(f"Video path must point to a file: {self.video_path}")

    @property
    def source_type(self) -> SourceType:
        """Return source type identifier."""
        return "file"

    @property
    def source_ref(self) -> str:
        """Return file path as string reference."""
        return str(self.video_path)

    def open_capture(self) -> cv2.VideoCapture:
        """Open cv2 capture for a local file source."""
        return cv2.VideoCapture(self.source_ref)


@dataclass(frozen=True)
class RtspSourceAdapter(InferenceSourceAdapter):
    """Adapter for RTSP inference sources."""

    rtsp_uri: str

    def __post_init__(self) -> None:
        """Validate RTSP URI."""
        if not isinstance(self.rtsp_uri, str):
            raise TypeError("rtsp_uri must be a string")
        normalized = self.rtsp_uri.strip()
        if not normalized:
            raise ValueError("rtsp_uri must not be empty")
        if not normalized.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("rtsp_uri must start with rtsp:// or rtsps://")
        object.__setattr__(self, "rtsp_uri", normalized)

    @property
    def source_type(self) -> SourceType:
        """Return source type identifier."""
        return "rtsp"

    @property
    def source_ref(self) -> str:
        """Return RTSP URI."""
        return self.rtsp_uri

    def open_capture(self) -> cv2.VideoCapture:
        """Open cv2 capture for an RTSP source."""
        return cv2.VideoCapture(self.source_ref)


def normalize_source_type(source_type: str) -> SourceType:
    """Normalize and validate a source type value."""
    if not isinstance(source_type, str):
        raise TypeError("source_type must be a string")
    normalized = source_type.strip().lower()
    if normalized == "file":
        return "file"
    if normalized == "rtsp":
        return "rtsp"
    raise ValueError("source_type must be one of: file, rtsp")


def build_source_adapter(
    source_type: str,
    source_ref: Path | str,
) -> InferenceSourceAdapter:
    """Build a source adapter for a given source type and source reference."""
    normalized_source_type = normalize_source_type(source_type)
    if normalized_source_type == "file":
        if isinstance(source_ref, Path):
            return FileSourceAdapter(video_path=source_ref)
        if isinstance(source_ref, str):
            return FileSourceAdapter(video_path=Path(source_ref))
        raise TypeError("file source_ref must be a pathlib.Path or string path")

    if not isinstance(source_ref, str):
        raise TypeError("rtsp source_ref must be a string URI")
    return RtspSourceAdapter(rtsp_uri=source_ref)


__all__ = [
    "InferenceSourceAdapter",
    "FileSourceAdapter",
    "RtspSourceAdapter",
    "build_source_adapter",
    "normalize_source_type",
]
