"""Tracker abstractions and simple tracking backends for inference results."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.inference.engine import InferenceResult


class BaseTracker(ABC):
    """Abstract tracker interface for assigning track IDs to inference results."""

    @abstractmethod
    def assign_track_ids(self, results: List[InferenceResult]) -> List[Optional[int]]:
        """Assign track IDs to inference results.

        Args:
            results: Inference results to associate across frames/windows.

        Returns:
            List of track IDs aligned with the input results.
        """
        raise NotImplementedError

class SingleTrackTracker(BaseTracker):
    """Simple tracker that assigns a single persistent track ID to all results.

    Assumptions:
        - The current pipeline represents a single continuous subject/track.
        - Results belong to one identity across consecutive inference windows.

    Limitations:
        - Does not perform multi-object association.
        - Does not use spatial matching, bounding boxes, or re-identification.
        - Intended as an initial stable backend until richer tracking inputs are available.
    """

    def __init__(self, track_id: int = 1) -> None:
        """Initialize the tracker with a fixed track ID."""
        if not isinstance(track_id, int) or isinstance(track_id, bool):
            raise TypeError("track_id must be an integer")

        self.track_id = track_id

    def assign_track_ids(self, results: List[InferenceResult]) -> List[Optional[int]]:
        """Assign the same track ID to every result."""
        return [self.track_id] * len(results)