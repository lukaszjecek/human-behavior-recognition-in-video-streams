"""Action event record schema for JSON serialization.

Defines the data structure for detected actions/behaviors with confidence scores
and temporal/spatial metadata.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ActionEvent:
    """Represents a single detected action/behavior event with temporal and confidence metadata.

    Attributes:
        start_frame_index: Starting frame index of the detection window.
        end_frame_index: Ending frame index of the detection window.
        start_timestamp: Optional starting timestamp in seconds (float).
        end_timestamp: Optional ending timestamp in seconds (float).
        label: String label of the detected action/behavior class.
        confidence: Confidence score of the prediction (0.0 to 1.0).
        track_id: Optional tracking ID for multi-object tracking scenarios.
    """
    start_frame_index: int
    end_frame_index: int
    label: str
    confidence: float
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    track_id: Optional[int] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.start_frame_index < 0:
            raise ValueError("start_frame_index must be >= 0")
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("end_frame_index must be >= start_frame_index")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.start_timestamp is not None and self.end_timestamp is not None:
            if self.end_timestamp < self.start_timestamp:
                raise ValueError("end_timestamp must be >= start_timestamp")

    def to_dict(self) -> dict:
        """Convert to dictionary, filtering out None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionEvent":
        """Create ActionEvent from dictionary."""
        return cls(**data)


class ActionEventLog:
    """Container for a log of action events with serialization support."""

    def __init__(self):
        self.events: list[ActionEvent] = []

    def add_event(self, event: ActionEvent) -> None:
        """Add an action event to the log."""
        self.events.append(event)

    def add_events(self, events: list[ActionEvent]) -> None:
        """Add multiple action events to the log."""
        self.events.extend(events)

    def to_dict(self) -> dict:
        """Convert log to dictionary format."""
        return {
            "events": [event.to_dict() for event in self.events],
            "event_count": len(self.events),
        }

    def to_json(self) -> str:
        """Convert log to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save_to_file(self, filepath: str) -> None:
        """Save log to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "ActionEventLog":
        """Load log from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        log = cls()
        for event_data in data.get("events", []):
            log.add_event(ActionEvent.from_dict(event_data))
        return log

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
