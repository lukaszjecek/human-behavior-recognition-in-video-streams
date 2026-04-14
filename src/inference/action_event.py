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

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not isinstance(self.start_frame_index, int):
            raise TypeError("start_frame_index must be an integer")
        if not isinstance(self.end_frame_index, int):
            raise TypeError("end_frame_index must be an integer")
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise TypeError("confidence must be a float")
        
        if self.start_frame_index < 0:
            raise ValueError("start_frame_index must be >= 0")
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("end_frame_index must be >= start_frame_index")
        if not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        
        if self.start_timestamp is not None:
            is_number = isinstance(self.start_timestamp, (int, float))
            is_bool = isinstance(self.start_timestamp, bool)
            if not is_number or is_bool:
                raise TypeError("start_timestamp must be a float")
            if self.start_timestamp < 0:
                raise ValueError("start_timestamp must be >= 0")
        if self.end_timestamp is not None:
            is_number = isinstance(self.end_timestamp, (int, float))
            is_bool = isinstance(self.end_timestamp, bool)
            if not is_number or is_bool:
                raise TypeError("end_timestamp must be a float")
            if self.end_timestamp < 0:
                raise ValueError("end_timestamp must be >= 0")
        if self.start_timestamp is not None and self.end_timestamp is not None:
            if self.end_timestamp < self.start_timestamp:
                raise ValueError("end_timestamp must be >= start_timestamp")
        
        if self.track_id is not None:
            if not isinstance(self.track_id, int) or isinstance(self.track_id, bool):
                raise TypeError("track_id must be an integer")

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

    def __init__(self) -> None:
        """Initialize the ActionEventLog container."""
        self.events: list[ActionEvent] = []

    def add_event(self, event: ActionEvent) -> None:
        """Add an action event to the log.
        
        Args:
            event: ActionEvent instance to add.
            
        Raises:
            TypeError: If event is not an ActionEvent instance.
        """
        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent instance")
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
        """Load log from JSON file.
        
        Args:
            filepath: Path to JSON file.
            
        Returns:
            ActionEventLog instance.
            
        Raises:
            ValueError: If loaded event_count doesn't match number of events.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        log = cls()
        for event_data in data.get("events", []):
            log.add_event(ActionEvent.from_dict(event_data))
        
        # Validate event_count consistency
        loaded_count = data.get("event_count")
        if loaded_count is not None and loaded_count != len(log.events):
            raise ValueError(
                f"event_count mismatch: file claims {loaded_count} events "
                f"but {len(log.events)} were loaded"
            )
        return log

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
