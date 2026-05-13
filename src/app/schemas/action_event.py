"""Event schemas for JSON serialization.

Defines the data structure for detected actions/behaviors with confidence scores
and temporal/spatial metadata.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


class EventType(str, Enum):
    """Supported event categories for serialized event payloads."""
    DETECTION = "DETECTION"
    ALERT = "ALERT"


class ContextData(BaseModel):
    """Contextual metadata for the detection."""
    scene_tag: StrictStr = Field(min_length=1)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)


class ActionEvent(BaseModel):
    """Represents a single detected action/behavior event with temporal and confidence metadata."""
    model_config = ConfigDict(validate_assignment=True)

    start_frame_index: StrictInt = Field(ge=0)
    end_frame_index: StrictInt = Field(ge=0)
    label: StrictStr = Field(min_length=1)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    start_timestamp: Optional[StrictFloat] = Field(default=None, ge=0.0)
    end_timestamp: Optional[StrictFloat] = Field(default=None, ge=0.0)
    track_id: Optional[StrictInt] = None
    context: Optional[ContextData] = None

    @model_validator(mode='after')
    def validate_interdependent_fields(self) -> "ActionEvent":
        """Validate interdependent fields."""
        if not self.label.strip():
            raise ValueError("label must not be empty or consist only of whitespace")
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("end_frame_index must be >= start_frame_index")
        if self.start_timestamp is not None and self.end_timestamp is not None:
            if self.end_timestamp < self.start_timestamp:
                raise ValueError("end_timestamp must be >= start_timestamp")
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary, filtering out None values."""
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionEvent":
        """Create ActionEvent from dictionary."""
        return cls(**data)


class AlertData(BaseModel):
    """Data specific to alerts."""
    severity: StrictStr
    message: StrictStr
    action_event: ActionEvent


class EventPayload(BaseModel):
    """Wrapper for all events in the system."""
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: Optional[StrictStr] = None
    version: StrictStr = "1.0"
    event_type: EventType
    data: Union[AlertData, ActionEvent]

    @model_validator(mode="after")
    def validate_event_type_matches_data(self) -> "EventPayload":
        """Ensure the declared event type matches the concrete data payload."""
        if self.event_type == EventType.DETECTION and not isinstance(self.data, ActionEvent):
            raise ValueError("event_type DETECTION requires data to be an ActionEvent")
        if self.event_type == EventType.ALERT and not isinstance(self.data, AlertData):
            raise ValueError("event_type ALERT requires data to be an AlertData")
        return self


class ActionEventLog(BaseModel):
    """Container for a log of action events with serialization support."""
    events: List[ActionEvent] = Field(default_factory=list)

    @property
    def event_count(self) -> int:
        """Return the number of events in the log."""
        return len(self.events)

    def add_event(self, event: object) -> None:
        """Add an action event to the log.
        
        Args:
            event: ActionEvent instance to add.
            
        Raises:
            TypeError: If event is not an ActionEvent instance.
        """
        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent instance")
        self.events.append(event)

    def add_events(self, events: List[ActionEvent]) -> None:
        """Add multiple action events to the log."""
        for event in events:
            if not isinstance(event, ActionEvent):
                raise TypeError("event must be an ActionEvent instance")
        self.events.extend(events)

    def to_dict(self) -> dict:
        """Convert log to dictionary format."""
        return {
            "events": [event.to_dict() for event in self.events],
            "event_count": self.event_count,
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
