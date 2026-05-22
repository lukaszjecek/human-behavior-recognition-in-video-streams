"""Database ORM models.

Defines the schema structure for persisting events and alerts.
"""

import uuid
from typing import Any

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import UUID as SQLUUID

Base: Any = declarative_base()


class DBEvent(Base):
    """Database record for system event and alert payloads.

    Stores the primary metadata fields directly as columns for indexing
    and fast queries, and keeps the full EventPayload structured structure
    in a JSON column.
    """

    __tablename__ = "events"

    event_id = Column(
        SQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the event.",
    )
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp of when the event occurred.",
    )
    camera_id = Column(
        String,
        nullable=True,
        index=True,
        doc="Optional ID of the source camera/video.",
    )
    event_type = Column(
        String,
        nullable=False,
        index=True,
        doc="Category of the event (e.g. DETECTION, ALERT).",
    )
    payload = Column(
        JSON,
        nullable=False,
        doc="Full serialized EventPayload JSON data.",
    )
