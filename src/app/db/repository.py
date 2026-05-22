"""Persistence layer repositories.

Implements database write-path operations, queries, and write-path validation.
"""

import logging
from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.app.db.models import DBEvent
from src.app.schemas.action_event import EventPayload

logger = logging.getLogger(__name__)


def save_event(db: Session, payload: EventPayload) -> DBEvent:
    """Validate and persist an EventPayload record in the database.

    Args:
        db: The SQLAlchemy Session to use for writing.
        payload: The validated EventPayload instance to save.

    Returns:
        DBEvent: The persisted database model instance.

    Raises:
        ValueError: If write-path validation fails (e.g. invalid event_id).
        Exception: If the database write fails.
    """
    if not isinstance(payload, EventPayload):
        raise TypeError("payload must be an instance of EventPayload")

    if not payload.event_id:
        raise ValueError("Event payload must contain a valid event_id")

    try:
        db_event = DBEvent(
            event_id=payload.event_id,
            timestamp=payload.timestamp,
            camera_id=payload.camera_id,
            event_type=payload.event_type.value,
            session_id=payload.session_id,
            payload=payload.model_dump(mode="json"),
        )
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        logger.info(
            "Successfully persisted event %s (type: %s) to database.",
            db_event.event_id,
            db_event.event_type,
        )
        return db_event
    except Exception:
        db.rollback()
        logger.exception("Failed to write event %s to database.", payload.event_id)
        raise


def get_events(
    db: Session,
    event_type: str | None = None,
    camera_id: str | None = None,
    session_id: UUID | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[DBEvent]:
    """Query persisted events with filtering, ordering, and pagination.

    Args:
        db: The database session.
        event_type: Optional filter for event type (DETECTION / ALERT).
        camera_id: Optional filter for source camera reference.
        session_id: Optional filter for session ID.
        limit: Max number of records to return.
        offset: Number of records to skip.

    Returns:
        Sequence[DBEvent]: List of matching database records ordered by newest first.
    """
    stmt = select(DBEvent)
    if event_type:
        stmt = stmt.where(DBEvent.event_type == event_type)
    if camera_id:
        stmt = stmt.where(DBEvent.camera_id == camera_id)
    if session_id:
        stmt = stmt.where(DBEvent.session_id == session_id)

    # Order by newest first
    stmt = stmt.order_by(DBEvent.timestamp.desc()).offset(offset).limit(limit)
    return db.scalars(stmt).all()


def get_event_by_id(db: Session, event_id: UUID) -> DBEvent | None:
    """Retrieve a single persisted event by its UUID identifier.

    Args:
        db: The database session.
        event_id: The UUID of the event.

    Returns:
        DBEvent | None: The database record if found, else None.
    """
    return db.get(DBEvent, event_id)


def get_distinct_session_ids(db: Session) -> list[UUID]:
    """Query the database for all unique, non-null session UUIDs associated with stored events.

    Args:
        db: The database session.

    Returns:
        list[UUID]: List of unique session UUIDs.
    """
    stmt = select(DBEvent.session_id).where(DBEvent.session_id.isnot(None)).distinct()
    return list(db.scalars(stmt).all())
