"""Implementation of event and alert history REST endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.app.db.repository import get_event_by_id, get_events
from src.app.db.session import get_db
from src.app.schemas.action_event import EventPayload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=list[EventPayload],
    summary="Get Event and Alert History",
)
def read_events(
    event_type: str | None = Query(
        default=None,
        description="Filter by event type (DETECTION or ALERT)",
    ),
    camera_id: str | None = Query(
        default=None,
        description="Filter by camera/video reference",
    ),
    session_id: UUID | None = Query(
        default=None,
        description="Filter by session ID",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[EventPayload]:
    """Retrieve persisted event and alert payloads, supporting filtering and pagination."""
    db_events = get_events(
        db,
        event_type=event_type,
        camera_id=camera_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )

    payloads: list[EventPayload] = []
    for db_evt in db_events:
        try:
            # db_evt.payload is parsed as dict automatically by SQLAlchemy JSON type
            payloads.append(EventPayload.model_validate(db_evt.payload))
        except Exception as e:
            logger.error(
                "Failed to deserialize stored payload for event %s: %s",
                db_evt.event_id,
                e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error parsing stored event payload",
            )
    return payloads


@router.get(
    "/{event_id}",
    response_model=EventPayload,
    summary="Get Event by ID",
)
def read_event(
    event_id: UUID,
    db: Session = Depends(get_db),
) -> EventPayload:
    """Retrieve a single event or alert payload by its unique ID."""
    db_event = get_event_by_id(db, event_id)
    if db_event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )

    try:
        return EventPayload.model_validate(db_event.payload)
    except Exception as e:
        logger.error(
            "Failed to deserialize stored payload for event %s: %s",
            event_id,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error parsing stored event payload",
        )
