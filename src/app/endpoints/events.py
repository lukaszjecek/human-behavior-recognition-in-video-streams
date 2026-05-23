"""Implementation of event and alert history REST endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from starlette.requests import Request

from src.app.db.repository import get_distinct_session_ids, get_event_by_id, get_events
from src.app.db.session import get_db
from src.app.schemas.action_event import EventPayload
from src.inference.runtime_logging import (
    RuntimeLogContext,
    log_event,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=list[EventPayload],
    summary="Get Event and Alert History",
)
def read_events(
    request: Request,
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
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.DEBUG,
        "events_query_requested",
        f"Querying event and alert history. Filters: type={event_type}, camera={camera_id}, session={session_id}",
        log_context,
        query_event_type=event_type,
        query_camera_id=camera_id,
        query_session_id=str(session_id) if session_id else None,
        limit=limit,
        offset=offset,
    )
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
            log_event(
                logger,
                logging.ERROR,
                "events_query_failed",
                f"Failed to deserialize stored payload for event {db_evt.event_id}: {e}",
                log_context,
                exc_info=True,
                event_id=str(db_evt.event_id),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error parsing stored event payload",
            )
    log_event(
        logger,
        logging.DEBUG,
        "events_query_completed",
        f"Successfully retrieved {len(payloads)} event payloads.",
        log_context,
        retrieved_count=len(payloads),
    )
    return payloads


@router.get(
    "/sessions",
    response_model=list[UUID],
    summary="Get All Unique Session IDs with Stored Events",
)
def read_unique_sessions(
    request: Request,
    db: Session = Depends(get_db),
) -> list[UUID]:
    """Retrieve list of all unique session UUIDs that have persisted events in the database."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.DEBUG,
        "sessions_query_requested",
        "Querying all unique session IDs with stored events.",
        log_context,
    )
    session_ids = get_distinct_session_ids(db)
    log_event(
        logger,
        logging.DEBUG,
        "sessions_query_completed",
        f"Retrieved {len(session_ids)} unique session IDs.",
        log_context,
        session_ids_count=len(session_ids),
    )
    return session_ids


@router.get(
    "/sessions/{session_id}",
    response_model=list[EventPayload],
    summary="Get Events by Session ID",
)
def read_events_by_session(
    session_id: UUID,
    request: Request,
    event_type: str | None = Query(
        default=None,
        description="Filter by event type (DETECTION or ALERT)",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[EventPayload]:
    """Retrieve persisted event and alert payloads generated during a specific inference session."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.DEBUG,
        "session_events_query_requested",
        f"Querying events for session {session_id}. Filters: type={event_type}",
        log_context,
        target_session_id=str(session_id),
        query_event_type=event_type,
        limit=limit,
        offset=offset,
    )
    db_events = get_events(
        db,
        event_type=event_type,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )

    payloads: list[EventPayload] = []
    for db_evt in db_events:
        try:
            payloads.append(EventPayload.model_validate(db_evt.payload))
        except Exception as e:
            log_event(
                logger,
                logging.ERROR,
                "events_query_failed",
                f"Failed to deserialize stored payload for event {db_evt.event_id}: {e}",
                log_context,
                exc_info=True,
                event_id=str(db_evt.event_id),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error parsing stored event payload",
            )
    log_event(
        logger,
        logging.DEBUG,
        "session_events_query_completed",
        f"Retrieved {len(payloads)} events for session {session_id}.",
        log_context,
        target_session_id=str(session_id),
        retrieved_count=len(payloads),
    )
    return payloads


@router.get(
    "/{event_id}",
    response_model=EventPayload,
    summary="Get Event by ID",
)
def read_event(
    event_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> EventPayload:
    """Retrieve a single event or alert payload by its unique ID."""
    request_id = getattr(request.state, "request_id", None)
    log_context = RuntimeLogContext(session_id=request_id)
    log_event(
        logger,
        logging.DEBUG,
        "event_query_requested",
        f"Querying single event with ID {event_id}.",
        log_context,
        target_event_id=str(event_id),
    )
    db_event = get_event_by_id(db, event_id)
    if db_event is None:
        log_event(
            logger,
            logging.WARNING,
            "event_not_found",
            f"Event with ID {event_id} not found.",
            log_context,
            target_event_id=str(event_id),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )

    try:
        payload = EventPayload.model_validate(db_event.payload)
        log_event(
            logger,
            logging.DEBUG,
            "event_query_completed",
            f"Successfully retrieved event {event_id}.",
            log_context,
            target_event_id=str(event_id),
        )
        return payload
    except Exception as e:
        log_event(
            logger,
            logging.ERROR,
            "events_query_failed",
            f"Failed to deserialize stored payload for event {event_id}: {e}",
            log_context,
            exc_info=True,
            event_id=str(event_id),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error parsing stored event payload",
        )
