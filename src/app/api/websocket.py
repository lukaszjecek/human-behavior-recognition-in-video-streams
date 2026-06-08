"""Placeholder WebSocket routes for the backend.

Provides a namespace for future websocket endpoints.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.app.services.camera_stream_manager import handle_camera_websocket
from src.app.services.websocket_manager import websocket_manager
from src.inference.runtime_logging import (
    RuntimeLogContext,
    configure_runtime_logging,
    log_event,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/ping", summary="Websocket ping placeholder")
async def ws_ping() -> dict[str, str]:
    """HTTP placeholder endpoint for websocket namespace."""
    return {"message": "websocket placeholder"}


@router.websocket("/echo")
async def websocket_echo(ws: WebSocket) -> None:
    """Simple echo websocket that sends back received text prefixed with 'echo: '."""
    configure_runtime_logging()
    request_id = ws.headers.get("X-Request-ID") or uuid.uuid4().hex
    log_context = RuntimeLogContext(session_id=request_id)
    await ws.accept()
    log_event(
        logger,
        logging.INFO,
        "websocket_connected",
        "Websocket client connected.",
        log_context,
        ws_path="/ws/echo",
    )
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"echo: {data}")
    except WebSocketDisconnect:
        # Normalne rozłączenie klienta - nie traktujemy tego jako błąd
        log_event(
            logger,
            logging.INFO,
            "websocket_disconnected",
            "Websocket client disconnected.",
            log_context,
            ws_path="/ws/echo",
        )
    except Exception as exc:
        # Obsługa nieoczekiwanych błędów
        log_event(
            logger,
            logging.ERROR,
            "websocket_failed",
            "Websocket handler failed with an exception.",
            log_context,
            exc_info=True,
            ws_path="/ws/echo",
            error_type=type(exc).__name__,
        )
    finally:
        # Upewniamy się, że połączenie zostanie zamknięte, jeśli jeszcze nie jest
        try:
            await ws.close()
        except RuntimeError:
            # Rzucane, jeśli socket został już zamknięty przez klienta
            pass


@router.websocket("/live")
async def websocket_live(ws: WebSocket) -> None:
    """Live streaming websocket that sends detection and alert events to clients."""
    configure_runtime_logging()
    request_id = ws.headers.get("X-Request-ID") or uuid.uuid4().hex
    log_context = RuntimeLogContext(session_id=request_id)
    await websocket_manager.connect(ws)
    log_event(
        logger,
        logging.INFO,
        "websocket_connected",
        "Websocket live client connected.",
        log_context,
        ws_path="/ws/live",
    )
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        log_event(
            logger,
            logging.INFO,
            "websocket_disconnected",
            "Websocket live client disconnected.",
            log_context,
            ws_path="/ws/live",
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "websocket_failed",
            "Websocket live handler failed with an exception.",
            log_context,
            exc_info=True,
            ws_path="/ws/live",
            error_type=type(exc).__name__,
        )
    finally:
        websocket_manager.disconnect(ws)
        try:
            await ws.close()
        except RuntimeError:
            pass


@router.websocket("/camera")
async def websocket_camera(ws: WebSocket) -> None:
    """Browser-camera WebSocket endpoint.

    Delegates the stream lifecycle, validation, and frame processing loop
    to the service layer.
    """
    await handle_camera_websocket(ws)
