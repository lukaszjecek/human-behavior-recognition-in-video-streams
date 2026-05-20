"""Placeholder WebSocket routes for the backend.

Provides a namespace for future websocket endpoints.
"""
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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