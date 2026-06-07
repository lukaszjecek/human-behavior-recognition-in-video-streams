"""Placeholder WebSocket routes for the backend.

Provides a namespace for future websocket endpoints.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.app.services.camera_stream_manager import CameraStreamSession
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

    Accepts initial JSON configuration metadata, then decodes and processes
    incoming binary camera frames, sending detection/alert events back to the caller.
    """
    configure_runtime_logging()
    request_id = ws.headers.get("X-Request-ID") or uuid.uuid4().hex
    log_context = RuntimeLogContext(session_id=request_id)

    await ws.accept()

    log_event(
        logger,
        logging.INFO,
        "camera_websocket_connected",
        "Camera websocket client connected.",
        log_context,
        ws_path="/ws/camera",
    )

    session: CameraStreamSession | None = None
    try:
        # 1. Parse initial JSON init message
        try:
            init_message = await ws.receive_json()
        except Exception as e:
            await ws.send_json(
                {
                    "message_type": "STATUS",
                    "session_id": request_id,
                    "status": "initialization_failed",
                    "message": "Invalid JSON initial message received.",
                    "error": str(e),
                }
            )
            await ws.close(code=4000)
            return

        checkpoint_path_str = init_message.get("checkpoint_path")
        config_path_str = init_message.get("config_path")
        device_str = init_message.get("device")
        session_id_str = init_message.get("session_id")

        if not checkpoint_path_str or not config_path_str:
            await ws.send_json(
                {
                    "message_type": "STATUS",
                    "session_id": request_id,
                    "status": "initialization_failed",
                    "message": "Missing checkpoint_path or config_path in initialization message.",
                }
            )
            await ws.close(code=4000)
            return

        # Generate or parse session ID
        session_uuid = None
        if session_id_str:
            try:
                session_uuid = uuid.UUID(session_id_str)
            except ValueError:
                await ws.send_json(
                    {
                        "message_type": "STATUS",
                        "session_id": request_id,
                        "status": "initialization_failed",
                        "message": f"Invalid session_id UUID format: {session_id_str}",
                    }
                )
                await ws.close(code=4000)
                return
        else:
            try:
                session_uuid = uuid.UUID(request_id)
            except ValueError:
                session_uuid = uuid.uuid4()

        try:
            session = CameraStreamSession(
                checkpoint_path=Path(checkpoint_path_str),
                config_path=Path(config_path_str),
                device=device_str,
                session_id=session_uuid,
            )
        except Exception as e:
            await ws.send_json(
                {
                    "message_type": "STATUS",
                    "session_id": str(session_uuid),
                    "status": "initialization_failed",
                    "message": "Pipeline initialization failed.",
                    "error": str(e),
                }
            )
            await ws.close(code=4000)
            return

        # Send successful initialization status back
        await ws.send_json(
            {
                "message_type": "STATUS",
                "session_id": str(session.session_id),
                "status": "initialized",
                "message": "Camera session successfully initialized.",
            }
        )

        # 2. Receive and process binary frames
        while True:
            msg = await ws.receive()
            if "text" in msg:
                text_data = msg["text"]
                if text_data == "stop":
                    log_event(
                        logger,
                        logging.INFO,
                        "camera_websocket_stop_requested",
                        "Stop request received from client.",
                        session.log_context,
                    )
                    await ws.send_json(
                        {
                            "message_type": "STATUS",
                            "session_id": str(session.session_id),
                            "status": "stopped",
                            "message": "Camera streaming stopped by request.",
                        }
                    )
                    break
                else:
                    # Ignore other text messages
                    continue
            elif "bytes" in msg:
                binary_frame = msg["bytes"]
                try:
                    events = await session.process_frame(binary_frame)
                    # Send generated events back to the client
                    for event in events:
                        await ws.send_json(event.model_dump(mode="json"))
                except Exception as frame_err:
                    log_event(
                        logger,
                        logging.ERROR,
                        "camera_websocket_frame_failed",
                        f"Failed to process frame: {frame_err}",
                        session.log_context,
                        exc_info=True,
                        error_type=type(frame_err).__name__,
                    )
                    # Send a status message to notify the frontend,
                    # but do NOT crash or close the connection
                    await ws.send_json(
                        {
                            "message_type": "STATUS",
                            "session_id": str(session.session_id),
                            "status": "running",
                            "message": f"Error processing frame: {frame_err}",
                        }
                    )

    except WebSocketDisconnect:
        log_event(
            logger,
            logging.INFO,
            "camera_websocket_disconnected",
            "Camera websocket client disconnected.",
            session.log_context if session else log_context,
            ws_path="/ws/camera",
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "camera_websocket_failed",
            "Camera websocket handler failed with an exception.",
            session.log_context if session else log_context,
            exc_info=True,
            ws_path="/ws/camera",
            error_type=type(exc).__name__,
        )
        if session:
            try:
                await ws.send_json(
                    {
                        "message_type": "STATUS",
                        "session_id": str(session.session_id),
                        "status": "failed",
                        "message": "Internal error occurred.",
                        "error": str(exc),
                    }
                )
            except Exception:
                pass
    finally:
        if session:
            session.close()
        try:
            await ws.close()
        except RuntimeError:
            pass
