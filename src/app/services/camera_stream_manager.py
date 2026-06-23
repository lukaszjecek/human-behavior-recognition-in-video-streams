"""Service layer for managing live camera WebSocket streams and inference."""

import asyncio
import logging
import os
import threading
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect

from src.app.db.repository import save_event
from src.app.db.session import SessionLocal
from src.app.schemas.action_event import EventPayload
from src.app.services.websocket_manager import websocket_manager
from src.inference.alert_state_machine import AlertStateMachine
from src.inference.engine import InferenceEngine
from src.inference.json_writer import ActionEventWriter
from src.inference.pipeline import InferenceEventPipeline
from src.inference.runtime import (
    WindowModelAdapter,
    load_model_from_checkpoint,
    load_runtime_settings,
    resolve_inference_device,
)
from src.inference.runtime_logging import (
    RuntimeLogContext,
    log_audit_event,
    log_event,
)
from src.inference.tensorize import FrameTensorizer

logger = logging.getLogger(__name__)


def is_relative_to_safe(path: Path, parent: Path) -> bool:
    """Check if path is relative to parent safely (supporting cross-drive/different formats)."""
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except ValueError:
        return False


def validate_safe_path(path: Path, allowed_extensions: list[str]) -> bool:
    """Validate that path is safe, has correct suffix, and exists.

    Prevents directory traversal attacks.
    """
    try:
        resolved_path = path.resolve()
    except Exception:
        return False

    if resolved_path.suffix.lower() not in allowed_extensions:
        return False

    if not resolved_path.is_file():
        return False

    cwd = Path.cwd().resolve()

    # Check standard temp directories (important for running tests safely)
    temp_paths = [Path(os.environ.get("TEMP", "/tmp"))]
    if "TMP" in os.environ:
        temp_paths.append(Path(os.environ["TMP"]))

    # Check relative to cwd, any of the temp paths, or /app (for Docker container deployment)
    if is_relative_to_safe(resolved_path, cwd):
        return True
    for tp in temp_paths:
        if is_relative_to_safe(resolved_path, tp):
            return True
    if is_relative_to_safe(resolved_path, Path("/app")):
        return True

    return False


class ModelCache:
    """Thread-safe cache for loaded PyTorch models to avoid expensive per-session reloads."""

    def __init__(self) -> None:
        """Initialize the model cache registry and re-entrant lock."""
        self._cache: dict[tuple[str, str], torch.nn.Module] = {}
        self._lock = threading.Lock()

    def get_model(self, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
        """Resolve path and retrieve/load the model in a thread-safe manner."""
        resolved_path = checkpoint_path.resolve()
        key = (str(resolved_path), str(device))
        with self._lock:
            if key not in self._cache:
                model = load_model_from_checkpoint(resolved_path, device)
                self._cache[key] = model
            return self._cache[key]

    def clear(self) -> None:
        """Clear all loaded models from cache."""
        with self._lock:
            self._cache.clear()


model_cache = ModelCache()


class CameraStreamSession:
    """Manages inference state for a browser camera WebSocket stream session."""

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        device: str | None = None,
        session_id: UUID | None = None,
    ) -> None:
        """Initialize the camera stream session.

        This sets up the settings, resolved hardware device, model, and the
        inference state machine.
        """
        self.session_id = session_id or uuid4()
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device_request = device

        self.log_context = RuntimeLogContext(
            session_id=str(self.session_id),
            source_type="websocket_camera",
            source_ref="live_stream",
        )

        log_event(
            logger,
            logging.INFO,
            "camera_session_initializing",
            "Initializing camera stream session.",
            self.log_context,
            checkpoint_path=str(self.checkpoint_path),
            config_path=str(self.config_path),
            device_request=self.device_request,
        )

        # 1. Load configuration and resolve paths/device (with safe path validation)
        if not validate_safe_path(self.config_path, [".yml", ".yaml"]):
            raise ValueError(
                f"Configuration file path is invalid or restricted: {self.config_path}"
            )
        if not validate_safe_path(self.checkpoint_path, [".pth", ".pt"]):
            raise ValueError(
                f"Model checkpoint file path is invalid or restricted: {self.checkpoint_path}"
            )

        self.settings = load_runtime_settings(self.config_path)
        self.device = resolve_inference_device(
            cli_device=self.device_request,
            config_device=self.settings.device,
        )

        # 2. Load model & adapters (utilizing ModelCache to reuse model weights)
        self.model = model_cache.get_model(self.checkpoint_path, self.device)
        self.tensorizer = FrameTensorizer(target_resolution=self.settings.target_resolution)
        self.model_adapter = WindowModelAdapter(
            model=self.model,
            tensorizer=self.tensorizer,
            device=self.device,
        )

        # 3. Create prediction pipeline engine
        self.engine = InferenceEngine(
            window_size=self.settings.window_size,
            stride=1,
            model=self.model_adapter,
        )
        self.alert_sm = AlertStateMachine(
            persistence_threshold=self.settings.persistence_threshold,
            resolve_threshold=self.settings.resolve_threshold,
            danger_labels=self.settings.danger_labels,
        )
        self.writer = ActionEventWriter(class_labels=self.settings.class_labels)

        context_module = None
        try:
            from src.inference.context_adapter import ContextModule

            context_module = ContextModule()
        except Exception as exc:
            logger.warning(
                "ContextModule failed to initialize for browser_camera session: %s", exc
            )

        self.bbox_hook = None
        if self.settings.bbox_enabled:
            try:
                from src.inference.bbox_detector import get_or_create_bbox_enricher

                self.bbox_hook = get_or_create_bbox_enricher(
                    model_name=self.settings.bbox_model_name,
                    confidence_threshold=self.settings.bbox_confidence_threshold,
                    weights_dir=self.settings.bbox_weights_dir,
                    frame_selector=self.settings.bbox_frame_selector,
                )
            except Exception as exc:
                logger.warning(
                    "BBoxEnricher failed to initialize for browser_camera session: %s", exc
                )

        self.pipeline = InferenceEventPipeline(
            engine=self.engine,
            writer=self.writer,
            alert_processor=self.alert_sm,
            context_module=context_module,
            bbox_hook=self.bbox_hook,
            camera_id="browser_camera",
            session_id=self.session_id,
            track_id=self.settings.default_track_id,
        )

        self._current_events: list[EventPayload] = []

        log_event(
            logger,
            logging.INFO,
            "camera_session_initialized",
            "Camera stream session successfully initialized.",
            self.log_context,
            window_size=self.settings.window_size,
            stride=self.settings.stride,
            target_resolution=self.settings.target_resolution,
            device=str(self.device),
        )

    async def process_frame(self, binary_frame: bytes) -> list[EventPayload]:
        """Decode binary frame bytes and process them through the pipeline.

        Decoding + inference runs in a background thread to prevent blocking ASGI.
        """
        return await asyncio.to_thread(self.process_frame_sync, binary_frame)

    def process_frame_sync(self, binary_frame: bytes) -> list[EventPayload]:
        """Decode and process a binary frame through the pipeline and generate events."""
        self._current_events = []

        nparr = np.frombuffer(binary_frame, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode binary frame bytes into BGR image.")

        payloads = self.pipeline.push_frame(frame_bgr)
        for payload in payloads:
            self._current_events.append(payload)
            self._handle_event_outputs(payload)

        return list(self._current_events)

    def _handle_event_outputs(self, payload: EventPayload) -> None:
        """Broadcast events to /ws/live and persist them in db + audit trail."""
        # Broadcast event to other live listening clients
        websocket_manager.broadcast_sync(payload)

        # Save event to database
        try:
            with SessionLocal() as db:
                save_event(db, payload)
        except Exception as db_err:
            log_event(
                logger,
                logging.ERROR,
                "database_write_failed",
                (
                    f"Database write-path failure for browser camera "
                    f"event {payload.event_id}: {db_err}"
                ),
                self.log_context,
                exc_info=True,
                event_id=str(payload.event_id),
                error_type=type(db_err).__name__,
            )

        # Write to local file audit.log
        try:
            log_audit_event(payload)
        except Exception as audit_err:
            log_event(
                logger,
                logging.ERROR,
                "audit_file_write_failed",
                (
                    f"Failed to write browser camera event {payload.event_id} "
                    f"to audit log: {audit_err}"
                ),
                self.log_context,
                exc_info=True,
                event_id=str(payload.event_id),
                error_type=type(audit_err).__name__,
            )

        # Log structured audit details
        is_detection = payload.event_type.value == "DETECTION"
        event_name = "audit_detection_published" if is_detection else "audit_alert_triggered"
        log_event(
            logger,
            logging.INFO,
            event_name,
            f"Published {payload.event_type.value} event {payload.event_id} "
            f"for camera session {self.session_id}.",
            self.log_context,
            event_id=str(payload.event_id),
            camera_id=payload.camera_id,
            event_type=payload.event_type.value,
        )

    def close(self) -> None:
        """Close session resources and log closure."""
        log_event(
            logger,
            logging.INFO,
            "camera_session_closed",
            f"Camera stream session {self.session_id} closed.",
            self.log_context,
        )


async def handle_camera_websocket(ws: WebSocket) -> None:
    """Handles the lifecycle of a browser-camera WebSocket connection.

    This includes receiving the initial JSON configuration, validating paths/session,
    initializing the inference pipeline session (reusing cached model weights),
    and processing incoming binary video frames in real-time.
    """
    import uuid

    from src.inference.runtime_logging import configure_runtime_logging

    configure_runtime_logging()
    request_id = ws.headers.get("X-Request-ID") or uuid.uuid4().hex
    log_context = RuntimeLogContext(
        session_id=request_id,
        source_type="websocket_camera",
        source_ref="live_stream",
    )

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
    session_uuid = None
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
                    "error_type": type(e).__name__,
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
        if session_id_str:
            try:
                session_uuid = UUID(session_id_str)
            except ValueError as e:
                await ws.send_json(
                    {
                        "message_type": "STATUS",
                        "session_id": request_id,
                        "status": "initialization_failed",
                        "message": f"Invalid session_id UUID format: {session_id_str}",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                await ws.close(code=4000)
                return
        else:
            try:
                session_uuid = UUID(request_id)
            except ValueError:
                session_uuid = uuid4()

        # Initialize the session inside the thread pool to avoid blocking the event loop
        try:
            session = await asyncio.to_thread(
                CameraStreamSession,
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
                    "error_type": type(e).__name__,
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
                    # Send generated events back to the client immediately (realtime)
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
                            "error": str(frame_err),
                            "error_type": type(frame_err).__name__,
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
                        "error_type": type(exc).__name__,
                    }
                )
            except Exception:
                pass
        else:
            try:
                await ws.send_json(
                    {
                        "message_type": "STATUS",
                        "session_id": str(session_uuid) if session_uuid else request_id,
                        "status": "failed",
                        "message": "Internal error occurred.",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
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
