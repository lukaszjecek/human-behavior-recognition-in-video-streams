"""Service layer for managing live camera WebSocket streams and inference."""

import asyncio
import logging
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

from src.app.db.repository import save_event
from src.app.db.session import SessionLocal
from src.app.schemas.action_event import AlertData, EventPayload, EventType
from src.app.services.websocket_manager import websocket_manager
from src.inference.alert_state_machine import AlertStateMachine
from src.inference.engine import InferenceEngine
from src.inference.json_writer import ActionEventWriter
from src.inference.runtime import (
    WindowModelAdapter,
    expand_batched_inference_results,
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

        # 1. Load configuration and resolve paths/device
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint file not found: {self.checkpoint_path}")

        self.settings = load_runtime_settings(self.config_path)
        self.device = resolve_inference_device(
            cli_device=self.device_request,
            config_device=self.settings.device,
        )

        # 2. Load model & adapters
        self.model = load_model_from_checkpoint(self.checkpoint_path, self.device)
        self.tensorizer = FrameTensorizer(target_resolution=self.settings.target_resolution)
        self.model_adapter = WindowModelAdapter(
            model=self.model,
            tensorizer=self.tensorizer,
            device=self.device,
        )

        # 3. Create prediction pipeline engine
        self.engine = InferenceEngine(
            window_size=self.settings.window_size,
            stride=self.settings.stride,
            model=self.model_adapter,
        )
        self.alert_sm = AlertStateMachine(
            persistence_threshold=self.settings.persistence_threshold,
            resolve_threshold=self.settings.resolve_threshold,
            danger_labels=self.settings.danger_labels,
        )
        self.writer = ActionEventWriter(class_labels=self.settings.class_labels)

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

        res = self.engine.process_frame(frame_bgr)
        if res is not None:
            expanded = expand_batched_inference_results([res])
            for r in expanded:
                tid = self.settings.default_track_id
                added = self.writer.add_result(r, track_id=tid)
                if added:
                    evt = self.writer.get_log().events[-1]

                    # 1. Detection payload
                    detection_payload = EventPayload(
                        event_type=EventType.DETECTION,
                        data=evt,
                        camera_id="browser_camera",
                        session_id=self.session_id,
                    )
                    self._current_events.append(detection_payload)
                    self._handle_event_outputs(detection_payload)

                    # 2. Check state machine alerts
                    alert_evt = self.alert_sm.process_event(evt)
                    if alert_evt is not None:
                        alert_data = AlertData(
                            severity="HIGH",
                            message=f"Alert triggered for label: {alert_evt.label}",
                            action_event=alert_evt.triggering_event,
                        )
                        alert_payload = EventPayload(
                            event_type=EventType.ALERT,
                            data=alert_data,
                            camera_id="browser_camera",
                            session_id=self.session_id,
                        )
                        self._current_events.append(alert_payload)
                        self._handle_event_outputs(alert_payload)

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
