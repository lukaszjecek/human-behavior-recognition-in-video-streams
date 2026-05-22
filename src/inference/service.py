"""Reusable service entrypoint for adapter-based inference sources."""

import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Optional
from typing import Callable

import torch

from src.app.schemas.action_event import ActionEvent, AlertData, EventPayload, EventType
from src.inference.alert_state_machine import AlertStateMachine
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.offline_runtime import RuntimeFailureState, run_source_with_reconnect
from src.inference.runtime import (
    InferenceRuntimeSettings,
    WindowModelAdapter,
    expand_batched_inference_results,
    load_model_from_checkpoint,
    load_runtime_settings,
    resolve_inference_device,
)
from src.inference.runtime_logging import (
    RuntimeLogContext,
    configure_runtime_logging,
    get_build_metadata,
    log_event,
)
from src.inference.source_adapters import (
    InferenceSourceAdapter,
    build_source_adapter,
    normalize_source_type,
)
from src.inference.tensorize import FrameTensorizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceServiceRequest:
    """Input contract for programmatic inference over file/RTSP sources."""

    checkpoint_path: Path
    config_path: Path
    video_path: Path | None = None
    source_uri: str | None = None
    source_type: str = "file"
    device: str | None = None


@dataclass(frozen=True)
class InferenceServiceResult:
    """Result object returned by the reusable inference service."""

    frame_count: int
    inference_count: int
    inference_results: tuple[InferenceResult, ...]
    action_events: tuple[ActionEvent, ...]
    runtime_settings: InferenceRuntimeSettings
    resolved_device: torch.device

    @property
    def event_count(self) -> int:
        """Return number of generated action events."""
        return len(self.action_events)


def run_inference(
    request: InferenceServiceRequest,
    stop_event: Event | None = None,
    on_event: Optional[Callable[[EventPayload], None]] = None,
    session_id: str | None = None,
) -> InferenceServiceResult:
    """Run inference and return typed in-memory results.

    Args:
        request: Input request describing source and model settings.
        stop_event: Optional stop flag for graceful shutdown.
        session_id: Optional correlation ID for runtime logs.
    """
    if not isinstance(request, InferenceServiceRequest):
        raise TypeError("request must be an InferenceServiceRequest instance")

    configure_runtime_logging()
    _validate_request(request)
    source_adapter = _build_request_source_adapter(request)

    resolved_session_id = session_id or uuid.uuid4().hex
    log_context = RuntimeLogContext(
        session_id=resolved_session_id,
        source_type=source_adapter.source_type,
        source_ref=source_adapter.source_ref,
    )
    build_metadata = get_build_metadata()
    log_event(
        logger,
        logging.INFO,
        "inference_session_started",
        "Inference session started.",
        log_context,
        checkpoint_path=request.checkpoint_path,
        config_path=request.config_path,
        device_request=request.device,
        **build_metadata,
    )

    start_time = time.monotonic()
    settings = load_runtime_settings(request.config_path)
    device = resolve_inference_device(
        cli_device=request.device,
        config_device=settings.device,
    )
    model = load_model_from_checkpoint(request.checkpoint_path, device)

    tensorizer = FrameTensorizer(target_resolution=settings.target_resolution)
    model_adapter = WindowModelAdapter(
        model=model,
        tensorizer=tensorizer,
        device=device,
    )
    engine = InferenceEngine(
        window_size=settings.window_size,
        stride=settings.stride,
        model=model_adapter,
    )

    alert_sm = AlertStateMachine(
        persistence_threshold=settings.persistence_threshold,
        resolve_threshold=settings.resolve_threshold,
        danger_labels=settings.danger_labels,
    )
    writer = ActionEventWriter(class_labels=settings.class_labels)

    def handle_result(res: InferenceResult) -> None:
        expanded = expand_batched_inference_results([res])
        for r in expanded:
            tid = settings.default_track_id
            added = writer.add_result(r, track_id=tid)
            if added:
                evt = writer.get_log().events[-1]
                if on_event is not None:
                    detection_payload = EventPayload(
                        event_type=EventType.DETECTION,
                        data=evt,
                        camera_id=str(request.video_path.name) if request.video_path else None
                    )
                    on_event(detection_payload)

                alert_evt = alert_sm.process_event(evt)
                if alert_evt is not None:
                    alert_data = AlertData(
                        severity="HIGH",
                        message=f"Alert triggered for label: {alert_evt.label}",
                        action_event=alert_evt.triggering_event
                    )
                    if on_event is not None:
                        alert_payload = EventPayload(
                            event_type=EventType.ALERT,
                            data=alert_data,
                            camera_id=str(request.video_path.name) if request.video_path else None
                        )
                        on_event(alert_payload)

    frame_count, inference_count, inference_results, _ = run_source_with_reconnect(
        source_adapter=source_adapter,
        engine=engine,
        emit_runtime_summary=False,
        stop_event=stop_event,
        on_result=handle_result,
    log_event(
        logger,
        logging.INFO,
        "inference_runtime_configured",
        "Inference runtime configured.",
        log_context,
        window_size=settings.window_size,
        stride=settings.stride,
        target_resolution=settings.target_resolution,
        device=str(device),
        class_label_count=len(settings.class_labels),
    )

    try:
        frame_count, inference_count, inference_results, _ = run_source_with_reconnect(
            source_adapter=source_adapter,
            engine=engine,
            emit_runtime_summary=False,
            stop_event=stop_event,
            log_context=log_context,
        )
    except RuntimeFailureState as exc:
        log_event(
            logger,
            logging.ERROR,
            "inference_session_failed",
            "Inference session failed during runtime execution.",
            log_context,
            exc_info=True,
            phase=exc.phase,
            frames_before_failure=exc.frames_before_failure,
            error_type=type(exc.error).__name__,
        )
        raise
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "inference_session_failed",
            "Inference session failed with an unexpected error.",
            log_context,
            exc_info=True,
            error_type=type(exc).__name__,
        )
        raise
    expanded_results = expand_batched_inference_results(inference_results)

    log_event(
        logger,
        logging.INFO,
        "inference_session_completed",
        "Inference session completed.",
        log_context,
        frame_count=frame_count,
        inference_count=inference_count,
        event_count=len(writer.get_log().events),
        duration_s=round(time.monotonic() - start_time, 3),
    )

    return InferenceServiceResult(
        frame_count=frame_count,
        inference_count=inference_count,
        inference_results=tuple(expanded_results),
        action_events=tuple(writer.get_log().events),
        runtime_settings=settings,
        resolved_device=device,
    )


def run_offline_mp4_inference(

    request: InferenceServiceRequest,
    stop_event: Event | None = None,
    on_event: Optional[Callable[[EventPayload], None]] = None,
    session_id: str | None = None,
) -> InferenceServiceResult:
    """Run offline inference for local MP4 files only.

    Args:
        request: Input request describing source and model settings.
        stop_event: Optional stop flag for graceful shutdown.
        session_id: Optional correlation ID for runtime logs.
    """
    if not isinstance(request, InferenceServiceRequest):
        raise TypeError("request must be an InferenceServiceRequest instance")
    _validate_offline_mp4_request(request)
    return run_inference(request, stop_event=stop_event, on_event=on_event)
    return run_inference(request, stop_event=stop_event, session_id=session_id)


def _validate_offline_mp4_request(request: InferenceServiceRequest) -> None:
    """Validate MP4-only constraints for the backward-compatible wrapper."""
    source_type = normalize_source_type(request.source_type)
    if source_type != "file":
        raise ValueError(
            "run_offline_mp4_inference supports only file sources (source_type='file')"
        )
    if request.video_path is None:
        raise ValueError(
            "run_offline_mp4_inference requires request.video_path pointing to a video file"
        )
    if request.source_uri is not None:
        raise ValueError(
            "run_offline_mp4_inference does not accept request.source_uri; "
            "provide request.video_path"
        )
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".webp"}
    if request.video_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            "run_offline_mp4_inference requires request.video_path "
            f"with a supported video extension ({', '.join(sorted(allowed_extensions))})"
        )


def _validate_request(request: InferenceServiceRequest) -> None:
    """Validate request shape and path requirements."""
    if not isinstance(request.checkpoint_path, Path):
        raise TypeError(
            "request.checkpoint_path must be a pathlib.Path instance")
    if not isinstance(request.config_path, Path):
        raise TypeError("request.config_path must be a pathlib.Path instance")
    if request.video_path is not None and not isinstance(request.video_path, Path):
        raise TypeError(
            "request.video_path must be a pathlib.Path instance or None")
    if request.source_uri is not None and not isinstance(request.source_uri, str):
        raise TypeError("request.source_uri must be a string or None")
    normalize_source_type(request.source_type)
    if request.device is not None and not isinstance(request.device, str):
        raise TypeError("request.device must be a string or None")

    if request.video_path is not None and request.source_uri is not None:
        raise ValueError(
            "Provide either request.video_path or request.source_uri, not both")

    source_type = normalize_source_type(request.source_type)
    if source_type == "file" and request.video_path is None and request.source_uri is None:
        raise ValueError(
            "File source requires request.video_path or request.source_uri")
    if source_type == "rtsp" and request.source_uri is None:
        raise ValueError("RTSP source requires request.source_uri")


def _build_request_source_adapter(request: InferenceServiceRequest) -> InferenceSourceAdapter:
    """Build source adapter from request fields."""
    source_type = normalize_source_type(request.source_type)
    if source_type == "file":
        if request.video_path is not None:
            return build_source_adapter(source_type="file", source_ref=request.video_path)
        if request.source_uri is None:
            raise ValueError(
                "File source requires request.video_path or request.source_uri")
        return build_source_adapter(source_type="file", source_ref=Path(request.source_uri))

    if request.source_uri is None:
        raise ValueError("RTSP source requires request.source_uri")
    return build_source_adapter(source_type="rtsp", source_ref=request.source_uri)


def supports_session_id(func: Callable[..., object]) -> bool:
    """Return True when the callable accepts a session_id keyword argument."""
    target = getattr(func, "side_effect", None)
    if callable(target):
        func = target
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "session_id":
            return True
    return False
