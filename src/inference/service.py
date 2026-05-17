"""Reusable service entrypoint for adapter-based inference sources."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event

import torch

from src.inference.action_event import ActionEvent
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.offline_runtime import run_source
from src.inference.runtime import (
    InferenceRuntimeSettings,
    WindowModelAdapter,
    build_track_ids,
    expand_batched_inference_results,
    load_model_from_checkpoint,
    load_runtime_settings,
    resolve_inference_device,
)
from src.inference.source_adapters import (
    InferenceSourceAdapter,
    build_source_adapter,
    normalize_source_type,
)
from src.inference.tensorize import FrameTensorizer


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
) -> InferenceServiceResult:
    """Run inference and return typed in-memory results."""
    if not isinstance(request, InferenceServiceRequest):
        raise TypeError("request must be an InferenceServiceRequest instance")

    _validate_request(request)
    source_adapter = _build_request_source_adapter(request)

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

    frame_count, inference_count, inference_results, _ = run_source(
        source_adapter=source_adapter,
        engine=engine,
        emit_runtime_summary=False,
        stop_event=stop_event,
    )
    expanded_results = expand_batched_inference_results(inference_results)
    track_ids = build_track_ids(expanded_results, settings.default_track_id)

    writer = ActionEventWriter(class_labels=settings.class_labels)
    writer.add_results(expanded_results, track_ids=track_ids)

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
) -> InferenceServiceResult:
    """Run offline inference for local MP4 files only."""
    if not isinstance(request, InferenceServiceRequest):
        raise TypeError("request must be an InferenceServiceRequest instance")
    _validate_offline_mp4_request(request)
    return run_inference(request, stop_event=stop_event)


def _validate_offline_mp4_request(request: InferenceServiceRequest) -> None:
    """Validate MP4-only constraints for the backward-compatible wrapper."""
    source_type = normalize_source_type(request.source_type)
    if source_type != "file":
        raise ValueError(
            "run_offline_mp4_inference supports only file sources (source_type='file')"
        )
    if request.video_path is None:
        raise ValueError(
            "run_offline_mp4_inference requires request.video_path pointing to an .mp4 file"
        )
    if request.source_uri is not None:
        raise ValueError(
            "run_offline_mp4_inference does not accept request.source_uri; "
            "provide request.video_path"
        )
    if request.video_path.suffix.lower() != ".mp4":
        raise ValueError(
            "run_offline_mp4_inference requires request.video_path with .mp4 extension"
        )


def _validate_request(request: InferenceServiceRequest) -> None:
    """Validate request shape and path requirements."""
    if not isinstance(request.checkpoint_path, Path):
        raise TypeError("request.checkpoint_path must be a pathlib.Path instance")
    if not isinstance(request.config_path, Path):
        raise TypeError("request.config_path must be a pathlib.Path instance")
    if request.video_path is not None and not isinstance(request.video_path, Path):
        raise TypeError("request.video_path must be a pathlib.Path instance or None")
    if request.source_uri is not None and not isinstance(request.source_uri, str):
        raise TypeError("request.source_uri must be a string or None")
    normalize_source_type(request.source_type)
    if request.device is not None and not isinstance(request.device, str):
        raise TypeError("request.device must be a string or None")

    if request.video_path is not None and request.source_uri is not None:
        raise ValueError("Provide either request.video_path or request.source_uri, not both")

    source_type = normalize_source_type(request.source_type)
    if source_type == "file" and request.video_path is None and request.source_uri is None:
        raise ValueError("File source requires request.video_path or request.source_uri")
    if source_type == "rtsp" and request.source_uri is None:
        raise ValueError("RTSP source requires request.source_uri")


def _build_request_source_adapter(request: InferenceServiceRequest) -> InferenceSourceAdapter:
    """Build source adapter from request fields."""
    source_type = normalize_source_type(request.source_type)
    if source_type == "file":
        if request.video_path is not None:
            return build_source_adapter(source_type="file", source_ref=request.video_path)
        if request.source_uri is None:
            raise ValueError("File source requires request.video_path or request.source_uri")
        return build_source_adapter(source_type="file", source_ref=Path(request.source_uri))

    if request.source_uri is None:
        raise ValueError("RTSP source requires request.source_uri")
    return build_source_adapter(source_type="rtsp", source_ref=request.source_uri)
