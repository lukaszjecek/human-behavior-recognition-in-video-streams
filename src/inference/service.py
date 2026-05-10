"""Reusable service entrypoint for offline MP4 inference."""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.inference.action_event import ActionEvent
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.offline_runtime import run_video
from src.inference.runtime import (
    InferenceRuntimeSettings,
    WindowModelAdapter,
    build_track_ids,
    expand_batched_inference_results,
    load_model_from_checkpoint,
    load_runtime_settings,
    resolve_inference_device,
)
from src.inference.tensorize import FrameTensorizer


@dataclass(frozen=True)
class InferenceServiceRequest:
    """Input contract for programmatic offline MP4 inference."""

    video_path: Path
    checkpoint_path: Path
    config_path: Path
    device: str | None = None


@dataclass(frozen=True)
class InferenceServiceResult:
    """Result object returned by the offline inference service."""

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


def run_offline_mp4_inference(request: InferenceServiceRequest) -> InferenceServiceResult:
    """Run offline MP4 inference and return typed in-memory results."""
    if not isinstance(request, InferenceServiceRequest):
        raise TypeError("request must be an InferenceServiceRequest instance")

    _validate_request(request)

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

    frame_count, inference_count, inference_results, _ = run_video(
        str(request.video_path),
        engine=engine,
        emit_runtime_summary=False,
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


def _validate_request(request: InferenceServiceRequest) -> None:
    """Validate request shape and path requirements."""
    if not isinstance(request.video_path, Path):
        raise TypeError("request.video_path must be a pathlib.Path instance")
    if not isinstance(request.checkpoint_path, Path):
        raise TypeError("request.checkpoint_path must be a pathlib.Path instance")
    if not isinstance(request.config_path, Path):
        raise TypeError("request.config_path must be a pathlib.Path instance")
    if request.device is not None and not isinstance(request.device, str):
        raise TypeError("request.device must be a string or None")

    if not request.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {request.video_path}")
    if not request.video_path.is_file():
        raise ValueError(f"Video path must point to a file: {request.video_path}")
    if request.video_path.suffix.lower() != ".mp4":
        raise ValueError("video_path must point to an .mp4 file")
