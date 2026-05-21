"""MP4-to-JSON action inference CLI helpers."""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.inference import runtime as runtime_primitives
from src.inference.context_adapter import ContextModule
from src.inference.json_writer import ActionEventWriter
from src.inference.service import InferenceServiceRequest, run_offline_mp4_inference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceCliRequest:
    """Input contract for MP4-to-JSON action inference."""

    input_path: Path
    checkpoint_path: Path
    config_path: Path
    output_path: Path
    device: str | None = None


def run_mp4_to_json_action_inference(request: InferenceCliRequest) -> int:
    """Run end-to-end MP4 inference and save ActionEvent log as JSON."""
    if not isinstance(request, InferenceCliRequest):
        raise TypeError("request must be an InferenceCliRequest instance")
    _validate_request_paths(request)
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".webp"}
    if request.input_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            "input_path must point to a supported video file "
            f"({', '.join(sorted(allowed_extensions))})"
        )

    service_result = run_offline_mp4_inference(
        InferenceServiceRequest(
            video_path=request.input_path,
            checkpoint_path=request.checkpoint_path,
            config_path=request.config_path,
            device=request.device,
        )
    )

    context_data = {"scene_tag": "unknown", "confidence": 0.0}
    context_module = _try_create_context_module()

    if context_module and service_result.inference_results:
        try:
            first_frame = service_result.inference_results[0].window[0]
            context_data = context_module.get_context(first_frame)
        except (IndexError, TypeError, RuntimeError, ValueError) as error:
            logger.error("Context inference failed, using fallback: %s", error)

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ActionEventWriter(
        class_labels=service_result.runtime_settings.class_labels,
    )
    for event in service_result.action_events:
        event.context = context_data
    writer.get_log().add_events(list(service_result.action_events))
    writer.save(str(request.output_path))
    logger.info(
        "[OK] Wrote %d action events to: %s",
        service_result.event_count,
        request.output_path,
    )

    return 0


def _validate_request_paths(request: InferenceCliRequest) -> None:
    """Validate that request fields are Path objects."""
    if not isinstance(request.input_path, Path):
        raise TypeError("request.input_path must be a pathlib.Path instance")
    if not isinstance(request.checkpoint_path, Path):
        raise TypeError("request.checkpoint_path must be a pathlib.Path instance")
    if not isinstance(request.config_path, Path):
        raise TypeError("request.config_path must be a pathlib.Path instance")
    if not isinstance(request.output_path, Path):
        raise TypeError("request.output_path must be a pathlib.Path instance")
    if request.device is not None and not isinstance(request.device, str):
        raise TypeError("request.device must be a string or None")


def _try_create_context_module() -> ContextModule | None:
    """Initialize ContextModule with explicit fallback for unavailable runtime deps."""
    try:
        return ContextModule()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.warning("ContextModule failed to initialize: %s", error)
        return None


# Backward-compatible private alias while shared runtime helper is now public.
InferenceRuntimeSettings = runtime_primitives.InferenceRuntimeSettings
WindowModelAdapter = runtime_primitives.WindowModelAdapter
build_track_ids = runtime_primitives.build_track_ids
expand_batched_inference_results = runtime_primitives.expand_batched_inference_results
load_model_from_checkpoint = runtime_primitives.load_model_from_checkpoint
load_runtime_settings = runtime_primitives.load_runtime_settings
resolve_inference_device = runtime_primitives.resolve_inference_device
_expand_batched_inference_results = expand_batched_inference_results
