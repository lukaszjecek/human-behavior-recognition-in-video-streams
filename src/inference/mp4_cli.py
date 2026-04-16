"""MP4-to-JSON action inference CLI helpers."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.offline_runtime import run_video
from src.inference.tensorize import FrameTensorizer
from src.models.baseline import BaselineBehaviorModel
from src.models.dummy import DummyBehaviorModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceCliRequest:
    """Input contract for MP4-to-JSON action inference."""

    input_path: Path
    checkpoint_path: Path
    config_path: Path
    output_path: Path
    device: str | None = None


@dataclass(frozen=True)
class InferenceRuntimeSettings:
    """Typed runtime settings parsed from YAML config."""

    target_resolution: tuple[int, int]
    window_size: int
    stride: int
    class_labels: list[str]
    default_track_id: int | None
    device: str | None


class WindowModelAdapter:
    """Adapter that tensorizes frame windows before calling a torch model."""

    def __init__(
        self,
        model: torch.nn.Module,  # is model based on torch.nn.Module?
        tensorizer: FrameTensorizer,
        device: torch.device,
    ) -> None:
        """Initialize the adapter with a model, tensorizer, and target device."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module instance")
        if not isinstance(tensorizer, FrameTensorizer):
            raise TypeError("tensorizer must be a FrameTensorizer instance")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device instance")

        self._model = model
        self._tensorizer = tensorizer
        self._device = device

    def __call__(self, window: tuple[Any, ...]) -> torch.Tensor:
        """Run model inference on a frame window and return logits/probabilities."""
        if not isinstance(window, tuple):
            raise TypeError("window must be a tuple of frames")
        if len(window) == 0:
            raise ValueError("window cannot be empty")

        tensor = self._tensorizer.tensorize(list(window)).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)

        if not isinstance(output, torch.Tensor):
            raise TypeError("model output must be a torch.Tensor")

        if output.ndim == 1:
            return output.detach().cpu()

        if output.ndim == 2:  # warging: in the future, there will be bug caused by ignoring other batch
            if output.shape[0] < 1:
                raise ValueError(
                    "model output batch dimension must not be empty")
            return output.detach().cpu()

        raise ValueError("model output tensor must be 1D or 2D")


def run_mp4_to_json_action_inference(request: InferenceCliRequest) -> int:
    """Run end-to-end MP4 inference and save ActionEvent log as JSON."""
    if not isinstance(request, InferenceCliRequest):
        raise TypeError("request must be an InferenceCliRequest instance")
    _validate_request_paths(request)
    if request.input_path.suffix.lower() != ".mp4":
        raise ValueError("input_path must point to an .mp4 file")

    settings = load_runtime_settings(request.config_path)
    device = resolve_inference_device(
        cli_device=request.device,
        config_device=settings.device,
    )
    model = load_model_from_checkpoint(request.checkpoint_path, device)

    tensorizer = FrameTensorizer(target_resolution=settings.target_resolution)
    model_adapter = WindowModelAdapter(
        model=model, tensorizer=tensorizer, device=device)

    # initializing of engine in this place is more flexible than in the run_video()
    engine = InferenceEngine(
        window_size=settings.window_size,
        stride=settings.stride,
        model=model_adapter,
    )

    _, _, inference_results = run_video(str(request.input_path), engine=engine)
    inference_results = _expand_batched_inference_results(inference_results)
    track_ids = build_track_ids(inference_results, settings.default_track_id)

    writer = ActionEventWriter(class_labels=settings.class_labels)
    writer.add_results(inference_results, track_ids=track_ids)

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer.save(str(request.output_path))
    logger.info(
        "[OK] Wrote %d action events to: %s",
        len(writer.get_log().events),
        request.output_path,
    )

    return 0


def load_runtime_settings(config_path: Path) -> InferenceRuntimeSettings:
    """Load and validate inference runtime settings from YAML config."""
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be a pathlib.Path instance")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        # found but it is not a file
        raise ValueError(f"Config path must point to a file: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise TypeError("Config root must be a mapping/object")

    pipeline_cfg = _ensure_mapping(raw_config.get("pipeline", {}), "pipeline")
    inference_cfg = _ensure_mapping(
        raw_config.get("inference", {}), "inference")
    tracking_cfg = _ensure_mapping(raw_config.get("tracking", {}), "tracking")

    target_resolution = _parse_target_resolution(
        pipeline_cfg.get("target_resolution", (224, 224)),
    )
    window_size = _parse_positive_int(
        pipeline_cfg.get("temporal_window", 16),
        "pipeline.temporal_window",
    )
    stride = _parse_positive_int(
        inference_cfg.get("stride", 1),
        "inference.stride",
    )
    class_labels = _parse_class_labels(inference_cfg.get("class_labels"))
    default_track_id = _parse_optional_track_id(
        tracking_cfg.get("default_track_id"))
    device = _parse_optional_device(inference_cfg.get("device"), "inference.device")

    return InferenceRuntimeSettings(
        target_resolution=target_resolution,
        window_size=window_size,
        stride=stride,
        class_labels=class_labels,
        default_track_id=default_track_id,
        device=device,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a model from checkpoint and move it to the target device."""
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path instance")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device instance")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(
            f"Checkpoint path must point to a file: {checkpoint_path}")

    raw_checkpoint = torch.load(
        # against Arbitrary Code Execution
        str(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(raw_checkpoint, dict):
        raise TypeError("Checkpoint must contain a mapping/object payload")

    raw_state_dict = raw_checkpoint.get("model_state_dict", raw_checkpoint)
    state_dict = _validate_state_dict(raw_state_dict)
    num_classes = _parse_num_classes(raw_checkpoint.get("num_classes"))

    model_name = raw_checkpoint.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise TypeError(
            "model_name in checkpoint must be a string when provided")

    candidates = _resolve_model_candidates(model_name)
    errors: list[str] = []
    for candidate in candidates:
        model = _build_model(candidate, num_classes)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as runtime_error:
            errors.append(f"{candidate}: {runtime_error}")
            continue

        model.to(device)
        model.eval()
        return model

    joined_errors = "; ".join(errors)
    raise ValueError(
        "Could not load checkpoint into a supported model architecture. "
        f"Attempted: {', '.join(candidates)}. Details: {joined_errors}",
    )


def build_track_ids(
    results: list[InferenceResult],
    default_track_id: int | None,
) -> list[int | None]:
    """Build per-result tracking IDs used by ActionEventWriter."""
    if not isinstance(results, list):
        raise TypeError("results must be a list of InferenceResult objects")
    if default_track_id is not None and (
        not isinstance(default_track_id, int) or isinstance(
            default_track_id, bool)
    ):
        raise TypeError("default_track_id must be an integer or None")
    if isinstance(default_track_id, int) and default_track_id < 0:
        raise ValueError("default_track_id must be >= 0")

    for result in results:
        if not isinstance(result, InferenceResult):
            raise TypeError(
                "results must only contain InferenceResult objects")

    if default_track_id is None:
        return [None] * len(results)
    return [default_track_id] * len(results)


def _expand_batched_inference_results(
    results: list[InferenceResult],
) -> list[InferenceResult]:
    """Expand 2D tensor predictions (batch, classes) into per-item results."""
    expanded_results: list[InferenceResult] = []
    for result in results:
        prediction = result.prediction
        if isinstance(prediction, torch.Tensor) and prediction.ndim == 2:
            if prediction.shape[0] < 1:
                raise ValueError("model output batch dimension must not be empty")
            for item_prediction in prediction:
                expanded_results.append(
                    InferenceResult(
                        window=result.window,
                        start_frame_index=result.start_frame_index,
                        end_frame_index=result.end_frame_index,
                        start_timestamp=result.start_timestamp,
                        end_timestamp=result.end_timestamp,
                        prediction=item_prediction,
                    )
                )
            continue
        expanded_results.append(result)
    return expanded_results


def _ensure_mapping(value: object, field_name: str) -> dict[str, Any]:
    """Ensure value is a dictionary-like mapping represented as dict."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping/object")
    return value


def _validate_request_paths(request: InferenceCliRequest) -> None:
    """Validate that request fields are Path objects."""
    if not isinstance(request.input_path, Path):
        raise TypeError("request.input_path must be a pathlib.Path instance")
    if not isinstance(request.checkpoint_path, Path):
        raise TypeError(
            "request.checkpoint_path must be a pathlib.Path instance")
    if not isinstance(request.config_path, Path):
        raise TypeError("request.config_path must be a pathlib.Path instance")
    if not isinstance(request.output_path, Path):
        raise TypeError("request.output_path must be a pathlib.Path instance")
    if request.device is not None and not isinstance(request.device, str):
        raise TypeError("request.device must be a string or None")


def _parse_positive_int(value: object, field_name: str) -> int:
    """Parse a positive integer while rejecting booleans."""
    if not isinstance(value, int) or isinstance(value, bool):  # bool inherit from int
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _parse_target_resolution(value: object) -> tuple[int, int]:
    """Parse and validate target resolution as (width, height)."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "pipeline.target_resolution must be a list/tuple of two integers")
    if len(value) != 2:
        raise ValueError(
            "pipeline.target_resolution must contain exactly 2 values")

    width = _parse_positive_int(value[0], "pipeline.target_resolution[0]")
    height = _parse_positive_int(value[1], "pipeline.target_resolution[1]")
    return (width, height)


def _parse_class_labels(value: object) -> list[str]:
    """Parse optional class labels list."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("inference.class_labels must be a list of strings")

    class_labels: list[str] = []
    for label in value:
        if not isinstance(label, str):
            raise TypeError("inference.class_labels must contain only strings")
        if not label.strip():
            raise ValueError(
                "inference.class_labels must not contain empty strings")
        class_labels.append(label)
    return class_labels


def _parse_optional_track_id(value: object) -> int | None:
    """Parse optional default track ID."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("tracking.default_track_id must be an integer or null")
    if value < 0:
        raise ValueError("tracking.default_track_id must be >= 0")
    return value


def _parse_optional_device(value: object, field_name: str) -> str | None:
    """Parse optional inference device override."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be one of: auto, cpu, cuda, mps")
    normalized = value.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"{field_name} must be one of: auto, cpu, cuda, mps")
    return normalized


def resolve_inference_device(
    cli_device: str | None,
    config_device: str | None,
) -> torch.device:
    """Resolve torch device from CLI/config overrides and available backends."""
    cli_choice = _parse_optional_device(cli_device, "request.device")
    config_choice = _parse_optional_device(config_device, "inference.device")
    preferred_device = cli_choice if cli_choice is not None else config_choice

    if preferred_device in {None, "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred_device == "cpu":
        return torch.device("cpu")

    if preferred_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but not available on this machine",
            )
        return torch.device("cuda")

    if preferred_device == "mps":
        if not _is_mps_available():
            raise ValueError(
                "MPS device requested but not available on this machine",
            )
        return torch.device("mps")

    raise ValueError(f"Unsupported device selection: {preferred_device}")


def _is_mps_available() -> bool:
    """Return True when torch MPS backend is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not hasattr(mps_backend, "is_available"):
        return False
    return bool(mps_backend.is_available())


def _validate_state_dict(value: object) -> dict[str, torch.Tensor]:
    """Validate and normalize checkpoint state_dict."""
    if not isinstance(value, dict):
        raise TypeError("model_state_dict must be a mapping/object")
    if len(value) == 0:
        raise ValueError("model_state_dict must not be empty")

    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str):
            raise TypeError("model_state_dict keys must be strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "model_state_dict values must be torch.Tensor objects")
        normalized[key] = tensor
    return normalized


def _parse_num_classes(value: object) -> int:
    """Parse model class count from checkpoint metadata."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("num_classes in checkpoint must be an integer")
    if value <= 0:
        raise ValueError("num_classes in checkpoint must be > 0")
    return value


def _resolve_model_candidates(model_name: str | None) -> list[str]:
    """Resolve model loading order from optional model_name metadata."""
    if model_name is None:
        return ["baseline", "dummy"]

    normalized_name = model_name.strip().lower()
    if normalized_name not in {"baseline", "dummy"}:
        raise ValueError("model_name must be either 'baseline' or 'dummy'")
    return [normalized_name]


def _build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Build a supported model instance by name."""
    if model_name == "baseline":
        return BaselineBehaviorModel(num_classes=num_classes)
    if model_name == "dummy":
        return DummyBehaviorModel(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {model_name}")
