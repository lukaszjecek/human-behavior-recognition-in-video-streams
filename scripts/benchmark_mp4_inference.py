"""Measure MP4 inference wall-clock performance and write a JSON summary."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.inference.service import InferenceServiceRequest, run_offline_mp4_inference  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the MP4 benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run MP4 inference once and write measured performance metrics as JSON.",
    )
    parser.add_argument(
        "--input",
        required=True,
        dest="input_path",
        help="Path to the input MP4 video.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        dest="checkpoint_path",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/data_pipeline.yml",
        dest="config_path",
        help="Path to the inference/data pipeline YAML config.",
    )
    parser.add_argument(
        "--output",
        default="data/logs/benchmark_summary.json",
        dest="output_path",
        help="Path where the benchmark summary JSON should be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Requested inference device.",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run offline MP4 inference and return measured benchmark data."""
    input_path = Path(args.input_path)
    checkpoint_path = Path(args.checkpoint_path)
    config_path = Path(args.config_path)

    _validate_existing_file(input_path, "--input")
    _validate_existing_file(checkpoint_path, "--checkpoint")
    _validate_existing_file(config_path, "--config")
    if input_path.suffix.lower() != ".mp4":
        raise ValueError(f"--input must point to an MP4 file: {input_path}")

    request = InferenceServiceRequest(
        video_path=input_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device,
    )

    start_time = time.perf_counter()
    result = run_offline_mp4_inference(request)
    total_duration_s = time.perf_counter() - start_time

    frame_count = result.frame_count
    inference_count = result.inference_count
    event_count = result.event_count

    processed_fps = _safe_rate(frame_count, total_duration_s)
    inference_windows_per_second = _safe_rate(inference_count, total_duration_s)
    average_time_per_inference_window_s = (
        total_duration_s / inference_count if inference_count else None
    )

    settings = result.runtime_settings
    return {
        "timestamp": _utc_timestamp(),
        "host": _host_info(),
        "python_version": platform.python_version(),
        "cpu": _cpu_info(),
        "cuda": _cuda_info(),
        "requested_device": args.device,
        "resolved_runtime_device": str(result.resolved_device),
        "input_video_path": str(input_path),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "window_size": settings.window_size,
        "stride": settings.stride,
        "target_resolution": list(settings.target_resolution),
        "frame_count": frame_count,
        "inference_count": inference_count,
        "event_count": event_count,
        "total_wall_clock_duration_s": _round_float(total_duration_s),
        "approx_processed_fps": _round_float(processed_fps),
        "approx_inference_windows_per_second": _round_float(
            inference_windows_per_second,
        ),
        "average_time_per_inference_window_s": _round_float(
            average_time_per_inference_window_s,
        ),
        "limitations": [
            "Single-run offline MP4 wall-clock measurement; "
            "repeat runs before using as final data.",
            "Includes model loading, video decode, preprocessing, inference, "
            "and event pipeline work.",
            "Does not measure live camera capture-to-display latency.",
            "Current checkpoint should not be treated as final until issue #130 publishes it.",
        ],
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Write benchmark summary JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_existing_file(path: Path, argument_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{argument_name} file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{argument_name} must point to a file: {path}")


def _safe_rate(count: int, duration_s: float) -> float | None:
    if duration_s <= 0:
        return None
    return count / duration_s


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _host_info() -> dict[str, str]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "platform": platform.platform(),
    }


def _cpu_info() -> dict[str, Any]:
    return {
        "processor": platform.processor(),
        "model_name": _linux_cpu_model_name(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
    }


def _linux_cpu_model_name() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None

    try:
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_available": False,
            "cuda_available": False,
            "notes": "torch import failed; CUDA availability was not checked.",
        }

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = [
        torch.cuda.get_device_name(index)
        for index in range(cuda_device_count)
    ]
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": device_names,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output_path)

    try:
        summary = run_benchmark(args)
        write_summary(summary, output_path)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    print("MP4 inference benchmark completed.")
    print(f"Output JSON: {output_path}")
    print(f"Frames: {summary['frame_count']}")
    print(f"Inference windows: {summary['inference_count']}")
    print(f"Action events: {summary['event_count']}")
    print(f"Wall-clock duration: {summary['total_wall_clock_duration_s']} s")
    print(f"Approx processed FPS: {summary['approx_processed_fps']}")
    print(
        "Approx inference windows/s: "
        f"{summary['approx_inference_windows_per_second']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
