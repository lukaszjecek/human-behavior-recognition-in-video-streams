"""Application entrypoints for startup summary and MP4-to-JSON inference CLI."""

import argparse
import json
import sys
from os import getenv
from pathlib import Path
from typing import Sequence

import yaml

from src.inference.mp4_cli import (
    InferenceCliRequest,
    run_mp4_to_json_action_inference,
)

DATA_DIR = Path(getenv("DATA_DIR", "/app/data/raw"))
LOG_DIR = Path(getenv("LOG_DIR", "/app/data/logs"))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_startup_summary() -> int:
    """Scan dataset folder and write startup summary JSON."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"[ERROR] DATA_DIR does not exist: {DATA_DIR}")
        return 1

    video_files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    class_dirs = sorted({p.parent.name for p in video_files})

    summary = {
        "data_dir": str(DATA_DIR),
        "video_count": len(video_files),
        "class_count": len(class_dirs),
        "classes_preview": class_dirs[:20],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = LOG_DIR / "startup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote summary to: {summary_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for startup summary and Sprint 2 inference flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Run startup summary (default) or MP4-to-JSON action inference "
            "when --input and --checkpoint are provided."
        ),
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=None,
        help="Path to input .mp4 file for inference mode",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth) for inference mode",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="configs/data_pipeline.yml",
        help="Path to YAML config with runtime settings",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=str,
        default="data/logs/actions.json",
        help="Path where action inference JSON should be written",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Optional device override for inference (auto, cpu, cuda, mps)",
    )
    return parser


def _run_inference_mode(args: argparse.Namespace) -> int:
    """Execute MP4-to-JSON inference mode from parsed CLI args."""
    request = InferenceCliRequest(
        input_path=Path(args.input_path),
        checkpoint_path=Path(args.checkpoint_path),
        config_path=Path(args.config_path),
        output_path=Path(args.output_path),
        device=args.device,
    )

    try:
        return run_mp4_to_json_action_inference(request)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        yaml.YAMLError,
    ) as error:
        print(f"[ERROR] {error}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run app in startup-summary mode or MP4 inference mode."""
    parser = build_parser()
    parsed_args = parser.parse_args(list(argv) if argv is not None else [])

    has_input = parsed_args.input_path is not None
    has_checkpoint = parsed_args.checkpoint_path is not None

    if not has_input and not has_checkpoint:
        return run_startup_summary()

    if has_input != has_checkpoint:
        print("[ERROR] --input and --checkpoint must be provided together")
        return 2

    return _run_inference_mode(parsed_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
