"""Structured logging helpers for inference runtime flows."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_LEVEL = "INFO"
_DEFAULT_DETAIL = "standard"
_STRUCTURED_LOGGER_NAME = "hbr.structured"
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
_DETAIL_LEVELS = {"MINIMAL", "STANDARD", "VERBOSE"}
_MINIMAL_FIELDS = {
    "error_type",
    "phase",
    "frames_before_failure",
}
_STANDARD_FIELDS = _MINIMAL_FIELDS | {
    "attempt",
    "app_name",
    "app_version",
    "backoff_factor",
    "build_sha",
    "class_label_count",
    "checkpoint_path",
    "config_path",
    "debug",
    "delay_s",
    "device",
    "device_request",
    "duration_s",
    "event_count",
    "frame_count",
    "frames_read",
    "http_method",
    "http_path",
    "image_tag",
    "inference_count",
    "input_path",
    "max_retries",
    "output_path",
    "retry_delay",
    "source_ref",
    "status_code",
    "target_resolution",
    "total_frames_processed",
    "total_frames_skipped",
    "total_inferences",
    "window_size",
    "stride",
    "ws_path",
}


@dataclass(frozen=True)
class RuntimeLogContext:
    """Correlation fields shared across runtime log entries."""

    session_id: str
    source_type: str | None = None
    source_ref: str | None = None


class JsonLogFormatter(logging.Formatter):
    """Format log records as JSON for structured ingestion."""

    def __init__(self, detail: str) -> None:
        """Initialize the JSON log formatter with a detail level."""
        super().__init__()
        self._detail = detail

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string for structured logging."""
        logger_name = getattr(record, "logger_name", None) or record.name
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": logger_name,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        for key in ("session_id", "source_type"):
            value = getattr(record, key, None)
            if value:
                payload[key] = value
        if self._detail != "minimal":
            source_ref = getattr(record, "source_ref", None)
            if source_ref:
                payload["source_ref"] = source_ref

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if value is None or key in payload:
                    continue
                if self._detail == "verbose":
                    payload[key] = value
                elif self._detail == "standard" and key in _STANDARD_FIELDS:
                    payload[key] = value
                elif self._detail == "minimal" and key in _MINIMAL_FIELDS:
                    payload[key] = value

        if record.exc_info:
            exc = record.exc_info[1]
            if self._detail == "verbose":
                payload["exception"] = self.formatException(record.exc_info)
            elif self._detail == "standard":
                payload["exception"] = str(exc)
            else:
                payload["exception_type"] = type(exc).__name__

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_runtime_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Configure structured logging for inference runtime output."""
    _get_structured_logger(level=level, log_file=log_file)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    context: RuntimeLogContext | None = None,
    *,
    exc_info: BaseException | bool | None = None,
    **fields: object,
) -> None:
    """Log a structured runtime event with optional correlation context."""
    extra: dict[str, object] = {"event": event}
    extra["logger_name"] = logger.name
    if context is not None:
        extra.update(
            {
                "session_id": context.session_id,
                "source_type": context.source_type,
                "source_ref": context.source_ref,
            }
        )
    if fields:
        extra["fields"] = fields
    structured_logger = _get_structured_logger()
    structured_logger.log(level, message, extra=extra, exc_info=exc_info)


def get_build_metadata() -> dict[str, str]:
    """Return optional build metadata for correlation in logs."""
    build_sha = os.getenv("BUILD_SHA") or os.getenv("GIT_SHA")
    image_tag = os.getenv("IMAGE_TAG")
    app_version = os.getenv("APP_VERSION")
    metadata: dict[str, str] = {}
    if build_sha:
        metadata["build_sha"] = build_sha
    if image_tag:
        metadata["image_tag"] = image_tag
    if app_version:
        metadata["app_version"] = app_version
    return metadata


def _resolve_log_level(level: str | int | None) -> int:
    """Resolve a logging level from explicit value or environment."""
    if level is None:
        level = os.getenv("INFERENCE_LOG_LEVEL") or os.getenv("LOG_LEVEL") or _DEFAULT_LEVEL

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized in _LEVELS:
            return _LEVELS[normalized]

    return logging.INFO


def _resolve_log_detail() -> str:
    """Resolve log detail level from environment."""
    detail = os.getenv("INFERENCE_LOG_DETAIL") or os.getenv("LOG_DETAIL") or _DEFAULT_DETAIL
    if isinstance(detail, str):
        normalized = detail.strip().upper()
        if normalized in _DETAIL_LEVELS:
            return normalized.lower()
    return _DEFAULT_DETAIL


def _get_structured_logger(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a dedicated structured logger configured with JSON output."""
    logger = logging.getLogger(_STRUCTURED_LOGGER_NAME)
    resolved_level = _resolve_log_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Check if a StreamHandler is already registered
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    detail = _resolve_log_detail()
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter(detail))
        logger.addHandler(handler)

    if log_file:
        log_dir_env = os.getenv("LOG_DIR")
        if log_dir_env:
            log_file_path = Path(log_dir_env) / log_file
        else:
            log_file_path = Path(log_file)

        # Check if file handler for this path already exists
        has_file_handler = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if Path(h.baseFilename).resolve() == log_file_path.resolve():
                        has_file_handler = True
                        break
                except Exception:
                    pass

        if not has_file_handler:
            try:
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
                file_handler.setFormatter(JsonLogFormatter(detail))
                logger.addHandler(file_handler)
            except Exception as e:
                import sys
                print(f"Error configuring FileHandler for structured logger: {e}", file=sys.stderr)

    return logger


def log_audit_event(payload: object) -> None:
    """Write a serialized EventPayload to a dedicated audit.log file."""
    log_dir = os.getenv("LOG_DIR") or "data/logs"
    try:
        audit_path = Path(log_dir) / "audit.log"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            data = payload.model_dump(mode="json")
        else:
            data = payload
        line = json.dumps(data, ensure_ascii=False)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        structured_logger = logging.getLogger(_STRUCTURED_LOGGER_NAME)
        structured_logger.error(f"Failed to write event to audit log: {e}")
