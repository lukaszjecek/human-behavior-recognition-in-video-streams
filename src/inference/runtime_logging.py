"""Structured logging helpers for inference runtime flows."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_DEFAULT_LEVEL = "INFO"
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


@dataclass(frozen=True)
class RuntimeLogContext:
    """Correlation fields shared across runtime log entries."""

    session_id: str
    source_type: str | None = None
    source_ref: str | None = None


class JsonLogFormatter(logging.Formatter):
    """Format log records as JSON for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        for key in ("session_id", "source_type", "source_ref"):
            value = getattr(record, key, None)
            if value:
                payload[key] = value

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if value is not None and key not in payload:
                    payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_runtime_logging(level: str | int | None = None) -> None:
    """Configure root logging for structured inference runtime output."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    resolved_level = _resolve_log_level(level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    context: RuntimeLogContext | None = None,
    *,
    exc_info: BaseException | bool | None = None,
    **fields: Any,
) -> None:
    """Log a structured runtime event with optional correlation context."""
    extra: dict[str, Any] = {"event": event}
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
    logger.log(level, message, extra=extra, exc_info=exc_info)


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
