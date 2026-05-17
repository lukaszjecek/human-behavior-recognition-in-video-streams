"""Inference module for action detection and serialization."""

from src.inference.action_event import ActionEvent, ActionEventLog
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.service import (
    InferenceServiceRequest,
    InferenceServiceResult,
    run_inference,
    run_offline_mp4_inference,
)
from src.inference.session import InferenceSession, SessionStatus
from src.inference.source_adapters import (
    FileSourceAdapter,
    InferenceSourceAdapter,
    RtspSourceAdapter,
    build_source_adapter,
)

__all__ = [
    "ActionEvent",
    "ActionEventLog",
    "ActionEventWriter",
    "InferenceEngine",
    "InferenceResult",
    "InferenceServiceRequest",
    "InferenceServiceResult",
    "InferenceSourceAdapter",
    "FileSourceAdapter",
    "InferenceSession",
    "RtspSourceAdapter",
    "SessionStatus",
    "build_source_adapter",
    "run_inference",
    "run_offline_mp4_inference",
]
