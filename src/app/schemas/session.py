"""Schemas for inference session REST endpoints."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FilePath


class SessionStatus(str, Enum):
    """Lifecycle states of an inference session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SessionStartRequest(BaseModel):
    """Payload for starting a new inference session."""

    video_path: Path = Field(
        ..., description="Absolute path to the input .mp4 video file."
    )
    checkpoint_path: FilePath = Field(
        ..., description="Absolute path to the model checkpoint file."
    )
    config_path: FilePath = Field(
        ..., description="Absolute path to the runtime configuration file (e.g., .yml)."
    )
    device: Optional[str] = Field(
        default=None, description="Hardware device to use (e.g., 'cuda:0', 'cpu')."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_path": "data/raw/sample.mp4",
                "checkpoint_path": "data/logs/checkpoints/baseline_epoch_10.pth",
                "config_path": "configs/data_pipeline.yml",
                "device": "cpu"
            }
        }
    )


class SessionResponse(BaseModel):
    """Response payload containing session details."""

    id: UUID = Field(..., description="Unique identifier of the session.")
    status: SessionStatus = Field(..., description="Current status of the session.")
    created_at: datetime = Field(..., description="When the session was created.")
    updated_at: datetime = Field(..., description="When the session status was last updated.")
    error: Optional[str] = Field(
        default=None, description="Error message if the session failed."
    )

    model_config = ConfigDict(from_attributes=True)
