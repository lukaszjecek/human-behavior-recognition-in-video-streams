"""Schemas for inference session REST endpoints."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator


class SessionStatus(str, Enum):
    """Lifecycle states of an inference session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SessionStartRequest(BaseModel):
    """Payload for starting a new inference session."""

    video_path: Optional[Path] = Field(
        default=None,
        description="Backend-visible path to the input .mp4 video file.",
    )
    video_id: Optional[UUID] = Field(
        default=None,
        description="Stable ID returned by POST /api/videos/upload.",
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

    @model_validator(mode="after")
    def require_one_video_source(self) -> "SessionStartRequest":
        """Require either a backend path or an uploaded video ID, but not both."""
        if self.video_path is None and self.video_id is None:
            raise ValueError("Provide either video_path or video_id.")
        if self.video_path is not None and self.video_id is not None:
            raise ValueError("Provide either video_path or video_id, not both.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "b8bd3e7e-9539-42f2-a7d6-f6d7840b1f43",
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
