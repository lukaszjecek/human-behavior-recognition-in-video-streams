"""Schemas for uploaded video API endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field


class VideoUploadResponse(BaseModel):
    """Response returned after storing an uploaded MP4."""

    video_id: UUID = Field(..., description="Stable uploaded video identifier.")
    original_filename: str = Field(..., description="Original browser-provided filename.")
    filename: str = Field(..., description="Generated server-side filename.")
    size_bytes: int = Field(..., description="Stored file size in bytes.")
