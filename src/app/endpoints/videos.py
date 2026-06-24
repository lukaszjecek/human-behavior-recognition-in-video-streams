"""Implementation of uploaded video REST endpoints."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.requests import Request

from src.app.schemas.video import VideoUploadResponse

router = APIRouter()


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload MP4 Video",
)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
) -> VideoUploadResponse:
    """Store an operator-selected MP4 under a generated server-side name."""
    original_filename = Path(file.filename or "").name
    if Path(original_filename).suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .mp4 uploads are supported.",
        )

    app_settings = getattr(request.app.state, "settings", None)
    upload_dir = getattr(app_settings, "upload_dir", Path("/app/data/uploads"))
    upload_root = upload_dir.resolve()
    upload_root.mkdir(parents=True, exist_ok=True)

    video_id = uuid4()
    filename = f"{video_id}.mp4"
    target_path = (upload_root / filename).resolve(strict=False)
    try:
        target_path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload target resolves outside upload directory.",
        ) from exc

    size_bytes = 0
    try:
        with target_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                destination.write(chunk)
    except OSError as exc:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded video.",
        ) from exc
    finally:
        await file.close()

    return VideoUploadResponse(
        video_id=video_id,
        original_filename=original_filename,
        filename=filename,
        size_bytes=size_bytes,
    )
