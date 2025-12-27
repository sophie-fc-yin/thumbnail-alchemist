"""Video upload models."""

from typing import Literal

from pydantic import BaseModel, Field


class VideoUploadResponse(BaseModel):
    """Response from video upload endpoint."""

    status: Literal["success", "warning"]
    message: str
    gcs_path: str = Field(
        ...,
        description="GCS path where video was uploaded (gs://bucket/path)",
    )
    user_id: str = Field(
        ...,
        description="User ID who uploaded the file",
    )
    project_id: str | None = Field(
        None,
        description="Project ID if provided",
    )
    filename: str = Field(
        ...,
        description="Original filename",
    )
    file_size_mb: float = Field(
        ...,
        description="File size in megabytes",
    )
    overwritten: bool = Field(
        default=False,
        description="True if an existing file with the same name was overwritten",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Video uploaded successfully",
                    "gcs_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
                    "user_id": "120accfe-aa23-41a3-b04f-36f581714d52",
                    "project_id": None,
                    "filename": "1116_1_.mp4",
                    "file_size_mb": 125.4,
                    "overwritten": False,
                }
            ]
        }
    }


class VideoUploadError(BaseModel):
    """Error response from video upload endpoint."""

    error: str
    detail: str
    allowed_formats: list[str] = Field(
        default=["mp4", "mov", "avi", "mkv", "webm", "flv"],
        description="List of allowed video formats",
    )
    max_size_mb: int = Field(
        default=500,
        description="Maximum allowed file size in megabytes",
    )
