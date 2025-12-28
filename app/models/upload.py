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


class SignedUrlRequest(BaseModel):
    """Request for generating a signed upload URL."""

    user_id: str = Field(
        ...,
        description="User ID who will upload the file",
    )
    filename: str = Field(
        ...,
        description="Name of the file to be uploaded",
    )
    content_type: str = Field(
        default="video/mp4",
        description="Content type of the file (e.g., 'video/mp4', 'image/png')",
    )
    subfolder: str = Field(
        default="videos",
        description="Subfolder within user directory (e.g., 'videos', 'avatar', 'brand')",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "120accfe-aa23-41a3-b04f-36f581714d52",
                    "filename": "my-video.mp4",
                    "content_type": "video/mp4",
                    "subfolder": "videos",
                }
            ]
        }
    }


class SignedUrlResponse(BaseModel):
    """Response containing signed URL for upload."""

    signed_url: str = Field(
        ...,
        description="Signed URL for uploading file directly to GCS",
    )
    gcs_path: str = Field(
        ...,
        description="GCS path where the file will be stored (gs://bucket/path)",
    )
    expires_in_seconds: int = Field(
        default=3600,
        description="Number of seconds until the signed URL expires",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "signed_url": "https://storage.googleapis.com/clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4?X-Goog-Algorithm=...",
                    "gcs_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4",
                    "expires_in_seconds": 3600,
                }
            ]
        }
    }


class VideoUrlRequest(BaseModel):
    """Request for generating a signed URL to view/download a video."""

    gcs_path: str = Field(
        ...,
        description="GCS path of the video to access (gs://bucket/path or just the path)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gcs_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4"
                },
                {"gcs_path": "users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4"},
            ]
        }
    }


class VideoUrlResponse(BaseModel):
    """Response containing signed URL for viewing/downloading a video."""

    signed_url: str = Field(
        ...,
        description="Signed URL for viewing/downloading the video from GCS",
    )
    gcs_path: str = Field(
        ...,
        description="GCS path of the video (gs://bucket/path)",
    )
    expires_in_seconds: int = Field(
        default=3600,
        description="Number of seconds until the signed URL expires",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "signed_url": "https://storage.googleapis.com/clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4?X-Goog-Algorithm=...",
                    "gcs_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4",
                    "expires_in_seconds": 3600,
                }
            ]
        }
    }
