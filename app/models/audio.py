"""Audio analysis and breakdown models."""

from typing import Optional

from pydantic import BaseModel, Field


class AudioBreakdownRequest(BaseModel):
    """Request for audio breakdown endpoint."""

    project_id: Optional[str] = Field(
        None,
        description="Optional project identifier. If not provided, a new UUID will be generated.",
        examples=["60a8336a-3d5d-45eb-a390-b52ab9f2dcb2"],
    )
    video_path: str = Field(
        ...,
        description="URL or path to video/audio file",
        examples=[
            "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4"
        ],
    )
    language: str = Field(
        "en",
        description="Language code for transcription (e.g., 'en', 'es', 'fr')",
        examples=["en", "es", "fr"],
    )
    max_duration_seconds: int = Field(
        600,
        description="Maximum duration to analyze in seconds (default: 600 = 10 minutes)",
        examples=[600, 300, 120],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "60a8336a-3d5d-45eb-a390-b52ab9f2dcb2",
                    "video_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
                    "language": "en",
                    "max_duration_seconds": 600,
                }
            ]
        }
    }


class AudioBreakdownResponse(BaseModel):
    """Response from audio breakdown endpoint."""

    project_id: str
    audio_path: str
    transcript: str
    duration_seconds: float
    speakers: list[dict]
    speech_tone: dict
    music_tone: dict
    timeline: list[dict]
