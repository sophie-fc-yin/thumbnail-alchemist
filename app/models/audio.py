"""Audio analysis and breakdown models."""

from typing import Optional

from pydantic import BaseModel, Field

from app.constants import DEFAULT_MAX_DURATION_SECONDS


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
    language: Optional[str] = Field(
        None,
        description="Language code for transcription (e.g., 'en', 'es', 'fr'). If None, OpenAI will auto-detect the language.",
        examples=["en", "es", "fr", None],
    )
    max_duration_seconds: int = Field(
        DEFAULT_MAX_DURATION_SECONDS,
        description=f"Maximum duration to analyze in seconds (default: {DEFAULT_MAX_DURATION_SECONDS} = 30 minutes)",
        examples=[1800, 600, 300],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "60a8336a-3d5d-45eb-a390-b52ab9f2dcb2",
                    "video_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
                    "language": "en",
                    "max_duration_seconds": DEFAULT_MAX_DURATION_SECONDS,
                }
            ]
        }
    }


class AudioBreakdownResponse(BaseModel):
    """Response from audio breakdown endpoint."""

    project_id: str
    audio_path: str = Field(
        ...,
        description="GCS URL to speech audio file (empty string if no speech detected)",
    )
    transcript: str = Field(
        ...,
        description="Full text transcript of speech",
    )
    duration_seconds: float = Field(
        ...,
        description="Total duration of audio in seconds",
    )
    stream_a_results: list[dict] | None = Field(
        None,
        description="Stream A: Speech semantics analysis - list of segments with tone, emotion, narrative context, and importance",
    )
    stream_b_results: list[dict] | None = Field(
        None,
        description="Stream B: Audio saliency analysis - list of segments with saliency features (energy peaks, spectral changes, etc.)",
    )
    audio_features: list[dict] | None = Field(
        None,
        description="Raw audio features - list of segments with pitch, energy, spectral features",
    )
