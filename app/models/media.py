"""Media source models for handling input files and content."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SourceMedia(BaseModel):
    """Video and/or still images provided by the creator."""

    video_path: Optional[str] = Field(
        None,
        description="URL or path to source video (https://storage.googleapis.com/... or local path)",
    )
    image_paths: list[str] = Field(
        default_factory=list,
        description="URLs or paths to existing frames, screenshots, or inspiration images",
    )

    @model_validator(mode="after")
    def ensure_media_present(self) -> "SourceMedia":
        """Require at least one media source."""
        if not self.video_path and not self.image_paths:
            raise ValueError("Provide at least one of video_path or image_paths.")
        return self
