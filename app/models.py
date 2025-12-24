from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


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


class Target(BaseModel):
    """Target platform and optimization settings."""

    platform: str = Field(
        "youtube",
        description="Platform/orientation to optimize for (e.g. YouTube, Shorts, Reels, TikTok).",
    )
    optimization: Optional[str] = Field(
        None, description="Primary KPI to optimize for, e.g. CTR or retention."
    )


class CreativeBrief(BaseModel):
    """Creative direction and brand guidelines for the thumbnail design."""

    mood: Optional[str] = Field(
        None,
        description="Desired tone or vibe, e.g. 'dramatic, high-contrast, cinematic'.",
        max_length=120,
    )
    title_hint: Optional[str] = Field(
        None, description="Optional working title or topic to steer the concept."
    )
    brand_colors: list[str] = Field(
        default_factory=list, description="Preferred colors as names or hex codes."
    )
    notes: Optional[str] = Field(
        None,
        description="Any extra creative guidance or constraints.",
        max_length=1000,
    )


class ThumbnailRequest(BaseModel):
    """Payload describing the creative intent for a thumbnail."""

    project_id: Optional[str] = Field(
        None,
        description="Optional project identifier. If not provided, a new UUID will be generated. Use the same project_id to overwrite previous frames.",
    )
    content_sources: SourceMedia = Field(
        ...,
        description="Video and images from the actual content being thumbnailed - provides context for background, composition, and theme.",
    )
    profile_photos: list[str] = Field(
        default_factory=list,
        description="URLs or paths to creator photos (https://storage.googleapis.com/... or local paths) to create the user avatar for the thumbnail design.",
    )
    target: Target = Field(
        default_factory=Target,
        description="Target platform and optimization settings.",
    )
    creative_brief: CreativeBrief = Field(
        default_factory=CreativeBrief,
        description="Creative direction and brand guidelines for the design.",
    )


class CompositionLayer(BaseModel):
    """Layer description for the composed video thumbnail."""

    kind: str = Field(..., description="Layer type, e.g. background, subject, title.")
    description: str = Field(..., description="What the layer contributes visually.")
    asset_url: Optional[HttpUrl] = Field(None, description="Static asset preview if available.")


class ThumbnailResponse(BaseModel):
    """Static demo response returned by the generate endpoint."""

    project_id: str
    status: Literal["draft", "final"]
    recommended_title: str
    thumbnail_url: HttpUrl
    selected_frame_url: Optional[HttpUrl]
    profile_variant_url: Optional[HttpUrl]
    layers: list[CompositionLayer]
    summary: str
