from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceMedia(BaseModel):
    """Video and/or still images provided by the creator."""

    video_url: Optional[HttpUrl] = Field(
        None, description="Optional source video to analyze for candidate frames."
    )
    image_urls: list[HttpUrl] = Field(
        default_factory=list,
        description="Existing frames, screenshots, or inspiration images.",
    )


class ThumbnailRequest(BaseModel):
    """Payload describing the creative intent for a thumbnail."""

    sources: SourceMedia = Field(
        ...,
        description="Media to source the thumbnail components (backgrounds, frames, inspiration).",
    )
    profile_photo_url: Optional[HttpUrl] = Field(
        None,
        description="Creator photo (face, upper body, or full body) to blend into the design.",
    )
    target_platform: str = Field(
        "youtube",
        description="Platform/orientation to optimize for (e.g. YouTube, Shorts, Reels, TikTok).",
    )
    mood: Optional[str] = Field(
        None,
        description="Desired tone or vibe, e.g. 'dramatic, high-contrast, cinematic'.",
    )
    title_hint: Optional[str] = Field(
        None, description="Optional working title or topic to steer the concept."
    )
    goal: Optional[str] = Field(
        None, description="Primary KPI to optimize for, e.g. CTR or retention."
    )
    brand_colors: list[str] = Field(
        default_factory=list, description="Preferred colors as names or hex codes."
    )
    notes: Optional[str] = Field(None, description="Any extra creative guidance or constraints.")


class CompositionLayer(BaseModel):
    """Simplified layer description for the composed thumbnail."""

    kind: str = Field(..., description="Layer type, e.g. background, subject, title.")
    description: str = Field(..., description="What the layer contributes visually.")
    asset_url: Optional[HttpUrl] = Field(None, description="Static asset preview if available.")


class ThumbnailResponse(BaseModel):
    """Static demo response returned by the generate endpoint."""

    request_id: str
    status: Literal["draft", "final"]
    recommended_title: str
    thumbnail_url: HttpUrl
    selected_frame_url: Optional[HttpUrl]
    profile_variant_url: Optional[HttpUrl]
    layers: list[CompositionLayer]
    summary: str
