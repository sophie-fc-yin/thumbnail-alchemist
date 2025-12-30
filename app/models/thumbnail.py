"""Thumbnail generation and composition models."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from app.models.media import SourceMedia


class Target(BaseModel):
    """Target platform and optimization settings."""

    platform: str = Field(
        "youtube",
        description="Platform/orientation to optimize for (e.g. YouTube, Shorts, Reels, TikTok).",
    )
    optimization: Optional[str] = Field(
        None, description="Primary KPI to optimize for, e.g. CTR or retention."
    )
    audience_profile: Optional[str] = Field(
        None,
        description="Target audience description (e.g., 'tech-savvy millennials', 'beginner programmers', 'gaming enthusiasts ages 18-25'). Helps tailor thumbnail style and appeal.",
        max_length=200,
    )


class ChannelProfile(BaseModel):
    """Information about the creator's channel and growth stage."""

    stage: Optional[str] = Field(
        None,
        description="Channel growth stage (e.g., 'new/starter', 'growing', 'established', 'large/mainstream'). Affects thumbnail strategy.",
        max_length=50,
    )
    subscriber_count: Optional[int] = Field(
        None,
        description="Approximate subscriber count (helps determine appropriate thumbnail approach).",
        ge=0,
    )
    content_niche: Optional[str] = Field(
        None,
        description=(
            "Primary content category or niche. Affects scoring weights, aesthetic criteria, and advisory guidance. "
            "Supported niches: 'gaming', 'tech'/'tech reviews'/'educational', 'beauty'/'lifestyle', "
            "'commentary'/'reaction', 'cooking'/'food', 'fitness'/'health', 'business'/'finance', "
            "'entertainment'/'comedy', 'music', 'news'/'journalism'. "
            "Defaults to 'general' if not specified."
        ),
        max_length=100,
    )
    upload_frequency: Optional[str] = Field(
        None,
        description="How often content is published (e.g., 'daily', 'weekly', 'monthly'). Affects consistency expectations.",
        max_length=50,
    )
    growth_goal: Optional[str] = Field(
        None,
        description="Primary growth objective (e.g., 'build authority', 'viral reach', 'loyal community', 'monetization').",
        max_length=100,
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
    channel_profile: ChannelProfile = Field(
        default_factory=ChannelProfile,
        description="Information about the creator's channel stage, niche, and growth goals.",
    )
    creative_brief: CreativeBrief = Field(
        default_factory=CreativeBrief,
        description="Creative direction and brand guidelines for the design.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "60a8336a-3d5d-45eb-a390-b52ab9f2dcb2",
                    "content_sources": {
                        "video_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
                    },
                    "profile_photos": [
                        "https://storage.cloud.google.com/clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/avatar/headshot.jpg"
                    ],
                    "target": {
                        "platform": "youtube",
                        "optimization": "CTR",
                        "audience_profile": "tech-savvy developers ages 25-40",
                    },
                    "channel_profile": {
                        "stage": "growing",
                        "subscriber_count": 50000,
                        "content_niche": "coding tutorials",
                        "upload_frequency": "weekly",
                        "growth_goal": "build authority",
                    },
                    "creative_brief": {
                        "mood": "professional, clean, modern",
                        "title_hint": "Learn React Hooks in 10 Minutes",
                        "brand_colors": ["#61DAFB", "#282C34"],
                        "notes": "Keep it simple and code-focused",
                    },
                }
            ]
        }
    }


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
