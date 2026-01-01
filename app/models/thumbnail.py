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
                        "optimization": "subscriber growth",
                        "audience_profile": "travel enthusiasts and people curious about New Orleans, ages 20-45",
                    },
                    "channel_profile": {
                        "stage": "new/starter",
                        "subscriber_count": 0,
                        "content_niche": "travel",
                        "upload_frequency": "as I travel",
                        "growth_goal": "build community",
                    },
                    "creative_brief": {
                        "mood": "casual, authentic, chill with energetic moments",
                        "title_hint": "Exploring New Orleans for the First Time | What to See & Do",
                        "brand_colors": [],
                        "notes": "Genuine reactions, mix of calm sightseeing and fun event moments. Show the real NOLA experience - food, culture, music, and hidden spots.",
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


class FrameOption(BaseModel):
    """A strategic frame option with reasoning."""

    frame_id: str = Field(..., description="Frame identifier (e.g., 'Frame 3')")
    frame_number: int = Field(..., description="Frame index number")
    timestamp: str = Field(..., description="Timestamp in video (e.g., '5.2s')")
    frame_url: Optional[HttpUrl] = Field(None, description="URL to frame image")
    one_liner: str = Field(..., description="Short explanation (max 18 words)")
    reasons: list[str] = Field(..., description="2 specific reasons for this choice")
    risk_notes: list[str] = Field(
        default_factory=list, description="Optional considerations (0-2 items)"
    )


class FrameDebugInfo(BaseModel):
    """Debug scoring information for a frame."""

    frame_id: str
    timestamp: str
    total_score: float
    moment_importance: float = Field(
        0.0,
        description="Phase-1 moment importance score derived from adaptive sampling (audio+visual saliency).",
    )
    aesthetic: float
    psychology: float
    editability: float
    face_quality: float
    creator_alignment: float
    emotion: str
    expression_intensity: float
    triggers: list[str]
    why_chosen_or_not: str


class AdvisoryMeta(BaseModel):
    """Metadata about the advisory decision."""

    confidence: Literal["low", "medium", "high"]
    what_changed: str = Field(..., description="Strategic differences between options")
    user_control_note: str = Field(..., description="Supportive reminder for creator")


class ThumbnailAdvisory(BaseModel):
    """AI advisory with strategic frame options."""

    safe: FrameOption = Field(..., description="Low-regret, defensible choice")
    high_variance: FrameOption = Field(..., description="Bold choice with upside potential")
    avoid: FrameOption = Field(..., description="Common pitfall to avoid")
    meta: AdvisoryMeta = Field(..., description="Advisory metadata")
    debug: dict = Field(
        default_factory=dict,
        description="Debug data including all_frames_scored and scoring_notes",
    )


class Phase1Pillars(BaseModel):
    """Phase-1 diagnostic pillars (observational, non-prescriptive)."""

    emotional_signal: str = Field(
        ...,
        description="What emotional signal is visible and how clearly it reads at a glance.",
    )
    curiosity_gap: str = Field(
        ...,
        description="Whether the frame hints at an outcome without resolving it (aligned to title/topic).",
    )
    attention_signals: list[str] = Field(
        default_factory=list,
        description="Observed attention signals present in the frame (e.g., face, number, symbol, motion, contrast).",
    )
    readability_speed: str = Field(
        ...,
        description="Whether it survives a small-screen, ~2-second scan (single focal point, separation, instant recognizability).",
    )


class Phase1MomentInsight(BaseModel):
    """A single ClickMoment candidate moment with Phase-1 diagnostic insight."""

    frame_id: str = Field(..., description="Frame identifier (e.g., 'Frame 3')")
    frame_number: int = Field(..., description="Frame index number")
    timestamp: str = Field(..., description="Timestamp in video (e.g., '5.2s')")
    frame_url: Optional[HttpUrl] = Field(None, description="URL to frame image")

    moment_summary: str = Field(
        ..., description="Short, observational summary of what the frame shows (max ~18 words)."
    )
    viewer_feel: str = Field(
        ...,
        description="What a viewer likely feels at a glance (fast, emotional, non-technical).",
    )
    why_this_reads: list[str] = Field(
        default_factory=list,
        description="Flattened, user-facing reasons (observational) drawn from the four pillars.",
    )
    optional_note: Optional[str] = Field(
        None,
        description="Optional nuance (e.g., if capture is weak, recreating the same moment may help).",
    )
    pillars: Phase1Pillars


class ClickMomentPhase1(BaseModel):
    """Phase-1 output: diagnostic (not prescriptive) clickable-moment surfacing."""

    positioning: str = Field(
        "ClickMoment identifies moments in your video that are already psychologically ready to earn clicks.",
        description="Core positioning statement shown to users.",
    )
    moments: list[Phase1MomentInsight] = Field(
        default_factory=list,
        description="Top moments (ordered, not numbered/ranked in copy).",
    )
    meta: dict = Field(default_factory=dict, description="Optional metadata (confidence, notes).")
    debug: dict = Field(
        default_factory=dict,
        description="Debug data including all_frames_scored and scoring_notes (developer-facing).",
    )


class ThumbnailResponse(BaseModel):
    """Response from thumbnail generation endpoint with AI advisory."""

    project_id: str
    status: Literal["draft", "final"]
    recommended_title: str

    # Legacy fields (for backwards compatibility)
    thumbnail_url: HttpUrl = Field(..., description="Default/safe option frame URL")
    selected_frame_url: Optional[HttpUrl] = Field(None, description="Alias for thumbnail_url")
    profile_variant_url: Optional[HttpUrl] = None
    layers: list[CompositionLayer] = Field(default_factory=list)

    # New advisory data
    advisory: Optional[ThumbnailAdvisory] = Field(
        None, description="AI strategic options (safe/bold/avoid)"
    )
    phase1: Optional[ClickMomentPhase1] = Field(
        None,
        description="Phase-1 ClickMoment diagnostic insights (observational, non-prescriptive).",
    )

    # Processing metadata
    total_frames_extracted: int = Field(..., description="Total frames from adaptive sampling")
    analysis_json_url: Optional[str] = Field(
        None, description="GCS URL to comprehensive analysis JSON"
    )
    cost_usd: Optional[float] = Field(None, description="AI selection cost in USD")
    gemini_model: Optional[str] = Field(None, description="Gemini model used for selection")

    # Summary
    summary: str
