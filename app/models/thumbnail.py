"""Thumbnail generation and composition models."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from app.models.media import SourceMedia


class CreatorContext(BaseModel):
    """
    Creator context signals for autonomous packaging.

    The system autonomously decides how content should be packaged,
    but strongly respects these creator signals when provided.
    """

    niche_hint: Optional[str] = Field(
        None,
        description=(
            "Content category signal (e.g., 'travel', 'tech', 'cooking'). "
            "When provided, system prioritizes this over inferred category. "
            "System analyzes content but defers to creator's understanding of their niche."
        ),
        max_length=80,
    )

    maturity_hint: Optional[str] = Field(
        None,
        description=(
            "Channel maturity signal (e.g., 'early', 'mid', 'established'). "
            "When provided, system adjusts packaging strategy accordingly. "
            "Helps calibrate tone and approach to match channel stage."
        ),
        max_length=40,
    )


class CreativeDirection(BaseModel):
    """
    Creative direction signals in natural language.

    The system makes autonomous packaging decisions, but treats these signals
    as strong guidance that shapes how content is presented.
    """

    mood: Optional[str] = Field(
        None,
        description=(
            "Desired vibe or tone (e.g., 'dramatic', 'calm', 'authentic'). "
            "System strongly aligns moment selection and presentation with this mood. "
            "Natural language - describe how you want the moment to feel."
        ),
        max_length=120,
    )

    title_hint: Optional[str] = Field(
        None,
        description=(
            "Working title or topic for this video. "
            "System uses this to understand content theme and select moments that align. "
            "Strong signal for what the video is about."
        ),
        max_length=120,
    )

    visual_preferences: Optional[str] = Field(
        None,
        description=(
            "Visual preferences in natural language (e.g., 'face-focused', 'show the scenery', 'high energy', 'avoid close-ups'). "
            "System prioritizes these preferences when scoring and selecting moments. "
            "Strong signal for visual style."
        ),
        max_length=200,
    )

    notes: Optional[str] = Field(
        None,
        description=(
            "Additional context about this specific video. "
            "System interprets this as important creative direction. "
            "Use this to emphasize what matters for this content."
        ),
        max_length=500,
    )


class ThumbnailRequest(BaseModel):
    """
    Request for autonomous content packaging.

    The system analyzes video content to determine optimal moment selection and presentation.
    All context fields are optional hints that guide, but don't prescribe, the system's decisions.
    """

    project_id: Optional[str] = Field(
        None,
        description="Optional project identifier. If not provided, a new UUID will be generated. Use the same project_id to overwrite previous frames.",
    )
    content_sources: SourceMedia = Field(
        ...,
        description="Video content to analyze and package. The system will extract and analyze key moments autonomously.",
    )
    profile_photos: list[str] = Field(
        default_factory=list,
        description="Optional creator photos for avatar extraction (https://storage.googleapis.com/... or local paths).",
    )
    creator_context: CreatorContext = Field(
        default_factory=CreatorContext,
        description="Optional lightweight hints about creator and channel. Used only when content analysis is ambiguous.",
    )
    creative_direction: CreativeDirection = Field(
        default_factory=CreativeDirection,
        description="Optional creative preferences in natural language. Guides style without overriding content-driven decisions.",
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
                    "creator_context": {
                        "niche_hint": "travel",
                        "maturity_hint": "early",
                    },
                    "creative_direction": {
                        "mood": "casual, authentic, chill with energetic moments",
                        "title_hint": "Exploring New Orleans for the First Time | What to See & Do",
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
    """Response from autonomous content packaging system."""

    project_id: str
    status: Literal["draft", "final"]
    title_hint: str = Field(..., description="Creator's provided title hint (passed through)")

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
