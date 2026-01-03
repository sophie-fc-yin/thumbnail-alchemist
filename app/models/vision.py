"""Vision analysis and frame scoring models."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class VisionBreakdownRequest(BaseModel):
    """Request for vision breakdown endpoint."""

    project_id: Optional[str] = Field(
        None,
        description="Optional project identifier. If not provided, a new UUID will be generated.",
        examples=["60a8336a-3d5d-45eb-a390-b52ab9f2dcb2"],
    )
    video_path: str = Field(
        ...,
        description="URL or path to video file for frame extraction",
        examples=[
            "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4"
        ],
    )
    max_frames: int = Field(
        50,
        description="Maximum number of frames to extract from video (distributed across full duration)",
        examples=[50, 100, 24],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "project_id": "60a8336a-3d5d-45eb-a390-b52ab9f2dcb2",
                    "video_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
                    "max_frames": 50,
                }
            ]
        }
    }


class FrameScore(BaseModel):
    """Score information for a single frame."""

    frame_path: str
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds from start of video")
    brightness: Optional[float]
    sharpness: Optional[float]
    motion: Optional[float]
    face_score: Optional[float]
    expression_score: Optional[float]
    highlight_score: Optional[float]
    rank: int


class ImportanceSegment(BaseModel):
    """Moment importance segment information."""

    start_time: float = Field(..., description="Segment start in seconds")
    end_time: float = Field(..., description="Segment end in seconds")
    avg_importance: float = Field(..., description="Average importance score [0, 1]")
    importance_level: str = Field(..., description="Importance level: low, medium, or high")


class ImportanceStatistics(BaseModel):
    """Overall moment importance statistics."""

    avg_importance: float = Field(..., description="Average importance across all segments")
    segment_counts: dict[str, int] = Field(
        ..., description="Count of low/medium/high importance segments"
    )
    total_segments: int = Field(..., description="Total number of importance segments")


class ProcessingStats(BaseModel):
    """Processing time breakdown."""

    audio_time: float = Field(..., description="Audio analysis time in seconds")
    initial_sampling_time: float = Field(..., description="Initial frame sampling time in seconds")
    face_analysis_time: float = Field(..., description="Face analysis time in seconds")
    importance_calculation_time: float = Field(
        ..., description="Moment importance calculation time in seconds"
    )
    adaptive_extraction_time: float = Field(
        ..., description="Adaptive frame extraction time in seconds"
    )
    total_time: float = Field(..., description="Total processing time in seconds")


class VisionBreakdownResponse(BaseModel):
    """Response from vision breakdown endpoint."""

    project_id: str
    frames: list[dict] = Field(
        ...,
        description="List of frame dictionaries, each containing: time, filename, shot_change, motion_score, motion_spike",
    )
    sample_interval: float = Field(
        ..., description="Calculated adaptive sampling interval in seconds"
    )
    stats: dict[str, float] = Field(
        ...,
        description="Timing statistics: initial_sampling_time, visual_analysis_time",
    )


class AdaptiveSamplingResponse(BaseModel):
    """Response from adaptive sampling endpoint with moment importance analysis."""

    project_id: str
    frames: list[str] = Field(..., description="List of frame paths (GCS URLs or local)")
    total_frames: int = Field(..., description="Total number of frames extracted")
    importance_segments: list[ImportanceSegment] = Field(
        ..., description="Video segments grouped by moment importance"
    )
    importance_statistics: ImportanceStatistics = Field(
        ..., description="Overall importance statistics"
    )
    processing_stats: ProcessingStats = Field(..., description="Processing time breakdown")
    summary: str = Field(..., description="Human-readable summary")


class VisionFeaturesRequest(BaseModel):
    """Request for vision features computation endpoint."""

    image_path: Optional[str] = Field(
        None,
        description="URL or path to image file (GCS URL, HTTP URL, or local path). Required if image_file is not provided.",
        examples=[
            "gs://clickmoment-prod-assets/projects/123/frames/frame_001.jpg",
            "https://example.com/image.jpg",
        ],
    )
    niche: Optional[str] = Field(
        "general",
        description="Content niche for editability scoring (e.g., 'general', 'beauty', 'tech', 'gaming')",
        examples=["general", "beauty", "tech", "gaming"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_path": "gs://clickmoment-prod-assets/projects/123/frames/frame_001.jpg",
                    "niche": "general",
                }
            ]
        }
    }


class ImageQuality(BaseModel):
    """Image quality metrics."""

    brightness: float = Field(..., description="Overall frame brightness [0, 1]")
    subject_brightness: float = Field(..., description="Center region (subject) brightness [0, 1]")
    contrast: float = Field(..., description="Contrast level [0, 1]")
    sharpness: float = Field(..., description="Sharpness score [0, 1]")
    noise_level: float = Field(..., description="Noise level estimate [0, 1]")
    is_too_dark: bool = Field(..., description="True if subject is too dark")
    is_too_bright: bool = Field(..., description="True if subject is overexposed")
    mean_brightness: float = Field(..., description="Raw mean brightness [0, 255]")


class EditabilityScores(BaseModel):
    """Editability scores for thumbnail creation."""

    overall_editability: float = Field(..., description="Overall editability score [0, 1]")
    crop_resilience: float = Field(..., description="Crop resilience score [0, 1]")
    zoom_potential: float = Field(..., description="Zoom potential score [0, 1]")
    text_overlay_space: float = Field(..., description="Text overlay space score [0, 1]")
    emotion_resilience: float = Field(..., description="Emotion resilience score [0, 1]")
    composition_flexibility: float = Field(..., description="Composition flexibility score [0, 1]")


class CompositionScores(BaseModel):
    """Composition quality scores."""

    overall_score: float = Field(..., description="Overall composition score [0, 1]")
    rule_of_thirds: float = Field(..., description="Rule of thirds alignment [0, 1]")
    contrast_strength: float = Field(..., description="Contrast strength [0, 1]")
    color_harmony: float = Field(..., description="Color harmony score [0, 1]")
    visual_balance: float = Field(..., description="Visual balance score [0, 1]")
    background_cleanliness: float = Field(..., description="Background cleanliness [0, 1]")


class TechnicalQuality(BaseModel):
    """Technical quality scores."""

    overall_score: float = Field(..., description="Overall technical quality [0, 1]")
    sharpness: float = Field(..., description="Sharpness score [0, 1]")
    noise_level: float = Field(
        ..., description="Noise level (inverted: higher = less noise) [0, 1]"
    )
    exposure_quality: float = Field(..., description="Exposure quality [0, 1]")


class FaceQualityScores(BaseModel):
    """Face quality metrics."""

    overall_score: float = Field(..., description="Overall face quality [0, 1]")
    expression_strength: float = Field(..., description="Expression strength [0, 1]")
    eye_contact_quality: float = Field(..., description="Eye contact quality [0, 1]")
    facial_clarity: float = Field(..., description="Facial clarity [0, 1]")
    emotion_authenticity: float = Field(..., description="Emotion authenticity [0, 1]")
    composition_quality: float = Field(..., description="Composition quality [0, 1]")


class AestheticsScores(BaseModel):
    """Aesthetic quality scores."""

    score: float = Field(..., description="Overall aesthetic score [0, 1]")
    image_quality: ImageQuality = Field(..., description="Image quality metrics")
    frame_features: dict = Field(..., description="Frame features (lighting, color palette, etc.)")


class VisionFeaturesResponse(BaseModel):
    """Response from vision features computation endpoint."""

    image_path: str = Field(..., description="Path or URL to the analyzed image")
    face_analysis: dict = Field(
        ..., description="Face analysis results from FaceExpressionAnalyzer"
    )
    aesthetics: AestheticsScores = Field(..., description="Aesthetic quality scores")
    editability: EditabilityScores = Field(..., description="Editability scores")
    composition: CompositionScores = Field(..., description="Composition quality scores")
    technical_quality: TechnicalQuality = Field(..., description="Technical quality scores")
    face_quality: FaceQualityScores = Field(..., description="Face quality scores")


class FrameWithFeatures(BaseModel):
    """A single frame with all computed vision features."""

    local_path: str = Field(..., description="Local path to the frame file")
    gcs_url: str = Field(..., description="GCS URL of the uploaded frame")
    time: float = Field(..., description="Timestamp in seconds")
    segment_index: int = Field(
        ..., description="Index of the importance segment this frame belongs to"
    )
    importance_level: str = Field(..., description="Importance level of the segment")
    importance_score: float = Field(..., description="Importance score [0, 1]")
    segment_start: float = Field(..., description="Segment start time in seconds")
    segment_end: float = Field(..., description="Segment end time in seconds")
    face_analysis: dict = Field(..., description="Face analysis results")
    visual_analysis: dict = Field(
        ..., description="All vision features (aesthetics, editability, composition, etc.)"
    )


class ProcessImportantMomentsRequest(BaseModel):
    """Request for processing important moments endpoint."""

    project_id: Optional[str] = Field(
        None,
        description="Optional project identifier. If not provided, a new UUID will be generated.",
        examples=["60a8336a-3d5d-45eb-a390-b52ab9f2dcb2"],
    )
    video_path: str = Field(
        ...,
        description="URL or path to video file",
        examples=[
            "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4"
        ],
    )
    stream_a_results: Optional[Any] = Field(
        default=None,
        description="Optional Stream A results (speech semantics). If not provided (null), will be loaded from GCS or computed. Should be a list of segment dictionaries with 'start', 'end', 'importance', 'narrative_context', etc.",
    )
    stream_b_results: Optional[Any] = Field(
        default=None,
        description="Optional Stream B results (audio saliency). If not provided (null), will be loaded from GCS or computed. Should be a list of segment dictionaries with 'start', 'end', 'saliency_score', etc.",
    )
    visual_frames: Optional[Any] = Field(
        default=None,
        description="Optional visual frames from initial vision analysis. If not provided (null), will be loaded from GCS or computed. Should be a list of frame dictionaries with 'time', 'shot_change', 'motion_spike', etc.",
    )
    niche: str = Field(
        "general",
        description="Content niche for editability scoring",
        examples=["general", "beauty", "tech", "gaming"],
    )


class ProcessImportantMomentsResponse(BaseModel):
    """Response from processing important moments endpoint."""

    project_id: str = Field(..., description="Project identifier")
    importance_segments: list[dict] = Field(
        ...,
        description="List of importance segments with start_time, end_time, importance_level, avg_importance, etc.",
    )
    frames: list[FrameWithFeatures] = Field(
        ...,
        description="List of extracted frames with all computed vision features",
    )
    stats: dict[str, float] = Field(
        ...,
        description="Processing statistics: importance_calculation_time, extraction_time, vision_analysis_time, total_time",
    )
    diagnostic: Optional[dict] = Field(
        None,
        description="Diagnostic information about why results might be empty (signals found, GCS loading status, etc.)",
    )
