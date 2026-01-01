"""Vision analysis and frame scoring models."""

from typing import Optional

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
    total_frames: int
    scored_frames: list[FrameScore]
    best_frame: FrameScore
    summary: str


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
