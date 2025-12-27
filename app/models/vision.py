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


class VisionBreakdownResponse(BaseModel):
    """Response from vision breakdown endpoint."""

    project_id: str
    total_frames: int
    scored_frames: list[FrameScore]
    best_frame: FrameScore
    summary: str
