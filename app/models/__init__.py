"""Models package for Thumbnail Alchemist API.

This package organizes models by domain:
- media: Source media handling (videos, images)
- audio: Audio breakdown and analysis
- vision: Frame extraction and scoring
- thumbnail: Thumbnail generation and composition
"""

# Media models
# Audio models
from app.models.audio import AudioBreakdownRequest, AudioBreakdownResponse
from app.models.media import SourceMedia

# Thumbnail models
from app.models.thumbnail import (
    ChannelProfile,
    CompositionLayer,
    CreativeBrief,
    Target,
    ThumbnailRequest,
    ThumbnailResponse,
)

# Upload models
from app.models.upload import (
    SignedUrlRequest,
    SignedUrlResponse,
    VideoUploadError,
    VideoUploadResponse,
    VideoUrlRequest,
    VideoUrlResponse,
)

# Vision models
from app.models.vision import (
    AdaptiveSamplingResponse,
    FrameScore,
    ImportanceSegment,
    ImportanceStatistics,
    ProcessingStats,
    VisionBreakdownRequest,
    VisionBreakdownResponse,
)

__all__ = [
    # Media
    "SourceMedia",
    # Audio
    "AudioBreakdownRequest",
    "AudioBreakdownResponse",
    # Vision
    "AdaptiveSamplingResponse",
    "FrameScore",
    "ImportanceSegment",
    "ImportanceStatistics",
    "ProcessingStats",
    "VisionBreakdownRequest",
    "VisionBreakdownResponse",
    # Thumbnail
    "ChannelProfile",
    "CompositionLayer",
    "CreativeBrief",
    "Target",
    "ThumbnailRequest",
    "ThumbnailResponse",
    # Upload
    "SignedUrlRequest",
    "SignedUrlResponse",
    "VideoUploadError",
    "VideoUploadResponse",
    "VideoUrlRequest",
    "VideoUrlResponse",
]
