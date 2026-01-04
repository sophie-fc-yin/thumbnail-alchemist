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
    CompositionLayer,
    CreativeDirection,
    CreatorContext,
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
    AestheticsScores,
    CompositionScores,
    EditabilityScores,
    FaceQualityScores,
    FrameScore,
    FrameWithFeatures,
    ImageQuality,
    ImportanceSegment,
    ImportanceStatistics,
    ProcessImportantMomentsRequest,
    ProcessImportantMomentsResponse,
    ProcessingStats,
    TechnicalQuality,
    VisionBreakdownRequest,
    VisionBreakdownResponse,
    VisionFeaturesRequest,
    VisionFeaturesResponse,
)

__all__ = [
    # Media
    "SourceMedia",
    # Audio
    "AudioBreakdownRequest",
    "AudioBreakdownResponse",
    # Vision
    "AdaptiveSamplingResponse",
    "AestheticsScores",
    "CompositionScores",
    "EditabilityScores",
    "FaceQualityScores",
    "FrameScore",
    "ImageQuality",
    "ImportanceSegment",
    "ImportanceStatistics",
    "ProcessingStats",
    "TechnicalQuality",
    "VisionBreakdownRequest",
    "VisionBreakdownResponse",
    "VisionFeaturesRequest",
    "VisionFeaturesResponse",
    "ProcessImportantMomentsRequest",
    "ProcessImportantMomentsResponse",
    "FrameWithFeatures",
    # Thumbnail
    "CompositionLayer",
    "CreativeDirection",
    "CreatorContext",
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
