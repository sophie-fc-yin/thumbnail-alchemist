"""Visual media handling - validation, preprocessing."""

import asyncio
import logging
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import google.auth
from google.auth import compute_engine
from google.auth.transport import requests
from google.cloud import storage  # type: ignore

from app.constants import (
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_VIDEO_FPS,
    FPS_VALID_RANGE_MAX,
    FPS_VALID_RANGE_MIN,
)
from app.models import SourceMedia

logger = logging.getLogger(__name__)


class MediaValidationError(Exception):
    """Raised when media validation fails."""

    pass


class VideoDurationExceededError(MediaValidationError):
    """Raised when video duration exceeds the maximum allowed duration."""

    pass


def parse_gcs_url(url: str) -> tuple[str, str] | None:
    """
    Parse GCS URL to extract bucket and blob name.

    Args:
        url: GCS URL in format gs://bucket/path, https://storage.googleapis.com/bucket/path,
             or https://storage.cloud.google.com/bucket/path

    Returns:
        Tuple of (bucket_name, blob_name) or None if not a GCS URL
    """
    if url.startswith("gs://"):
        parts = url.replace("gs://", "").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    elif url.startswith("https://storage.googleapis.com/"):
        parts = url.replace("https://storage.googleapis.com/", "").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    elif url.startswith("https://storage.cloud.google.com/"):
        parts = url.replace("https://storage.cloud.google.com/", "").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def generate_signed_url(
    url: str, expiration_minutes: int = 60, require_signing: bool = False
) -> str:
    """
    Generate a signed URL for a GCS object, or return public URL as-is.

    Args:
        url: GCS URL (gs://bucket/path or https://storage.googleapis.com/bucket/path)
        expiration_minutes: How long the signed URL should be valid (default: 60 minutes)
        require_signing: If True, raise error on signing failure. If False, return original URL.

    Returns:
        Signed URL for private files, or original URL for public files

    Raises:
        MediaValidationError: If require_signing=True and signing fails or blob doesn't exist
    """
    parsed = parse_gcs_url(url)
    if not parsed:
        # Not a GCS URL, return as-is (local path or other URL)
        return url

    bucket_name, blob_name = parsed

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if blob exists before generating signed URL
        if not blob.exists():
            raise MediaValidationError(
                f"GCS object does not exist: gs://{bucket_name}/{blob_name}. "
                "The file may not have been uploaded successfully, or there may be a delay in GCS availability."
            )

        # Get credentials and check if we need to use IAM signing
        credentials, project = google.auth.default()

        # If using compute engine credentials (no private key), use IAM-based signing
        if isinstance(credentials, compute_engine.Credentials):
            # Get the service account email from the metadata server
            auth_request = requests.Request()
            credentials.refresh(auth_request)
            service_account_email = credentials.service_account_email

            # Generate signed URL using IAM (no private key needed)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET",
                service_account_email=service_account_email,
                access_token=credentials.token,
            )
        else:
            # Use regular signing with private key
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET",
            )

        return signed_url

    except MediaValidationError:
        # Re-raise MediaValidationError as-is
        raise
    except Exception as e:
        # If signing fails and we don't require it, return original URL
        # This handles public buckets and cases where credentials don't have signing permission
        if not require_signing and url.startswith("https://"):
            return url
        raise MediaValidationError(f"Failed to generate signed URL for {url}: {e}") from e


async def get_video_duration(video_path: str, video_url: str | None = None) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file (local path, GCS URL, or HTTP URL)
        video_url: Optional pre-generated signed URL (avoids duplicate signing)

    Returns:
        Video duration in seconds

    Raises:
        MediaValidationError: If video duration cannot be determined
    """
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise MediaValidationError("ffprobe not found. Cannot determine video duration.")

    # Use pre-generated signed URL if provided, otherwise generate one
    if video_url:
        # Reuse pre-generated signed URL (avoids duplicate GCS API calls)
        pass  # video_url already set
    elif video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    try:
        process = await asyncio.create_subprocess_exec(
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise MediaValidationError(
                f"Failed to get video duration: ffprobe returned code {process.returncode}: {error_msg}"
            )

        duration_str = stdout.decode().strip()
        if not duration_str:
            raise MediaValidationError("ffprobe returned empty duration")

        duration = float(duration_str)
        if duration <= 0:
            raise MediaValidationError(f"Invalid video duration: {duration} seconds")

        return duration

    except ValueError as e:
        raise MediaValidationError(f"Failed to parse video duration: {e}") from e
    except Exception as e:
        raise MediaValidationError(f"Failed to get video duration: {e}") from e


async def get_video_fps(video_path: str, video_url: str | None = None) -> float:
    """
    Get video frame rate (FPS) using ffprobe.

    Args:
        video_path: Path to video file (local path, GCS URL, or HTTP URL)
        video_url: Optional pre-generated signed URL (avoids duplicate signing)

    Returns:
        Video frame rate (frames per second), defaults to DEFAULT_VIDEO_FPS if detection fails
    """

    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        logger.warning("ffprobe not found, using default FPS: %.1f", DEFAULT_VIDEO_FPS)
        return DEFAULT_VIDEO_FPS

    # Use pre-generated signed URL if provided, otherwise generate one
    if video_url:
        pass  # video_url already set
    elif video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    try:
        # Get r_frame_rate (most accurate for constant FPS videos)
        process = await asyncio.create_subprocess_exec(
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0 and stdout:
            fps_str = stdout.decode().strip()

            # Parse fraction format (e.g., "30000/1001" for 29.97 fps)
            if "/" in fps_str:
                numerator, denominator = fps_str.split("/")
                fps = float(numerator) / float(denominator)
            else:
                fps = float(fps_str)

            # Validate FPS (typical range: 15-120 fps)
            if FPS_VALID_RANGE_MIN <= fps <= FPS_VALID_RANGE_MAX:
                logger.info("Detected video FPS: %.2f", fps)
                return fps

        logger.warning("Failed to detect FPS, using default: %.1f", DEFAULT_VIDEO_FPS)
        return DEFAULT_VIDEO_FPS

    except Exception as e:
        logger.warning("FPS detection failed: %s, using default: %.1f", e, DEFAULT_VIDEO_FPS)
        return DEFAULT_VIDEO_FPS


async def validate_video_duration(
    video_path: str,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    video_url: str | None = None,
) -> None:
    """
    Validate that video duration does not exceed the maximum allowed duration.

    Args:
        video_path: Path to video file
        max_duration_seconds: Maximum allowed duration in seconds
        video_url: Optional pre-generated signed URL (avoids duplicate signing)

    Raises:
        VideoDurationExceededError: If video duration exceeds the limit
        MediaValidationError: If video duration cannot be determined
    """
    duration = await get_video_duration(video_path, video_url=video_url)

    if duration > max_duration_seconds:
        max_minutes = max_duration_seconds // 60
        actual_minutes = duration // 60
        raise VideoDurationExceededError(
            f"Video duration ({actual_minutes:.1f} minutes, {duration:.1f} seconds) "
            f"exceeds maximum allowed duration ({max_minutes} minutes, {max_duration_seconds} seconds). "
            f"Please upload a video that is {max_minutes} minutes or shorter."
        )


async def validate_and_load_content(sources: SourceMedia) -> dict[str, Any]:
    """
    Validate all content source URLs/paths and extract metadata.

    Args:
        sources: SourceMedia containing video_path and image_paths (URLs or local paths)

    Returns:
        Dictionary containing metadata about the content sources

    Raises:
        MediaValidationError: If any local media files are missing or inaccessible
    """
    media_sources: list[str] = []
    metadata: dict[str, Any] = {
        "video": None,
        "images": [],
        "total_files": 0,
    }

    # Collect all media sources
    if sources.video_path:
        media_sources.append(sources.video_path)

    if sources.image_paths:
        media_sources.extend(sources.image_paths)

    # Validate each source
    for source in media_sources:
        # URLs are validated when accessed (ffmpeg will handle errors)
        if source.startswith(("http://", "https://")):
            continue

        # Local paths - validate existence
        path = Path(source)
        if not path.exists():
            raise MediaValidationError(f"Media file not found: {path}")

        if not path.is_file():
            raise MediaValidationError(f"Path is not a file: {path}")

        # TODO: Check file permissions and readability
        # if not os.access(path, os.R_OK):
        #     raise MediaValidationError(f"File is not readable: {path}")

        # TODO: Validate file format/extension
        # allowed_extensions = {'.mp4', '.mov', '.avi', '.jpg', '.jpeg', '.png', '.webp'}
        # if path.suffix.lower() not in allowed_extensions:
        #     raise MediaValidationError(f"Unsupported file format: {path}")

    # Extract metadata from video if provided
    if sources.video_path:
        # TODO: Extract video metadata (duration, resolution, fps, codec, etc.)
        # video_info = await extract_video_metadata(sources.video_path)
        # metadata["video"] = video_info
        metadata["video"] = {
            "source": sources.video_path,
            "duration": None,  # TODO: Extract actual duration
            "resolution": None,  # TODO: Extract actual resolution
            "fps": None,  # TODO: Extract actual fps
        }

    # Extract metadata from images if provided
    if sources.image_paths:
        for image_source in sources.image_paths:
            # TODO: Extract image metadata (resolution, format, size, etc.)
            # image_info = await extract_image_metadata(image_source)
            # metadata["images"].append(image_info)
            metadata["images"].append(
                {
                    "source": image_source,
                    "resolution": None,  # TODO: Extract actual resolution
                    "format": Path(image_source).suffix
                    if not image_source.startswith("http")
                    else None,
                }
            )

    metadata["total_files"] = len(media_sources)

    return metadata


async def validate_profile_photos(profile_photos: list[str]) -> list[dict[str, Any]]:
    """
    Validate profile photo URLs/paths and extract metadata.

    Args:
        profile_photos: List of URLs or paths to profile photos

    Returns:
        List of validated profile photos with metadata

    Raises:
        MediaValidationError: If any local profile photos are missing or invalid
    """
    validated_photos: list[dict[str, Any]] = []

    for photo_source in profile_photos:
        # URLs are validated when accessed
        if photo_source.startswith(("http://", "https://")):
            validated_photos.append(
                {
                    "source": photo_source,
                    "resolution": None,  # TODO: Extract actual resolution
                    "has_face": None,  # TODO: Run face detection
                }
            )
            continue

        # Local paths - validate existence
        photo_path = Path(photo_source)
        if not photo_path.exists():
            raise MediaValidationError(f"Profile photo not found: {photo_path}")

        if not photo_path.is_file():
            raise MediaValidationError(f"Path is not a file: {photo_path}")

        # TODO: Validate image format
        # TODO: Check image dimensions (minimum size requirements)
        # TODO: Detect if image contains a face/person

        validated_photos.append(
            {
                "source": photo_source,
                "resolution": None,  # TODO: Extract actual resolution
                "has_face": None,  # TODO: Run face detection
            }
        )

    return validated_photos
