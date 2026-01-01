"""Visual media handling - validation, preprocessing."""

from datetime import timedelta
from pathlib import Path
from typing import Any

from app.models import SourceMedia


class MediaValidationError(Exception):
    """Raised when media validation fails."""

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
        import google.auth
        from google.auth import compute_engine
        from google.auth.transport import requests
        from google.cloud import storage  # type: ignore

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
