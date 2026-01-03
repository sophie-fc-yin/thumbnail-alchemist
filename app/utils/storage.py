"""Google Cloud Storage utilities for file uploads."""

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


def _get_storage_client() -> Any:
    """
    Lazily import and create a google.cloud.storage client.

    This keeps module import lightweight (important for tests and environments
    where GCS credentials/SSL trust store are unavailable).
    """
    try:
        from google.cloud import storage  # type: ignore

        return storage.Client()
    except Exception as e:  # pragma: no cover - depends on runtime env
        raise StorageError(f"Google Cloud Storage client unavailable: {e}") from e


def check_blob_exists(bucket_name: str, blob_path: str) -> bool:
    """
    Check if a blob exists in GCS bucket.

    Args:
        bucket_name: Name of GCS bucket
        blob_path: Path to blob within bucket

    Returns:
        True if blob exists, False otherwise
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception:
        return False


async def upload_file_to_gcs(
    file_content: bytes,
    filename: str,
    user_id: str,
    bucket_name: str = "clickmoment-prod-assets",
    base_path: str = "users",
    subfolder: str = "videos",
) -> tuple[str, bool]:
    """
    Upload file to GCS in user-specific directory.

    Args:
        file_content: Raw bytes of the file
        filename: Original filename (sanitized)
        user_id: User ID from authentication
        bucket_name: GCS bucket name
        base_path: Base path prefix (e.g., "users")
        subfolder: Subfolder within user directory (e.g., "videos", "avatar", "brand")

    Returns:
        Tuple of (gcs_url, was_overwritten)

    Raises:
        StorageError: If upload fails
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)

        # Construct path: users/{user_id}/videos/{filename}
        blob_path = f"{base_path}/{user_id}/{subfolder}/{filename}"
        blob = bucket.blob(blob_path)

        # Check if file already exists
        file_exists = blob.exists()

        # Upload file (overwrites if exists)
        blob.upload_from_string(
            file_content,
            content_type=None,  # Auto-detect from filename
        )

        # Verify upload succeeded by reloading blob metadata
        blob.reload()
        if not blob.exists():
            raise StorageError(
                f"Upload appeared to succeed but blob doesn't exist at gs://{bucket_name}/{blob_path}"
            )

        # Construct GCS URL
        gcs_url = f"gs://{bucket_name}/{blob_path}"

        return gcs_url, file_exists

    except Exception as e:
        # google.api_core exceptions can vary by environment; keep catch broad.
        raise StorageError(f"GCS upload failed: {str(e)}") from e
    except StorageError:
        # Re-raise StorageError as-is
        raise


def generate_signed_upload_url(
    filename: str,
    user_id: str,
    content_type: str = "video/mp4",
    bucket_name: str = "clickmoment-prod-assets",
    base_path: str = "users",
    subfolder: str = "videos",
    expiration_seconds: int = 3600,
) -> tuple[str, str]:
    """
    Generate a signed URL for direct client upload to GCS.

    Args:
        filename: Name of the file to be uploaded (should be sanitized)
        user_id: User ID from authentication
        content_type: MIME type of the file
        bucket_name: GCS bucket name
        base_path: Base path prefix (e.g., "users")
        subfolder: Subfolder within user directory (e.g., "videos", "avatar", "brand")
        expiration_seconds: Number of seconds until the signed URL expires

    Returns:
        Tuple of (signed_url, gcs_path)

    Raises:
        StorageError: If signed URL generation fails
    """
    try:
        import google.auth
        from google.auth import compute_engine
        from google.auth.transport import requests

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)

        # Construct path: users/{user_id}/videos/{filename}
        blob_path = f"{base_path}/{user_id}/{subfolder}/{filename}"
        blob = bucket.blob(blob_path)

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
                expiration=timedelta(seconds=expiration_seconds),
                method="PUT",
                content_type=content_type,
                service_account_email=service_account_email,
                access_token=credentials.token,
            )
        else:
            # Use regular signing with private key
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration_seconds),
                method="PUT",
                content_type=content_type,
            )

        # Construct GCS path
        gcs_path = f"gs://{bucket_name}/{blob_path}"

        return signed_url, gcs_path

    except Exception as e:
        raise StorageError(f"Unexpected error generating signed URL: {str(e)}") from e


def generate_signed_download_url(
    gcs_path: str,
    bucket_name: str = "clickmoment-prod-assets",
    expiration_seconds: int = 3600,
) -> str:
    """
    Generate a signed URL for viewing/downloading a file from GCS.

    Args:
        gcs_path: GCS path of the file (gs://bucket/path or just path)
        bucket_name: GCS bucket name (used if gcs_path doesn't include bucket)
        expiration_seconds: Number of seconds until the signed URL expires

    Returns:
        Signed URL for GET operation

    Raises:
        StorageError: If signed URL generation fails
    """
    try:
        import google.auth
        from google.auth import compute_engine
        from google.auth.transport import requests

        # Parse GCS path to extract bucket and blob path
        if gcs_path.startswith("gs://"):
            # Extract bucket and path from gs://bucket/path format
            path_parts = gcs_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""
        else:
            # Use provided bucket_name and treat gcs_path as blob path
            blob_path = gcs_path

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Check if blob exists
        if not blob.exists():
            raise StorageError(f"File not found: gs://{bucket_name}/{blob_path}")

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
                expiration=timedelta(seconds=expiration_seconds),
                method="GET",
                service_account_email=service_account_email,
                access_token=credentials.token,
            )
        else:
            # Use regular signing with private key
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration_seconds),
                method="GET",
            )

        return signed_url

    except StorageError:
        # Re-raise StorageError as-is
        raise
    except Exception as e:
        raise StorageError(f"Unexpected error generating signed URL: {str(e)}") from e


def download_json_from_gcs(
    project_id: str,
    directory: str,
    filename: str,
    bucket_name: str = "clickmoment-prod-assets",
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """
    Download JSON data from GCS.

    Args:
        project_id: Project identifier
        directory: Directory path within project (e.g., "signals/audio")
        filename: JSON filename (e.g., "stream_a_results.json")
        bucket_name: GCS bucket name

    Returns:
        Parsed JSON data (dict or list) if successful, None on failure
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob_path = f"projects/{project_id}/{directory}/{filename}"
        blob = bucket.blob(blob_path)

        logger.info("Full GCS URL: https://storage.cloud.google.com/%s/%s", bucket_name, blob_path)

        if not blob.exists():
            logger.warning("JSON file not found at gs://%s/%s", bucket_name, blob_path)
            logger.warning(
                "Please verify: bucket=%s, project_id=%s, directory=%s, filename=%s",
                bucket_name,
                project_id,
                directory,
                filename,
            )
            # Try to list blobs in the directory to see what's actually there
            try:
                prefix = f"projects/{project_id}/{directory}/"
                blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
                if blobs:
                    for b in blobs:
                        logger.info("  - %s", b.name)
                else:
                    logger.warning("No files found in directory %s", prefix)
                    # Also try listing at the project level to see what directories exist
                    project_prefix = f"projects/{project_id}/"
                    project_blobs = list(bucket.list_blobs(prefix=project_prefix, max_results=20))
                    if project_blobs:  # Show unique directory paths
                        dirs = set()
                        for b in project_blobs:
                            parts = b.name.split("/")
                            if len(parts) >= 3:
                                dirs.add("/".join(parts[:3]))
                        for d in sorted(dirs):
                            logger.info("  - %s/", d)
                    else:
                        logger.warning("No files found under project %s at all", project_id)
            except Exception as e:
                logger.warning("Could not list blobs in directory: %s", e)
            return None

        json_content = blob.download_as_text()
        if not json_content or json_content.strip() == "":
            logger.warning("JSON file is empty at gs://%s/%s", bucket_name, blob_path)
            return None

        data = json.loads(json_content)
        logger.info(
            "✓ Successfully downloaded JSON from gs://%s/%s (size: %d bytes, type: %s)",
            bucket_name,
            blob_path,
            len(json_content),
            type(data).__name__,
        )

        # Log if it's an empty list/dict
        if isinstance(data, list) and len(data) == 0:
            logger.warning("Downloaded JSON is an empty list - file may be empty")
        elif isinstance(data, dict) and len(data) == 0:
            logger.warning("Downloaded JSON is an empty dict - file may be empty")

        return data

    except Exception as e:
        logger.warning(
            "Failed to download JSON from GCS (projects/%s/%s/%s): %s",
            project_id,
            directory,
            filename,
            e,
        )
        return None


def upload_json_to_gcs(
    data: dict[str, Any] | list[dict[str, Any]],
    project_id: str,
    directory: str,
    filename: str,
    bucket_name: str = "clickmoment-prod-assets",
) -> str | None:
    """
    Upload JSON data to GCS in a project-specific directory.

    Handles serialization of Path objects and other non-serializable types.
    All exceptions are handled internally - returns None on failure.

    Args:
        data: JSON-serializable data (dict or list)
        project_id: Project identifier
        directory: Directory path within project (e.g., "signals/audio")
        filename: JSON filename (e.g., "speech_segments.json")
        bucket_name: GCS bucket name

    Returns:
        GCS URL (gs://bucket/path) if successful, None on failure
    """
    try:
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        serializable_data = convert_paths(data)
        json_content = json.dumps(serializable_data, indent=2, ensure_ascii=False)

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob_path = f"projects/{project_id}/{directory}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(json_content, content_type="application/json")

        gcs_url = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded JSON to %s", gcs_url)
        return gcs_url

    except Exception as e:
        logger.warning(
            "Failed to upload JSON to GCS (projects/%s/%s/%s): %s",
            project_id,
            directory,
            filename,
            e,
        )
        return None


def upload_project_file_to_gcs(
    file_path: Path | str,
    project_id: str,
    directory: str,
    filename: str,
    bucket_name: str = "clickmoment-prod-assets",
    cleanup_local: bool = False,
) -> str | None:
    """
    Upload any file to GCS in a project-specific directory.

    Generic file uploader that works for images, audio, JSON, or any file type.
    Handles file existence checks, upload, and optional local cleanup.
    All exceptions are handled internally - returns None on failure.

    Args:
        file_path: Path to the local file (Path or str)
        project_id: Project identifier
        directory: Directory path within project (e.g., "selected_frames", "top_frames")
        filename: Target filename (e.g., "frame_00004700ms.jpg")
        bucket_name: GCS bucket name
        cleanup_local: If True, delete local file after successful upload

    Returns:
        GCS URL (gs://bucket/path) if successful, None on failure
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.warning(
                "File does not exist, skipping upload: %s",
                file_path_obj,
            )
            return None

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob_path = f"projects/{project_id}/{directory}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path_obj))

        gcs_url = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded file to %s", gcs_url)

        # Clean up local file if requested
        if cleanup_local:
            file_path_obj.unlink(missing_ok=True)
            logger.debug("Cleaned up local file: %s", file_path_obj)

        return gcs_url

    except Exception as e:
        logger.error(
            "Failed to upload file %s to GCS: %s",
            file_path,
            e,
        )
        return None


def upload_audio_file_to_gcs(
    file_path: Path | str,
    project_id: str,
    directory: str,
    filename: str,
    bucket_name: str = "clickmoment-prod-assets",
    cleanup_local: bool = True,
) -> str | None:
    """
    Upload an audio file (WAV, MP3, etc.) to GCS in a project-specific directory.

    Handles file existence checks, upload, and optional local cleanup.
    All exceptions are handled internally - returns None on failure.

    Args:
        file_path: Path to the local audio file (Path or str)
        project_id: Project identifier
        directory: Directory path within project (e.g., "signals/audio")
        filename: Audio filename (e.g., "audio_speech.wav")
        bucket_name: GCS bucket name
        cleanup_local: If True, delete local file after successful upload

    Returns:
        GCS URL (gs://bucket/path) if successful, None on failure
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.warning(
                "Audio file does not exist, skipping upload: %s",
                file_path_obj,
            )
            return None

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob_path = f"projects/{project_id}/{directory}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path_obj))

        gcs_url = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded audio file to %s", gcs_url)

        # Clean up local file if requested
        if cleanup_local:
            file_path_obj.unlink(missing_ok=True)
            logger.debug("Cleaned up local audio file: %s", file_path_obj)

        return gcs_url

    except Exception as e:
        logger.warning(
            "Failed to upload audio file to GCS (projects/%s/%s/%s): %s",
            project_id,
            directory,
            filename,
            e,
        )
        return None


def upload_frame_to_gcs(
    file_path: Path | str,
    project_id: str,
    segment_start: float,
    segment_end: float,
    timestamp_ms: int,
    bucket_name: str = "clickmoment-prod-assets",
) -> str | None:
    """
    Upload a frame image to GCS in a project-specific directory.

    Frames are stored directly in: projects/{project_id}/frames/frame_{timestamp_ms}ms.jpg

    Handles file existence checks and upload.
    All exceptions are handled internally - returns None on failure.

    Args:
        file_path: Path to the local frame file (Path or str)
        project_id: Project identifier
        segment_start: Segment start time in seconds (unused, kept for API compatibility)
        segment_end: Segment end time in seconds (unused, kept for API compatibility)
        timestamp_ms: Frame timestamp in milliseconds
        bucket_name: GCS bucket name

    Returns:
        GCS URL (gs://bucket/path) if successful, None on failure
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.warning(
                "Frame file does not exist, skipping upload: %s",
                file_path_obj,
            )
            return None

        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        # Use 7-digit padding for timestamps (supports videos up to ~2.7 hours)
        blob_path = f"projects/{project_id}/frames/frame_{timestamp_ms:07d}ms.jpg"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(file_path_obj), content_type="image/jpeg")

        gcs_url = f"gs://{bucket_name}/{blob_path}"

        return gcs_url

    except Exception as e:
        logger.warning(
            "Failed to upload frame to GCS (projects/%s/frames/frame_%07dms.jpg): %s",
            project_id,
            timestamp_ms,
            e,
        )
        return None
