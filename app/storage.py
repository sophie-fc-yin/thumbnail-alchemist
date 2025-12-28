"""Google Cloud Storage utilities for file uploads."""

from datetime import timedelta

from google.api_core import exceptions as gcs_exceptions
from google.cloud import storage


class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


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
        client = storage.Client()
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
        client = storage.Client()
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

    except gcs_exceptions.GoogleAPIError as e:
        raise StorageError(f"GCS upload failed: {str(e)}") from e
    except StorageError:
        # Re-raise StorageError as-is
        raise
    except Exception as e:
        raise StorageError(f"Unexpected upload error: {str(e)}") from e


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
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Construct path: users/{user_id}/videos/{filename}
        blob_path = f"{base_path}/{user_id}/{subfolder}/{filename}"
        blob = bucket.blob(blob_path)

        # Generate signed URL for PUT operation
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration_seconds),
            method="PUT",
            content_type=content_type,
        )

        # Construct GCS path
        gcs_path = f"gs://{bucket_name}/{blob_path}"

        return signed_url, gcs_path

    except gcs_exceptions.GoogleAPIError as e:
        raise StorageError(f"Failed to generate signed URL: {str(e)}") from e
    except Exception as e:
        raise StorageError(f"Unexpected error generating signed URL: {str(e)}") from e
