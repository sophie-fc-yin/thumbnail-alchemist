"""Visual media handling - validation, preprocessing, and frame extraction."""

import asyncio
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.cloud import storage

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
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if blob exists before generating signed URL
        if not blob.exists():
            raise MediaValidationError(
                f"GCS object does not exist: gs://{bucket_name}/{blob_name}. "
                "The file may not have been uploaded successfully, or there may be a delay in GCS availability."
            )

        # Generate signed URL valid for specified duration
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


async def extract_candidate_frames(
    content_sources: SourceMedia,
    project_id: str,
    max_frames: int = 50,
    sampling_strategy: str = "adaptive",
    fps: float | None = None,
    every_n_seconds: float | None = None,
    output_dir: Path | None = None,
    ffmpeg_binary: str | None = None,
    upload_to_gcs: bool = True,
) -> list[Path | str]:
    """
    Stream video and extract frames distributed across the entire video duration.

    For GCS URLs, generates temporary signed URLs to access private files.
    ffmpeg streams the video progressively and extracts frames to local disk,
    then uploads them to GCS (if enabled).

    Args:
        content_sources: SourceMedia containing video_path (GCS URL, other URL, or local path)
        project_id: Unique identifier for the project (used to organize output files)
        max_frames: Maximum number of frames to extract (default: 50)
        sampling_strategy: "adaptive" (default) samples evenly across video duration,
                          "fps" forces a constant fps, and "time" samples every N seconds
        fps: Custom frames per second (used with "fps" strategy)
        every_n_seconds: Sample every N seconds (used with "time" strategy)
        output_dir: Custom output directory for extracted frames (overrides default structure)
        ffmpeg_binary: Path to ffmpeg binary (auto-detected if None)
        upload_to_gcs: If True, upload frames to GCS and delete local files (default: True)

    Returns:
        List of GCS URLs (if upload_to_gcs=True) or local file paths (if upload_to_gcs=False)
    """
    candidate_frames: list[Path] = []

    video_source = content_sources.video_path
    ffmpeg_path = ffmpeg_binary or shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not video_source:
        print("ERROR: No video source provided")
        return candidate_frames
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found in PATH")
        return candidate_frames
    if not ffprobe_path:
        print("ERROR: ffprobe not found in PATH")
        return candidate_frames

    # Check if video is local file (validate existence) or URL (generate signed URL if GCS)
    if not video_source.startswith(("http://", "https://", "gs://")):
        # Local file path - validate existence
        video_path_obj = Path(video_source)
        if not video_path_obj.exists():
            print(f"ERROR: Local video file not found: {video_source}")
            return candidate_frames
    else:
        # GCS URL - generate signed URL for private access
        # This allows ffmpeg to stream from private GCS files
        print(f"Generating signed URL for GCS path: {video_source}")
        try:
            video_source = generate_signed_url(video_source)
            print(f"Successfully generated signed URL (length: {len(video_source)})")
        except MediaValidationError as e:
            print(f"ERROR: Failed to generate signed URL: {e}")
            return candidate_frames
        except Exception as e:
            print(f"ERROR: Unexpected error generating signed URL: {e}")
            return candidate_frames

    if output_dir:
        target_dir = output_dir
    else:
        # Default to structured storage: clickmoment-prod-assets/projects/{project_id}/signals/frames
        target_dir = (
            Path("clickmoment-prod-assets") / "projects" / project_id / "signals" / "frames"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = target_dir / "frame_%03d.jpg"

    # For adaptive strategy, sample frames evenly across entire video duration
    if sampling_strategy == "adaptive":
        # Get video duration using ffprobe
        probe_process = await asyncio.create_subprocess_exec(
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_source,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await probe_process.communicate()

        try:
            duration = float(stdout.decode().strip())
            print(f"Video duration: {duration} seconds")
        except (ValueError, AttributeError):
            # Fallback to fps-based extraction if duration can't be determined
            duration = None
            stderr_text = stderr.decode() if stderr else ""
            print(f"ERROR: Could not determine video duration. ffprobe stderr: {stderr_text}")

        if duration and duration > 0:
            # Sample frames evenly across the video duration
            # Use select filter to pick frames at specific timestamps
            interval = duration / max_frames
            select_expr = "+".join([f"eq(n,{int(i * interval * 30)})" for i in range(max_frames)])

            print(f"Extracting frames with select expression (interval: {interval}s)")
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_source,
                "-vf",
                f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                "-vsync",
                "0",
                str(output_pattern),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if stderr:
                stderr_text = stderr.decode()
                if stderr_text.strip():
                    print(f"ffmpeg stderr: {stderr_text}")
            if process.returncode != 0:
                print(f"ERROR: ffmpeg exited with code {process.returncode}")

            # Rename frames to include timestamps in milliseconds
            temp_frames = sorted(target_dir.glob("frame_*.jpg"))
            for idx, temp_frame in enumerate(temp_frames):
                timestamp_ms = int(idx * interval * 1000)
                new_name = target_dir / f"frame_{timestamp_ms}ms.jpg"
                temp_frame.rename(new_name)
                print(f"Renamed {temp_frame.name} -> {new_name.name}")
        else:
            # Fallback: use fps-based extraction
            fps_value = 1.0
            print(f"Extracting frames using fps fallback (fps={fps_value})")
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_source,
                "-vf",
                f"fps=fps={fps_value}",
                "-frames:v",
                str(max_frames),
                str(output_pattern),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if stderr:
                stderr_text = stderr.decode()
                if stderr_text.strip():
                    print(f"ffmpeg stderr: {stderr_text}")
            if process.returncode != 0:
                print(f"ERROR: ffmpeg exited with code {process.returncode}")

            # Rename frames to include timestamps in milliseconds
            temp_frames = sorted(target_dir.glob("frame_*.jpg"))
            for idx, temp_frame in enumerate(temp_frames):
                timestamp_ms = int((idx / fps_value) * 1000)
                new_name = target_dir / f"frame_{timestamp_ms}ms.jpg"
                temp_frame.rename(new_name)
                print(f"Renamed {temp_frame.name} -> {new_name.name}")
    elif sampling_strategy == "time" and every_n_seconds and every_n_seconds > 0:
        fps_value = max(0.001, 1 / every_n_seconds)
        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_source,
            "-vf",
            f"fps=fps={fps_value}",
            "-frames:v",
            str(max_frames),
            str(output_pattern),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        # Rename frames to include timestamps in milliseconds
        temp_frames = sorted(target_dir.glob("frame_*.jpg"))
        for idx, temp_frame in enumerate(temp_frames):
            timestamp_ms = int(idx * every_n_seconds * 1000)
            new_name = target_dir / f"frame_{timestamp_ms}ms.jpg"
            temp_frame.rename(new_name)
    elif sampling_strategy == "fps" and fps:
        fps_value = max(0.001, fps)
        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_source,
            "-vf",
            f"fps=fps={fps_value}",
            "-frames:v",
            str(max_frames),
            str(output_pattern),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        # Rename frames to include timestamps in milliseconds
        temp_frames = sorted(target_dir.glob("frame_*.jpg"))
        for idx, temp_frame in enumerate(temp_frames):
            timestamp_ms = int((idx / fps_value) * 1000)
            new_name = target_dir / f"frame_{timestamp_ms}ms.jpg"
            temp_frame.rename(new_name)

    # Collect extracted frames
    local_frames = sorted(target_dir.glob("frame_*.jpg"))
    print(f"Collected {len(local_frames)} frames from {target_dir}")

    if upload_to_gcs:
        # Upload frames to GCS
        gcs_frames: list[str] = []
        try:
            print(f"Uploading {len(local_frames)} frames to GCS...")
            client = storage.Client()
            bucket = client.bucket("clickmoment-prod-assets")

            for frame_path in local_frames:
                # Upload to: clickmoment-prod-assets/projects/{project_id}/signals/frames/{filename}
                blob_path = f"projects/{project_id}/signals/frames/{frame_path.name}"
                blob = bucket.blob(blob_path)

                print(f"Uploading {frame_path.name} to gs://clickmoment-prod-assets/{blob_path}")
                blob.upload_from_filename(str(frame_path))

                # Return GCS URL
                gcs_url = f"gs://clickmoment-prod-assets/{blob_path}"
                gcs_frames.append(gcs_url)

                # Delete local file after upload
                frame_path.unlink()

            print(f"Successfully uploaded {len(gcs_frames)} frames to GCS")
            return gcs_frames
        except Exception as e:
            print(f"ERROR: Failed to upload frames to GCS: {e}")
            import traceback

            traceback.print_exc()
            # Fallback to local paths
            return local_frames
    else:
        return local_frames


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
