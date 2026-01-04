"""Audio and video processing functions.

This module contains reusable processing functions for audio analysis,
video analysis, and other processing tasks that can be used across
different endpoints and orchestration functions.
"""

import asyncio
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

from google.cloud import storage

from app.audio.extraction import (
    analyze_audio_features,
    extract_audio_from_video,
    transcribe_speech_audio,
)
from app.audio.saliency import detect_audio_saliency
from app.audio.speech_semantics import analyze_speech
from app.constants import (
    ADAPTIVE_INTERVAL_MEDIUM_THRESHOLD,
    ADAPTIVE_INTERVAL_SCALE_MAX,
    ADAPTIVE_INTERVAL_SHORT_THRESHOLD,
    ADAPTIVE_SAMPLE_INTERVAL_MAX,
    ADAPTIVE_SAMPLE_INTERVAL_MEAN,
    ADAPTIVE_SAMPLE_INTERVAL_MIN,
    ADAPTIVE_SAMPLE_INTERVAL_REF_DURATION,
    ADAPTIVE_SAMPLE_INTERVAL_SCALE,
    ADAPTIVE_SAMPLE_INTERVAL_STD_DEV,
    BASE_DENSE_INTERVAL_CRITICAL,
    BASE_DENSE_INTERVAL_HIGH,
    BASE_DENSE_INTERVAL_LOW,
    BASE_DENSE_INTERVAL_MEDIUM,
    BASE_DENSE_INTERVAL_MINIMAL,
    DEFAULT_FRAME_INTERVAL_FALLBACK,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_VIDEO_FPS,
    FFMPEG_OUTPUT_PATTERN,
    FPS_MULTIPLIER_MIN_INTERVAL,
    LOCAL_MEDIA_DIR,
    MAX_CONCURRENT_SEGMENT_EXTRACTIONS,
    MAX_IMPORTANCE_SEGMENT_DURATION,
    MIN_IMPORTANCE_SEGMENT_DURATION,
    SAMPLE_FRAMES_DIR,
    SHORT_SEGMENT_THRESHOLD,
    TEMP_DIR_NAME,
    VISUAL_CHANGE_HIGH_THRESHOLD,
    VISUAL_CHANGE_LOW_THRESHOLD,
)
from app.models import SourceMedia
from app.utils.storage import (
    download_json_from_gcs,
    upload_frame_to_gcs,
    upload_json_to_gcs,
)
from app.vision.extraction import generate_signed_url, get_video_fps
from app.vision.feature_analysis import compute_vision_features_batch
from app.vision.visual_change import analyze_visual_changes

logger = logging.getLogger(__name__)


def get_adaptive_intervals(video_duration: float, video_fps: float = 30.0) -> dict[str, float]:
    """
    Calculate adaptive sampling intervals based on video duration and FPS.

    Sampling intervals scale with video duration to prevent excessive frame extraction
    on longer videos while maintaining quality on shorter videos.

    Scaling Strategy:
    - Short videos (< 120s): No scaling (scale = 1.0)
    - Medium videos (120-600s): Linear scale 1.0x → 2.0x
    - Long videos (> 600s): Linear scale 2.0x → 3.0x (capped at 3.0x)

    Constraints:
    - Never sample faster than 1.5x the video's native frame duration
    - Ensures FPS-aware sampling to prevent duplicate frames

    Args:
        video_duration: Total video duration in seconds
        video_fps: Video frame rate (frames per second), defaults to 30.0

    Returns:
        Dictionary containing:
            - critical: Interval for critical importance segments
            - high: Interval for high importance segments
            - medium: Interval for medium importance segments
            - low: Interval for low importance segments
            - minimal: Interval for minimal importance segments
            - min_interval: Minimum interval based on FPS
            - scale_factor: Applied scaling factor

    Example:
        >>> get_adaptive_intervals(60, 30)  # 1 min video, 30 fps
        {'critical': 0.3, 'high': 0.5, ..., 'scale_factor': 1.0}
        >>> get_adaptive_intervals(900, 30)  # 15 min video, 30 fps
        {'critical': 0.7, 'high': 1.17, ..., 'scale_factor': 2.33}
    """
    # Calculate minimum interval from video FPS (never sample faster than 1.5x frame duration)
    min_interval = (1.0 / video_fps) * FPS_MULTIPLIER_MIN_INTERVAL

    # Calculate scaling factor based on video duration
    if video_duration <= ADAPTIVE_INTERVAL_SHORT_THRESHOLD:  # < 120s (2 min)
        scale = 1.0
    elif video_duration <= ADAPTIVE_INTERVAL_MEDIUM_THRESHOLD:  # 120-600s (2-10 min)
        # Linear scale from 1.0x to 2.0x
        scale = (
            1.0
            + (
                (video_duration - ADAPTIVE_INTERVAL_SHORT_THRESHOLD)
                / (ADAPTIVE_INTERVAL_MEDIUM_THRESHOLD - ADAPTIVE_INTERVAL_SHORT_THRESHOLD)
            )
            * 1.0
        )
    else:  # > 600s (10 min)
        # Linear scale from 2.0x to 3.0x, capped at ADAPTIVE_INTERVAL_SCALE_MAX
        scale = min(
            ADAPTIVE_INTERVAL_SCALE_MAX,
            2.0 + ((video_duration - ADAPTIVE_INTERVAL_MEDIUM_THRESHOLD) / 1800) * 1.0,
        )

    # Apply scaling to base intervals and enforce minimum interval
    return {
        "critical": max(BASE_DENSE_INTERVAL_CRITICAL * scale, min_interval),
        "high": max(BASE_DENSE_INTERVAL_HIGH * scale, min_interval),
        "medium": max(BASE_DENSE_INTERVAL_MEDIUM * scale, min_interval),
        "low": max(BASE_DENSE_INTERVAL_LOW * scale, min_interval),
        "minimal": max(BASE_DENSE_INTERVAL_MINIMAL * scale, min_interval),
        "min_interval": min_interval,
        "scale_factor": scale,
    }


async def _extract_sample_frames_to_temp(
    content_sources: SourceMedia,
    project_id: str,
    video_url: str | None = None,
    fixed_interval: float | None = None,
) -> list[Path]:
    """
    Extract sample frames for moment importance analysis.

    Frames are stored locally for processing.

    Args:
        content_sources: Video source
        project_id: Project ID
        video_url: Optional pre-generated signed URL
        fixed_interval: Fixed interval in seconds between frames (if None, uses fallback)

    Returns:
        List of local frame paths
    """
    video_path = content_sources.video_path

    # Setup ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return []

    # Use pre-generated signed URL if provided, otherwise generate one
    if video_url:
        # Reuse pre-generated signed URL (avoids duplicate GCS API calls)
        pass  # video_url already set
    elif video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    # Create local output directory (use absolute path for Cloud Run compatibility)
    output_dir = (Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / SAMPLE_FRAMES_DIR).resolve()

    # Clean directory if it exists (remove stale files from previous runs)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames evenly distributed
    output_pattern = output_dir / FFMPEG_OUTPUT_PATTERN

    # Determine sampling interval
    if fixed_interval is not None:
        interval = fixed_interval
    else:
        interval = DEFAULT_FRAME_INTERVAL_FALLBACK

    # Extract frames at fixed interval (no frame limit - extracts for full video duration)
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-i",
        video_url,
        "-vf",
        f"fps=1/{interval}",
        str(output_pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    # Check if ffmpeg succeeded
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        logger.error("ffmpeg failed to extract frames: %s", error_msg)
        return []

    # Collect and rename frames with millisecond timestamps
    sample_frames = sorted(output_dir.glob("sample_*.jpg"))

    if not sample_frames:
        logger.warning("No frames extracted by ffmpeg from %s", video_url)
        return []
    local_frames = []

    for idx, frame_path in enumerate(sample_frames):
        timestamp_ms = int(idx * interval * 1000)
        # Use 7-digit padding for timestamps (supports videos up to ~2.7 hours)
        new_name = output_dir / f"sample_{timestamp_ms:07d}ms.jpg"
        try:
            frame_path.rename(new_name)
            local_frames.append(new_name)
        except Exception as e:
            logger.warning("Failed to rename frame %s to %s: %s", frame_path, new_name, e)

    logger.debug("Extracted %d sample frames for importance analysis", len(local_frames))
    return local_frames


async def process_audio_analysis(
    video_path: str,
    project_id: str,
    video_url: str | None = None,
) -> dict[str, Any]:
    """
    Process complete audio analysis pipeline: extraction, transcription, and Stream A/B analysis.

    This function handles:
    1. Audio extraction from video
    2. Speech transcription and diarization
    3. Stream A: Speech semantics analysis (tone, emotion, narrative context)
    4. Stream B: Audio saliency detection (energy peaks, spectral changes, etc.)

    Args:
        video_path: Path to video file (GCS URL or local path)
        project_id: Project identifier for organizing outputs
        video_url: Optional pre-generated signed URL (if None, will be generated)

    Returns:
        Dictionary containing:
            - audio_result: Audio extraction results (speech, full_audio paths, segments)
            - transcription_result: Transcription results (transcript, segments, speakers)
            - stream_a_results: Speech semantics analysis (list of segments with tone, emotion, narrative)
            - stream_b_results: Audio saliency analysis (list of segments with saliency features)
            - audio_features: Raw audio features (list of segments with pitch, energy, etc.)
            - stats: Timing statistics:
                - audio_extraction_time: Time for audio extraction
                - transcription_time: Time for transcription
                - audio_breakdown_time: Total time (extraction + transcription)
                - audio_analysis_time: Time for Stream A + B analysis
    """
    stats = {
        "audio_extraction_time": 0.0,
        "transcription_time": 0.0,
        "audio_breakdown_time": 0.0,
        "audio_analysis_time": 0.0,
    }

    # ========================================================================
    # STAGE 1: Audio Breakdown
    # ========================================================================
    logger.info("Step 1/5: Audio Extraction...")
    step_start = time.time()

    source_media = SourceMedia(video_path=video_path)
    audio_result = await extract_audio_from_video(
        content_sources=source_media,
        project_id=project_id,
        max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS,
        video_url=video_url,
    )

    if not audio_result:
        raise ValueError("Failed to extract audio from video")

    stats["audio_extraction_time"] = time.time() - step_start
    logger.info("Audio extraction complete in %.2fs", stats["audio_extraction_time"])

    # Use speech lane for transcription (better accuracy)
    speech_path = audio_result.get("speech")  # May be None if no speech detected

    logger.info("Step 2/5: Speech Audio Transcription and Diarization...")
    transcription_start = time.time()
    # Only transcribe if speech was detected
    transcription_result = None
    if speech_path:
        transcription_result = await transcribe_speech_audio(
            audio_path=speech_path,  # Use speech-only lane for transcription
            project_id=project_id,
            language=None,  # Auto-detect language
            save_timeline=True,
        )
        stats["transcription_time"] = time.time() - transcription_start
        logger.info(
            "Transcription complete: %d segments, %d speakers (%.2fs)",
            len(transcription_result.get("segments", [])),
            len(transcription_result.get("speakers", [])),
            stats["transcription_time"],
        )
    else:
        stats["transcription_time"] = time.time() - transcription_start
        logger.info("No speech detected - skipping transcription")

    # Combined audio breakdown time (extraction + transcription)
    stats["audio_breakdown_time"] = stats["audio_extraction_time"] + stats["transcription_time"]

    # ========================================================================
    # STAGE 2: Stream A + B: Audio Analysis (run in parallel)
    # ========================================================================
    logger.info("Step 3/5: Running Stream A (Speech Semantics) + Stream B (Audio Saliency)...")

    audio_analysis_start = time.time()

    # Store audio_features outside async scope for later use
    audio_features: list[dict[str, Any]] | None = None

    # Define async tasks for parallel execution
    async def run_stream_a():
        """Stream A: Speech Semantics Analysis"""
        if transcription_result and transcription_result.get("transcript") and speech_path:
            try:
                # Use transcript_segments for both - they're the same segments now
                transcript_segs = transcription_result.get("segments", [])

                speech_segments = await analyze_speech(
                    audio_path=speech_path,
                    segments=transcript_segs,  # Same segments used for both tone detection and narrative analysis
                )
                return speech_segments
            except Exception as e:
                logger.error("Stream A failed: %s", e, exc_info=True)
                return None
        else:
            logger.info("Stream A skipped: No speech detected or transcription failed")
            return None

    async def run_stream_b():
        """Stream B: Audio Saliency Detection"""
        nonlocal audio_features
        full_audio_path = audio_result.get("full_audio")
        if full_audio_path:
            try:
                audio_features = await analyze_audio_features(full_audio_path)
                audio_saliency_results = detect_audio_saliency(
                    audio_features=audio_features,
                    speech_segments=audio_result.get("segments"),
                )
                return audio_saliency_results
            except Exception as e:
                logger.error("Stream B failed: %s", e, exc_info=True)
                return None
        else:
            logger.info("Stream B skipped: No full audio available")
            return None

    # Run Stream A and Stream B in parallel
    stream_a_results, stream_b_results = await asyncio.gather(
        run_stream_a(),
        run_stream_b(),
    )

    stats["audio_analysis_time"] = time.time() - audio_analysis_start

    return {
        "audio_result": audio_result,
        "transcription_result": transcription_result,
        "stream_a_results": stream_a_results,
        "stream_b_results": stream_b_results,
        "audio_features": audio_features,
        "stats": stats,
    }


async def process_initial_vision_analysis(
    video_path: str,
    project_id: str,
    video_duration: float,
    video_url: str | None = None,
) -> dict[str, Any]:
    """
    Process initial vision analysis: sparse frame sampling and visual change detection.

    This function handles:
    1. Calculate adaptive sampling interval based on video duration
    2. Extract sparse sample frames at adaptive intervals
    3. Analyze visual changes (shot/layout changes and motion spikes) on sample frames

    Args:
        video_path: Path to video file (GCS URL or local path)
        project_id: Project identifier for organizing outputs
        video_duration: Total duration of video in seconds
        video_url: Optional pre-generated signed URL (if None, will be generated)

    Returns:
        Dictionary containing:
            - visual_frames: List of frame dictionaries, each with:
                - time: Timestamp in seconds
                - filename: Frame file path
                - shot_change: Shot/layout change score [0, 1]
                - motion_score: Motion score [0, 1]
                - motion_spike: Boolean spike indicator
            - sample_interval: Calculated sampling interval in seconds
            - stats: Timing statistics:
                - initial_sampling_time: Time for sparse frame extraction
                - visual_analysis_time: Time for visual change analysis
    """
    stats = {
        "initial_sampling_time": 0.0,
        "visual_analysis_time": 0.0,
    }

    # ========================================================================
    # Step 1: Calculate Adaptive Sampling Interval
    # ========================================================================
    if video_duration > 0:
        log_duration = math.log(max(video_duration, 10) / ADAPTIVE_SAMPLE_INTERVAL_REF_DURATION)
        z_score = log_duration * ADAPTIVE_SAMPLE_INTERVAL_SCALE
        sample_interval = ADAPTIVE_SAMPLE_INTERVAL_MEAN + z_score * ADAPTIVE_SAMPLE_INTERVAL_STD_DEV
        sample_interval = max(
            ADAPTIVE_SAMPLE_INTERVAL_MIN,
            min(ADAPTIVE_SAMPLE_INTERVAL_MAX, sample_interval),
        )
    else:
        sample_interval = ADAPTIVE_SAMPLE_INTERVAL_MEAN

    # ========================================================================
    # Step 2: Extract Sparse Sample Frames
    # ========================================================================
    step_start = time.time()

    source_media = SourceMedia(video_path=video_path)
    local_sample_frames = await _extract_sample_frames_to_temp(
        content_sources=source_media,
        project_id=project_id,
        video_url=video_url,
        fixed_interval=sample_interval,
    )

    if not local_sample_frames:
        raise ValueError("Failed to extract sample frames")

    stats["initial_sampling_time"] = time.time() - step_start

    # ========================================================================
    # Step 2.5: Upload Sample Frames to GCS
    # ========================================================================
    logger.info("Uploading %d sample frames to GCS...", len(local_sample_frames))

    client = storage.Client()
    bucket = client.bucket("clickmoment-prod-assets")

    uploaded_count = 0
    for frame_path in local_sample_frames:
        # Check if file exists before uploading
        if not frame_path.exists():
            logger.warning("Skipping missing frame: %s", frame_path)
            continue

        # Upload to GCS with timestamp-based name
        blob_path = f"projects/{project_id}/signals/vision/sample_frames/{frame_path.name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(frame_path), content_type="image/jpeg")
        uploaded_count += 1

    logger.info("Uploaded %d/%d sample frames to GCS", uploaded_count, len(local_sample_frames))

    if uploaded_count < len(local_sample_frames):
        logger.warning(
            "Missing %d frames from extraction", len(local_sample_frames) - uploaded_count
        )

    # ========================================================================
    # Step 3: Visual Change Analysis
    # ========================================================================
    step_start = time.time()

    visual_frames = analyze_visual_changes(local_sample_frames)

    stats["visual_analysis_time"] = time.time() - step_start

    # ========================================================================
    # Step 4: Upload Results to GCS
    # ========================================================================
    upload_json_to_gcs(
        data=visual_frames,
        project_id=project_id,
        directory="signals/vision",
        filename="initial_sample_visual_frames.json",
    )

    return {
        "visual_frames": visual_frames,
        "sample_interval": sample_interval,
        "stats": stats,
    }


async def extract_dense_frames_parallel(
    importance_segments: list[dict[str, Any]],
    video_path: str,
    project_id: str,
    video_url: str | None = None,
    bucket_name: str = "clickmoment-prod-assets",
) -> dict[str, Any]:
    """
    Extract frames at different densities based on importance segments (parallel processing).

    For each importance segment, extract frames at the recommended interval and upload to GCS.
    All segments are processed in parallel for maximum speed.

    Args:
        importance_segments: List of importance segments from identify_important_moments()
        video_path: Path to video file (GCS URL or local path)
        project_id: Project identifier for organizing outputs
        video_url: Optional pre-generated signed URL (if None, will be generated)
        bucket_name: GCS bucket name for frame uploads

    Returns:
        Dictionary containing:
            - frames: List of extracted frame dictionaries:
                - time: Timestamp in seconds
                - filename: Local path to frame
                - gcs_url: GCS URL of uploaded frame
                - importance_level: Level from source segment
                - trigger_count: Trigger count from source segment
                - segment_index: Which importance segment this came from
            - stats: Processing statistics:
                - total_frames: Total frames extracted
                - extraction_time: Total time for parallel extraction
                - segments_processed: Number of segments processed
    """
    stats = {
        "total_frames": 0,
        "extraction_time": 0.0,
        "segments_processed": 0,
    }

    start_time = time.time()

    # Setup ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise ValueError("ffmpeg not found in PATH")

    # Use pre-generated signed URL if provided, otherwise generate one
    if video_url:
        pass  # video_url already set
    elif video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    # Create output directory (use absolute path for Cloud Run compatibility)
    output_dir = (Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / "dense_frames").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define async extraction function for a single segment
    async def extract_segment(segment: dict[str, Any], segment_index: int) -> list[dict[str, Any]]:
        """Extract frames for a single importance segment."""
        start = segment["start_time"]
        end = segment["end_time"]
        duration = end - start
        interval = segment["dense_sample_interval"]
        importance_level = segment["importance_level"]

        # Skip segments with zero or negative duration (invalid time range)
        if duration <= 0:
            logger.debug(
                "Skipping segment %d: invalid duration (start=%.2fs, end=%.2fs, duration=%.2fs)",
                segment_index,
                start,
                end,
                duration,
            )
            return []

        # Create output pattern with segment index in filename for uniqueness
        output_pattern = output_dir / f"frame_seg{segment_index:03d}_%04d.jpg"

        # If sampling interval >= segment duration, extract single frame at midpoint
        # This handles both short segments (interval == duration) and intervals larger than duration
        if interval >= duration:
            midpoint = start + (duration / 2.0)
            # Extract single frame at midpoint
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                "-ss",
                str(midpoint),
                "-i",
                video_url,
                "-vframes",
                "1",
                "-c:v",
                "mjpeg",  # Explicitly use MJPEG encoder
                "-pix_fmt",
                "yuvj420p",  # Compatible pixel format for JPEG
                "-q:v",
                "2",  # High quality
                "-f",
                "image2",  # Image output format
                str(output_pattern),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Normal case: extract multiple frames at specified interval
            vf_filter = f"fps=1/{interval}"
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                video_url,
                "-vf",
                vf_filter,
                "-c:v",
                "mjpeg",  # Explicitly use MJPEG encoder
                "-pix_fmt",
                "yuvj420p",  # Compatible pixel format for JPEG
                "-q:v",
                "2",  # High quality
                "-f",
                "image2",  # Image output format
                str(output_pattern),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            # Extract relevant error information for diagnostics
            error_summary = {
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "duration": duration,
                "interval": interval,
                "returncode": process.returncode,
            }

            if stderr:
                stderr_text = stderr.decode()
                lines = stderr_text.strip().split("\n")

                # Extract key error indicators
                error_indicators = []
                for line in lines:
                    line_lower = line.lower()
                    if (
                        "error" in line_lower
                        or "failed" in line_lower
                        or "nothing was written" in line_lower
                    ):
                        # Clean up the line
                        clean_line = line.strip()
                        if clean_line and not clean_line.startswith("ffmpeg"):
                            error_indicators.append(clean_line)

                if error_indicators:
                    error_summary["errors"] = error_indicators[-3:]  # Last 3 errors

                # Check for specific failure patterns
                stderr_lower = stderr_text.lower()
                if "encoder" in stderr_lower and "failed" in stderr_lower:
                    error_summary["likely_cause"] = (
                        "Encoder initialization failed (possibly HEVC/H.265 format issue)"
                    )
                elif "nothing was written" in stderr_lower:
                    error_summary["likely_cause"] = (
                        "No frames extracted (seek point may be invalid or video format issue)"
                    )
                elif "no filtered frames" in stderr_lower:
                    error_summary["likely_cause"] = (
                        "Fps filter produced no frames (ffmpeg internal issue)"
                    )

            logger.warning(
                "Segment %d extraction failed: duration=%.2fs, interval=%.2fs, start=%.2fs, end=%.2fs. "
                "Cause: %s. Errors: %s",
                segment_index,
                duration,
                interval,
                start,
                end,
                error_summary.get("likely_cause", "Unknown"),
                error_summary.get("errors", ["No error details"])[:2],  # First 2 errors
            )
            return []

        # Collect extracted frames, upload to GCS, and add metadata
        extracted_files = sorted(output_dir.glob(f"frame_seg{segment_index:03d}_*.jpg"))
        frames = []

        for frame_idx, frame_path in enumerate(extracted_files):
            # Calculate actual timestamp for this frame
            if interval >= duration:
                # Single frame at midpoint
                frame_time = start + (duration / 2.0)
            else:
                # Multiple frames at interval
                frame_time = start + (frame_idx * interval)
            timestamp_ms = int(frame_time * 1000)

            # Upload to GCS
            gcs_url = upload_frame_to_gcs(
                file_path=frame_path,
                project_id=project_id,
                segment_start=start,
                segment_end=end,
                timestamp_ms=timestamp_ms,
                bucket_name=bucket_name,
            )

            frames.append(
                {
                    "time": frame_time,
                    "filename": str(frame_path),
                    "gcs_url": gcs_url or "",
                    "importance_level": importance_level,
                    "trigger_count": segment.get("trigger_count", 0.0),
                    "segment_index": segment_index,
                }
            )

        logger.debug(
            "Segment %d [%.2fs-%.2fs] (%.2fs): extracted %d frames",
            segment_index,
            start,
            end,
            duration,
            len(frames),
        )
        return frames

    # Extract all segments in parallel with concurrency limit
    logger.info(
        "Extracting dense frames from %d importance segments (parallel, max %d concurrent)",
        len(importance_segments),
        MAX_CONCURRENT_SEGMENT_EXTRACTIONS,
    )

    # Create semaphore to limit concurrent extractions
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEGMENT_EXTRACTIONS)

    async def extract_with_limit(
        segment: dict[str, Any], segment_index: int
    ) -> list[dict[str, Any]]:
        """Extract segment with concurrency limit."""
        async with semaphore:
            return await extract_segment(segment, segment_index)

    tasks = [extract_with_limit(segment, idx) for idx, segment in enumerate(importance_segments)]

    results = await asyncio.gather(*tasks)

    # Flatten results and sort by time
    all_frames = []
    for segment_frames in results:
        all_frames.extend(segment_frames)

    all_frames.sort(key=lambda f: f["time"])

    stats["total_frames"] = len(all_frames)
    stats["segments_processed"] = len(importance_segments)
    stats["extraction_time"] = time.time() - start_time

    # Log summary statistics
    if all_frames:
        avg_frames_per_segment = (
            len(all_frames) / len(importance_segments) if importance_segments else 0
        )
        logger.info(
            "Frame extraction complete: %d frames from %d segments (%.1f frames/segment, %.2fs)",
            len(all_frames),
            len(importance_segments),
            avg_frames_per_segment,
            stats["extraction_time"],
        )
    else:
        logger.warning(
            "No frames extracted from %d segments",
            len(importance_segments),
        )

    return {
        "frames": all_frames,
        "stats": stats,
    }


def _normalize_input_data(data: Any, name: str) -> list[dict[str, Any]] | None:
    """Parse JSON strings and normalize to list format."""
    if data is None:
        return None

    # Parse JSON string if needed
    if isinstance(data, str):
        if not data.strip():
            return None
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s JSON string", name)
            return None

    # Convert to list if needed
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    logger.warning(
        "%s is not a valid format (type: %s), treating as None", name, type(data).__name__
    )
    return None


def _load_from_gcs(
    project_id: str | None,
    directory: str,
    filename: str,
    bucket_name: str,
    name: str,
) -> list[dict[str, Any]]:
    """Load JSON data from GCS, returning empty list if not found."""
    if not project_id or project_id.lower() in ["string", "none", "null", ""]:
        if project_id:
            logger.warning("⚠ project_id appears to be a placeholder ('%s')", project_id)
        return []

    data = download_json_from_gcs(
        project_id=project_id,
        directory=directory,
        filename=filename,
        bucket_name=bucket_name,
    )

    if data is None:
        return []

    if isinstance(data, list):
        if data:
            logger.info("Loaded %s from GCS: %d items", name, len(data))
        return data

    logger.warning("Loaded %s from GCS but unexpected type: %s", name, type(data).__name__)
    return []


def _load_visual_frames_from_gcs(
    project_id: str | None,
    bucket_name: str,
) -> list[dict[str, Any]]:
    """Load visual frames from GCS."""
    if not project_id or project_id.lower() in ["string", "none", "null", ""]:
        return []

    # Load initial_sample_visual_frames.json
    data = download_json_from_gcs(
        project_id=project_id,
        directory="signals/vision",
        filename="initial_sample_visual_frames.json",
        bucket_name=bucket_name,
    )

    if data is not None and isinstance(data, list):
        logger.info(
            "Loaded visual frames from initial_sample_visual_frames.json: %d frames", len(data)
        )
        return data

    return []


def _create_visual_boundaries(
    visual_frames: list[dict[str, Any]], video_duration: float
) -> list[float]:
    """
    Create visual boundaries from shot changes.

    Args:
        visual_frames: Visual frames with shot_change scores
        video_duration: Total video duration

    Returns:
        Sorted list of time boundaries (includes 0.0 and video_duration)
    """
    visual_boundaries = [0.0]

    for frame in visual_frames:
        shot_change = frame.get("shot_change", 0.0)
        if shot_change >= VISUAL_CHANGE_HIGH_THRESHOLD:
            visual_boundaries.append(frame.get("time", 0.0))

    visual_boundaries.append(video_duration)
    visual_boundaries = sorted(set(visual_boundaries))

    # Fallback: if no boundaries detected, create segments at fixed intervals
    if len(visual_boundaries) <= 2:
        logger.warning("No visual boundaries detected, creating segments at fixed intervals")
        interval = MAX_IMPORTANCE_SEGMENT_DURATION
        num_segments = max(1, int(video_duration / interval))
        visual_boundaries = [i * interval for i in range(num_segments + 1)]
        if visual_boundaries[-1] < video_duration:
            visual_boundaries.append(video_duration)
        visual_boundaries = sorted(set(visual_boundaries))

    return visual_boundaries


def _create_initial_segments(
    visual_boundaries: list[float],
    visual_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create initial segments from visual boundaries.

    Args:
        visual_boundaries: List of time boundaries
        visual_frames: Visual frames for calculating avg visual change

    Returns:
        List of segment dictionaries with start, end, duration, avg_visual_change
    """
    # Extract visual change points for efficiency
    visual_change_points = [
        {"time": frame.get("time", 0.0), "shot_change": frame.get("shot_change", 0.0)}
        for frame in visual_frames
        if frame.get("shot_change", 0.0) >= VISUAL_CHANGE_HIGH_THRESHOLD
    ]

    initial_segments = []
    for i in range(len(visual_boundaries) - 1):
        start = visual_boundaries[i]
        end = visual_boundaries[i + 1]

        # Calculate average visual change in this segment
        segment_visual_changes = [
            vc["shot_change"] for vc in visual_change_points if start <= vc["time"] <= end
        ]
        avg_visual_change = (
            sum(segment_visual_changes) / len(segment_visual_changes)
            if segment_visual_changes
            else 0.0
        )

        initial_segments.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "avg_visual_change": avg_visual_change,
            }
        )

    return initial_segments


def _merge_low_change_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge segments with low visual change.

    Args:
        segments: List of initial segments

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    merged_segments = []
    current_segment = segments[0].copy()

    for next_segment in segments[1:]:
        current_duration = current_segment["end"] - current_segment["start"]
        current_visual_change = current_segment["avg_visual_change"]
        next_visual_change = next_segment["avg_visual_change"]
        next_duration = next_segment["end"] - next_segment["start"]
        combined_duration = current_duration + next_duration

        # Merge if:
        # 1. Current segment is below minimum duration, OR
        # 2. Both have low visual change and combined duration < max
        should_merge = False
        if (
            current_duration < MIN_IMPORTANCE_SEGMENT_DURATION
            and combined_duration <= MAX_IMPORTANCE_SEGMENT_DURATION
        ):
            should_merge = True
        elif (
            current_visual_change < VISUAL_CHANGE_LOW_THRESHOLD
            and next_visual_change < VISUAL_CHANGE_LOW_THRESHOLD
            and combined_duration <= MAX_IMPORTANCE_SEGMENT_DURATION
        ):
            should_merge = True

        if should_merge:
            # Merge: extend current segment and recalculate avg visual change
            prev_duration = current_segment["end"] - current_segment["start"]
            current_segment["end"] = next_segment["end"]
            total_duration = current_segment["end"] - current_segment["start"]
            current_segment["avg_visual_change"] = (
                (current_visual_change * prev_duration + next_visual_change * next_duration)
                / total_duration
                if total_duration > 0
                else 0.0
            )
        else:
            # Keep separate
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()

    merged_segments.append(current_segment)

    # Iterative merging pass: keep merging until no short segments remain
    # This handles consecutive short segments and ensures NO segment is below minimum duration
    final_merged = merged_segments
    max_iterations = 10  # Prevent infinite loops (should never hit this in practice)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        had_short_segments = False
        temp_merged = []

        if not final_merged:
            break

        current = final_merged[0].copy()

        for next_seg in final_merged[1:]:
            current_dur = current["end"] - current["start"]

            # If current segment is too short, always merge with next
            if current_dur < MIN_IMPORTANCE_SEGMENT_DURATION:
                had_short_segments = True
                # Merge with next segment
                next_dur = next_seg["end"] - next_seg["start"]
                current["end"] = next_seg["end"]
                total_dur = current["end"] - current["start"]
                current["avg_visual_change"] = (
                    (
                        current["avg_visual_change"] * current_dur
                        + next_seg["avg_visual_change"] * next_dur
                    )
                    / total_dur
                    if total_dur > 0
                    else 0.0
                )
            else:
                # Current is long enough, save it and move to next
                temp_merged.append(current)
                current = next_seg.copy()

        # Handle the last segment
        current_dur = current["end"] - current["start"]
        if current_dur < MIN_IMPORTANCE_SEGMENT_DURATION and temp_merged:
            had_short_segments = True
            # Merge with previous segment
            prev = temp_merged[-1]
            prev_dur = prev["end"] - prev["start"]
            prev["end"] = current["end"]
            total_dur = prev["end"] - prev["start"]
            prev["avg_visual_change"] = (
                (prev["avg_visual_change"] * prev_dur + current["avg_visual_change"] * current_dur)
                / total_dur
                if total_dur > 0
                else 0.0
            )
        else:
            temp_merged.append(current)

        final_merged = temp_merged

        # If no short segments were found, we're done
        if not had_short_segments:
            break

    if iteration > 1:
        logger.debug("Iterative merging completed in %d iterations", iteration)

    return final_merged


def _split_oversized_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split segments that exceed maximum duration.

    Args:
        segments: List of merged segments

    Returns:
        List of final segments with no segment exceeding MAX_IMPORTANCE_SEGMENT_DURATION
    """
    final_segments = []
    segments_split = 0

    for segment in segments:
        duration = segment["end"] - segment["start"]
        if duration > MAX_IMPORTANCE_SEGMENT_DURATION:
            num_splits = int(duration / MAX_IMPORTANCE_SEGMENT_DURATION) + 1
            split_duration = duration / num_splits
            logger.info(
                "Splitting segment [%.2f - %.2f]s (%.2fs) into %d segments",
                segment["start"],
                segment["end"],
                duration,
                num_splits,
            )

            for i in range(num_splits):
                split_start = segment["start"] + (i * split_duration)
                split_end = segment["start"] + ((i + 1) * split_duration)
                if i == num_splits - 1:
                    split_end = segment["end"]  # Ensure last segment ends exactly at original end

                final_segments.append(
                    {
                        "start": split_start,
                        "end": split_end,
                        "duration": split_end - split_start,
                        "avg_visual_change": segment["avg_visual_change"],
                    }
                )
            segments_split += 1
        else:
            final_segments.append(segment)

    if segments_split > 0:
        logger.info(
            "Split %d oversized segments into %d total segments",
            segments_split,
            len(final_segments),
        )

    return final_segments


def _calculate_segment_importance(
    segments: list[dict[str, Any]],
    stream_a_results: list[dict[str, Any]],
    stream_b_results: list[dict[str, Any]],
    visual_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Calculate importance scores for each segment by overlaying audio/narrative signals.

    Args:
        segments: List of time segments
        stream_a_results: Stream A results (narrative + tone)
        stream_b_results: Stream B results (audio saliency)
        visual_frames: Visual frames (for motion spikes)

    Returns:
        List of timeline moments with importance scores
    """
    timeline_moments = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        interval_mid = (start + end) / 2.0

        # Find overlapping signals
        narrative_score = 0.0
        base_importance = 0.0
        audio_saliency = 0.0
        motion_spike = False

        # Stream A (narrative + tone)
        for seg in stream_a_results:
            if seg.get("start", 0.0) <= interval_mid <= seg.get("end", 0.0):
                narrative_contexts = seg.get("narrative_context", [])
                if narrative_contexts and isinstance(narrative_contexts, list):
                    narrative_score = max(
                        narrative_score,
                        max(
                            ctx.get("importance", 0.0)
                            for ctx in narrative_contexts
                            if isinstance(ctx, dict)
                        ),
                    )
                base_importance = max(base_importance, seg.get("importance", 0.0))

        # Stream B (audio saliency)
        for seg in stream_b_results:
            if seg.get("start", 0.0) <= interval_mid <= seg.get("end", 0.0):
                audio_saliency = max(audio_saliency, seg.get("saliency_score", 0.0))

        # Visual frames (motion spikes)
        for frame in visual_frames:
            if start <= frame.get("time", 0.0) <= end:
                motion_spike = motion_spike or frame.get("motion_spike", False)

        timeline_moments.append(
            {
                "start": start,
                "end": end,
                "narrative_score": narrative_score,
                "base_importance": base_importance,
                "audio_saliency": audio_saliency,
                "visual_change": segment["avg_visual_change"],
                "motion_spike": motion_spike,
            }
        )

    return timeline_moments


def _assign_sampling_intervals(
    timeline_moments: list[dict[str, Any]],
    adaptive_intervals: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Assign sampling intervals based on trigger-based importance calculation.

    Args:
        timeline_moments: List of moments with importance scores
        adaptive_intervals: Adaptive intervals dict from get_adaptive_intervals()

    Returns:
        List of importance segments with assigned sampling intervals
    """
    importance_segments = []

    for moment in timeline_moments:
        # STEP 1: Convert signals to triggers (binary/soft)
        # Narrative trigger
        if moment["narrative_score"] >= 0.7:
            narrative_trigger = 1.0
        elif moment["narrative_score"] >= 0.4:
            narrative_trigger = 0.5
        else:
            narrative_trigger = 0.0

        # Audio saliency trigger
        if moment["audio_saliency"] >= 0.6:
            audio_saliency_trigger = 1.0
        elif moment["audio_saliency"] >= 0.3:
            audio_saliency_trigger = 0.5
        else:
            audio_saliency_trigger = 0.0

        # Visual change trigger
        if moment["visual_change"] >= 0.6:
            visual_trigger = 1.0
        elif moment["visual_change"] >= 0.3:
            visual_trigger = 0.5
        else:
            visual_trigger = 0.0

        # Motion spike trigger
        motion_trigger = 1.0 if moment["motion_spike"] else 0.0

        # STEP 2: Apply modality weighting
        visual_modality_trigger = max(visual_trigger, motion_trigger)
        stream_a_weighted = narrative_trigger * 0.5
        stream_b_weighted = audio_saliency_trigger * 0.5

        # STEP 3: Sum weighted triggers
        num_triggers = visual_modality_trigger * 1.0 + stream_a_weighted + stream_b_weighted

        # STEP 4: Map triggers to importance level and adaptive interval
        if num_triggers >= 1.5:
            importance_level = "critical"
            dense_interval = adaptive_intervals["critical"]
        elif num_triggers >= 1.0:
            importance_level = "high"
            dense_interval = adaptive_intervals["high"]
        elif num_triggers >= 0.5:
            importance_level = "medium"
            dense_interval = adaptive_intervals["medium"]
        elif num_triggers > 0.0:
            importance_level = "low"
            dense_interval = adaptive_intervals["low"]
        else:
            importance_level = "minimal"
            dense_interval = adaptive_intervals["minimal"]

        avg_importance = min(1.0, num_triggers / 2.0)
        segment_duration = moment["end"] - moment["start"]

        # Skip invalid segments
        if segment_duration <= 0:
            continue

        # CRITICAL FIX: Short segment handling
        if segment_duration < SHORT_SEGMENT_THRESHOLD:
            # Extract single frame at midpoint (set interval = segment duration)
            dense_interval = segment_duration
            logger.debug(
                "Short segment [%.2f-%.2f]s (%.2fs): single frame at midpoint",
                moment["start"],
                moment["end"],
                segment_duration,
            )
        else:
            # Normal segment: enforce FPS-based minimum interval
            min_interval = adaptive_intervals["min_interval"]
            if dense_interval < min_interval:
                dense_interval = min_interval

        importance_segments.append(
            {
                "start_time": moment["start"],
                "end_time": moment["end"],
                "importance_level": importance_level,
                "avg_importance": avg_importance,
                "trigger_count": num_triggers,
                "triggers": {
                    "stream_a": narrative_trigger,
                    "stream_b": audio_saliency_trigger,
                    "visual": visual_modality_trigger,
                },
                "weighted_triggers": {
                    "stream_a_weighted": stream_a_weighted,
                    "stream_b_weighted": stream_b_weighted,
                    "visual_weighted": visual_modality_trigger * 1.0,
                },
                "signal_values": {
                    "narrative_score": moment["narrative_score"],
                    "audio_saliency": moment["audio_saliency"],
                    "visual_change": moment["visual_change"],
                    "motion_spike": moment["motion_spike"],
                    "base_importance": moment["base_importance"],
                },
                "dense_sample_interval": dense_interval,
            }
        )

    return importance_segments


def _calculate_importance_segments(
    stream_a_results: list[dict[str, Any]],
    stream_b_results: list[dict[str, Any]],
    visual_frames: list[dict[str, Any]],
    video_duration: float,
    adaptive_intervals: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Calculate importance segments using visual boundaries and multi-signal fusion.

    This function orchestrates 6 focused sub-functions to:
    1. Create visual boundaries from shot changes
    2. Create initial segments from boundaries
    3. Merge segments with low visual change
    4. Split oversized segments
    5. Calculate importance scores
    6. Assign adaptive sampling intervals

    Args:
        stream_a_results: Stream A results (narrative + tone analysis)
        stream_b_results: Stream B results (audio saliency)
        visual_frames: Visual frames (shot changes, motion spikes)
        video_duration: Total video duration in seconds
        adaptive_intervals: Adaptive intervals from get_adaptive_intervals()

    Returns:
        List of importance segments with start_time, end_time, importance_level, etc.
    """
    # Step 1: Create visual boundaries
    visual_boundaries = _create_visual_boundaries(visual_frames, video_duration)

    # Step 2: Create initial segments from boundaries
    initial_segments = _create_initial_segments(visual_boundaries, visual_frames)

    # Step 3: Merge segments with low visual change
    merged_segments = _merge_low_change_segments(initial_segments)

    # Step 4: Split oversized segments
    final_segments = _split_oversized_segments(merged_segments)

    # Step 5: Calculate importance scores
    timeline_moments = _calculate_segment_importance(
        final_segments,
        stream_a_results,
        stream_b_results,
        visual_frames,
    )

    # Step 6: Assign adaptive sampling intervals
    importance_segments = _assign_sampling_intervals(
        timeline_moments,
        adaptive_intervals,
    )

    for idx, segment in enumerate(importance_segments):
        segment["segment_index"] = idx

    return importance_segments


def calculate_importance_segments_for_analysis(
    stream_a_results: list[dict[str, Any]],
    stream_b_results: list[dict[str, Any]],
    visual_frames: list[dict[str, Any]],
    video_duration: float,
    video_fps: float = DEFAULT_VIDEO_FPS,
) -> list[dict[str, Any]]:
    """
    Calculate importance segments with triggers and segment indices for analysis tooling.

    This is a lightweight wrapper around _calculate_importance_segments to avoid
    reimplementing trigger logic outside of processing.
    """
    adaptive_intervals = get_adaptive_intervals(video_duration, video_fps)
    return _calculate_importance_segments(
        stream_a_results=stream_a_results,
        stream_b_results=stream_b_results,
        visual_frames=visual_frames,
        video_duration=video_duration,
        adaptive_intervals=adaptive_intervals,
    )


async def identify_important_moments(
    video_duration: float,
    stream_a_results: list[dict[str, Any]] | None = None,
    stream_b_results: list[dict[str, Any]] | None = None,
    visual_frames: list[dict[str, Any]] | None = None,
    project_id: str | None = None,
    bucket_name: str = "clickmoment-prod-assets",
    video_path: str | None = None,
    video_url: str | None = None,
    niche: str = "general",
) -> list[dict[str, Any]]:
    """
    Identify important moments, extract frames, and compute vision features.

    This function:
    1. Identifies important moments by combining Stream A, Stream B, and vision signals
    2. Extracts dense frames for those moments
    3. Computes comprehensive vision features for each frame (including face analysis)
    4. Returns a list of frames with all features

    This function can work standalone - if results aren't provided, it fetches them from GCS.

    Args:
        video_duration: Total duration of video in seconds
        stream_a_results: Optional Stream A results (narrative + tone analysis)
        stream_b_results: Optional Stream B results (audio saliency)
        visual_frames: Optional vision analysis results (shot changes, motion)
        project_id: Project identifier
        bucket_name: GCS bucket name
        video_path: Path to video file (required for frame extraction)
        video_url: Optional pre-generated signed URL
        niche: Content niche for editability scoring

    Returns:
        List of frame dictionaries, each containing:
            - time: Timestamp in seconds
            - gcs_url: GCS URL of uploaded frame
            - local_path: Local path to frame (before cleanup)
            - segment_index: Index of source importance segment
            - importance_level: "high", "medium", or "low"
            - importance_score: Combined importance score [0, 1]
            - segment_start: Start time of source segment
            - segment_end: End time of source segment
            - face_analysis: Face analysis results
            - aesthetics: Aesthetic scores and image quality
            - editability: Editability scores
            - composition: Composition scores
            - technical_quality: Technical quality scores
            - face_quality: Face quality scores
    """
    stats = {
        "signal_loading_time": 0.0,
        "fusion_time": 0.0,
        "extraction_time": 0.0,
        "vision_feature_time": 0.0,
    }

    start_time = time.time()

    # Normalize input data (parse JSON strings, convert to lists)
    stream_a_results = _normalize_input_data(stream_a_results, "stream_a_results")
    stream_b_results = _normalize_input_data(stream_b_results, "stream_b_results")
    visual_frames = _normalize_input_data(visual_frames, "visual_frames")

    # Load from GCS if not provided
    if stream_a_results is None:
        stream_a_results = _load_from_gcs(
            project_id=project_id,
            directory="signals/audio",
            filename="stream_a_results.json",
            bucket_name=bucket_name,
            name="Stream A results",
        )

    if stream_b_results is None:
        stream_b_results = _load_from_gcs(
            project_id=project_id,
            directory="signals/audio",
            filename="stream_b_results.json",
            bucket_name=bucket_name,
            name="Stream B results",
        )

    if visual_frames is None:
        visual_frames = _load_visual_frames_from_gcs(project_id=project_id, bucket_name=bucket_name)
    else:
        logger.debug(
            "Visual frames provided in request: %d frames",
            len(visual_frames) if isinstance(visual_frames, list) else 0,
        )

    stats["signal_loading_time"] = time.time() - start_time

    # ========================================================================
    # Detect video FPS and calculate adaptive intervals
    # ========================================================================
    # Detect video frame rate for FPS-aware sampling
    video_fps = DEFAULT_VIDEO_FPS  # Default fallback
    if video_path:
        try:
            video_fps = await get_video_fps(video_path, video_url)
        except Exception as e:
            logger.warning(
                "Failed to detect video FPS: %s, using default %.1f", e, DEFAULT_VIDEO_FPS
            )

    # Calculate adaptive sampling intervals based on video duration and FPS
    adaptive_intervals = get_adaptive_intervals(video_duration, video_fps)
    logger.info(
        "Adaptive intervals (duration=%.1fs, fps=%.1f, scale=%.2fx): critical=%.2fs, high=%.2fs, medium=%.2fs, low=%.2fs, minimal=%.2fs",
        video_duration,
        video_fps,
        adaptive_intervals["scale_factor"],
        adaptive_intervals["critical"],
        adaptive_intervals["high"],
        adaptive_intervals["medium"],
        adaptive_intervals["low"],
        adaptive_intervals["minimal"],
    )

    # ========================================================================
    # Combine signals and calculate importance segments
    # ========================================================================
    fusion_start = time.time()

    # Check if we have any signals at all
    stream_a_count = len(stream_a_results) if stream_a_results else 0
    stream_b_count = len(stream_b_results) if stream_b_results else 0
    visual_count = len(visual_frames) if visual_frames else 0

    if stream_a_count == 0 and stream_b_count == 0 and visual_count == 0:
        logger.warning(
            "No signals available for importance calculation: stream_a=%d, stream_b=%d, visual_frames=%d",
            stream_a_count,
            stream_b_count,
            visual_count,
        )
        return []

    importance_segments = _calculate_importance_segments(
        stream_a_results=stream_a_results,
        stream_b_results=stream_b_results,
        visual_frames=visual_frames,
        video_duration=video_duration,
        adaptive_intervals=adaptive_intervals,
    )

    stats["fusion_time"] = time.time() - fusion_start

    # ========================================================================
    # Extract frames and compute vision features if video_path is provided
    # ========================================================================
    frames_with_features = []

    if video_path and importance_segments:
        # Extract dense frames in parallel
        extraction_start = time.time()
        try:
            extraction_result = await extract_dense_frames_parallel(
                importance_segments=importance_segments,
                video_path=video_path,
                project_id=project_id or "unknown",
                video_url=video_url,
                bucket_name=bucket_name,
            )
            extracted_frames = extraction_result["frames"]
            stats["extraction_time"] = time.time() - extraction_start
        except Exception as e:
            logger.error("Frame extraction failed: %s", e, exc_info=True)
            extracted_frames = []

        # Compute vision features for each frame
        if extracted_frames:
            vision_start = time.time()

            # Collect all valid frame paths and their indices
            frame_paths = []
            valid_indices = []  # Track which frames are valid

            for idx, frame_data in enumerate(extracted_frames):
                local_path = Path(frame_data["filename"])
                if local_path.exists():
                    frame_paths.append(local_path)
                    valid_indices.append(idx)
                else:
                    logger.warning("Local frame file not found: %s", local_path)

            # Process all valid frames in batch (creates analyzer once internally)
            vision_features_list = (
                compute_vision_features_batch(
                    frame_paths=frame_paths,
                    niche=niche,
                )
                if frame_paths
                else []
            )

            # Create a mapping from frame index to vision features
            features_map = {
                valid_indices[i]: vision_features_list[i] for i in range(len(valid_indices))
            }

            # Combine metadata with vision features
            for idx, frame_data in enumerate(extracted_frames):
                local_path = Path(frame_data["filename"])
                frame_time = frame_data["time"]
                segment_index = frame_data["segment_index"]
                segment = importance_segments[segment_index]
                gcs_url = frame_data.get("gcs_url", "")

                # Get vision features if frame was processed
                vision_features = features_map.get(idx)

                # Build frame dictionary with all features
                frame_dict = {
                    "time": frame_time,
                    "gcs_url": gcs_url or "",
                    "local_path": str(local_path.resolve())
                    if local_path.exists()
                    else str(local_path),
                    "segment_index": segment_index,
                    "importance_level": segment["importance_level"],
                    "importance_score": segment["avg_importance"],
                    "segment_start": segment["start_time"],
                    "segment_end": segment["end_time"],
                }

                # Merge vision features into frame dict (includes face_analysis)
                if vision_features:
                    frame_dict.update(vision_features)

                frames_with_features.append(frame_dict)

                # NOTE: Do NOT delete local frame files here - they are needed for thumbnail selection
                # Cleanup will happen in adaptive_sampling.py or main.py after thumbnail selection completes

            stats["vision_feature_time"] = time.time() - vision_start
            logger.info(
                "Computed vision features for %d frames (%.2fs)",
                len(frames_with_features),
                stats["vision_feature_time"],
            )

    return frames_with_features
