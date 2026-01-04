"""Adaptive frame sampling orchestrator.

Coordinates the full pipeline:
1. Audio breakdown → timeline
2. Initial sparse sampling → sample frames
3. Face analysis → expression & motion
4. Moment importance calculation → importance scores & segments
5. Adaptive extraction → dense around important moments
6. Storage → upload frames to GCS

This module orchestrates the order of operations and manages data flow.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from app.analysis.processing import (
    calculate_importance_segments_for_analysis,
    identify_important_moments,
    process_audio_analysis,
    process_initial_vision_analysis,
)
from app.analysis.segment_analyzer import select_top_segments_and_frames
from app.constants import (
    DEFAULT_VIDEO_FPS,
    DERIVED_MEDIA_DIR,
    GCS_ASSETS_BUCKET,
    LOCAL_MEDIA_DIR,
    TEMP_DIR_NAME,
)
from app.utils.storage import (
    upload_audio_file_to_gcs,
    upload_json_to_gcs,
)
from app.vision.extraction import generate_signed_url, get_video_fps

logger = logging.getLogger(__name__)

# Legacy imports removed - these functions are no longer used
# The new adaptive sampling system uses trigger-based multi-signal fusion
# See: get_adaptive_intervals() and _calculate_importance_segments() in processing.py

# ============================================================================
# CONSTANTS
# ============================================================================
# All constants moved to app.constants - imported above


async def orchestrate_adaptive_sampling(
    video_path: str,
    project_id: str,
    video_url: str | None = None,
    niche: str = "general",
) -> dict[str, Any]:
    """
    Orchestrate adaptive frame sampling pipeline.

    This is the main entry point that coordinates:
    - Audio analysis
    - Face analysis
    - Moment importance calculation
    - Adaptive frame extraction

    Storage Architecture:
        - Sample frames stored locally for ML processing (face analysis)
        - Extracted frames uploaded to GCS immediately after extraction
        - All local files cleaned up after GCS upload
        - Only GCS URLs returned in results

    Args:
        video_path: GCS URL or local path to video
        project_id: Project identifier for organizing outputs
        video_url: Optional pre-generated signed URL

    Returns:
        Dictionary with:
            - frames: List of GCS URLs for extracted frames
            - importance_segments: List of importance segments with metadata
            - audio_timeline: Deprecated - use Stream A/B timelines instead
            - processing_stats: Timing and performance metrics
            - summary: Human-readable summary
    """
    start_time = time.time()
    stats = {
        "audio_breakdown_time": 0.0,
        "initial_sampling_time": 0.0,
        "visual_analysis_time": 0.0,
        "importance_calculation_time": 0.0,
        "adaptive_extraction_time": 0.0,
        "total_time": 0.0,
    }

    logger.info("Starting adaptive sampling for %s", video_path)

    # Generate signed URL once at the start (reuse for all operations)
    # This avoids multiple GCS API calls for the same video
    if not video_url:
        if video_path.startswith(("gs://", "http://", "https://")):
            video_url = generate_signed_url(video_path)
        else:
            video_url = video_path  # Local path, no signing needed

    # ========================================================================
    # STAGE 1: Audio Breakdown + Analysis
    # ========================================================================
    logger.info("Step 1/5: Audio Extraction and Analysis...")
    audio_analysis_result = await process_audio_analysis(
        video_path=video_path,
        project_id=project_id,
        video_url=video_url,
    )

    # Extract results
    audio_result = audio_analysis_result["audio_result"]
    stream_a_results = audio_analysis_result["stream_a_results"]
    stream_b_results = audio_analysis_result["stream_b_results"]

    # Update stats with audio analysis timing
    audio_stats = audio_analysis_result["stats"]
    stats["audio_extraction_time"] = audio_stats["audio_extraction_time"]
    stats["transcription_time"] = audio_stats["transcription_time"]
    stats["audio_breakdown_time"] = audio_stats["audio_breakdown_time"]
    stats["audio_analysis_time"] = audio_stats["audio_analysis_time"]

    # Upload audio files to GCS (extract_audio_from_video returns local paths)
    speech_path = audio_result.get("speech")
    if speech_path:
        upload_audio_file_to_gcs(
            file_path=speech_path,
            project_id=project_id,
            directory="signals/audio",
            filename="audio_speech.wav",
            bucket_name=GCS_ASSETS_BUCKET,
            cleanup_local=True,
        )

        full_audio_path = audio_result.get("full_audio")
        if full_audio_path:
            upload_audio_file_to_gcs(
                file_path=full_audio_path,
                project_id=project_id,
                directory="signals/audio",
                filename="audio_full.wav",
                bucket_name=GCS_ASSETS_BUCKET,
                cleanup_local=True,
            )

    # Upload Stream A and Stream B results as JSON
    if stream_a_results:
        upload_json_to_gcs(
            data=stream_a_results,
            project_id=project_id,
            directory="signals/audio",
            filename="stream_a_results.json",
            bucket_name=GCS_ASSETS_BUCKET,
        )
        logger.info("Uploaded Stream A results to GCS")

    if stream_b_results:
        upload_json_to_gcs(
            data=stream_b_results,
            project_id=project_id,
            directory="signals/audio",
            filename="stream_b_results.json",
            bucket_name=GCS_ASSETS_BUCKET,
        )
        logger.info("Uploaded Stream B results to GCS")

    # ========================================================================
    # STAGE 2: Initial Sparse Frame Sampling + Visual Change Analysis
    # ========================================================================
    logger.info("Step 2/5: Initial sparse sampling + visual change analysis...")

    video_duration = audio_result.get("total_duration", 0.0)
    vision_result = await process_initial_vision_analysis(
        video_path=video_path,
        project_id=project_id,
        video_duration=video_duration,
        video_url=video_url,
    )

    visual_frames = vision_result["visual_frames"]

    # Update stats from vision analysis
    stats["initial_sampling_time"] = vision_result["stats"]["initial_sampling_time"]
    stats["visual_analysis_time"] = vision_result["stats"]["visual_analysis_time"]

    # Upload visual frames JSON to GCS
    if visual_frames:
        upload_json_to_gcs(
            data=visual_frames,
            project_id=project_id,
            directory="signals/vision",
            filename="initial_sample_visual_frames.json",
            bucket_name=GCS_ASSETS_BUCKET,
        )
        logger.info("Uploaded visual frames to GCS")

    # ========================================================================
    # STAGE 3: Identify Important Moments + Dense Frame Extraction
    # ========================================================================
    logger.info("Step 3/5: Importance calculation + dense frame extraction...")
    step_start = time.time()

    video_duration = audio_result.get("total_duration", 0.0)

    # identify_important_moments already extracts frames AND computes vision features
    frames_with_features = await identify_important_moments(
        video_duration=video_duration,
        stream_a_results=stream_a_results,
        stream_b_results=stream_b_results,
        visual_frames=visual_frames,
        project_id=project_id,
        video_path=video_path,
        video_url=video_url,
        niche=niche,
    )

    stage3_total_time = time.time() - step_start
    stats["importance_calculation_time"] = stage3_total_time

    logger.info(
        "Step 3 complete: Extracted and analyzed %d frames in %.2fs",
        len(frames_with_features),
        stage3_total_time,
    )

    # Upload important moments analysis to GCS (frames with all features)
    if frames_with_features:
        upload_json_to_gcs(
            data=frames_with_features,
            project_id=project_id,
            directory="analysis",
            filename="important_moments_analysis.json",
            bucket_name=GCS_ASSETS_BUCKET,
        )
        logger.info(
            "Uploaded important moments analysis to GCS: %d frames", len(frames_with_features)
        )

    # ========================================================================
    # STAGE 4: Select Top Segments and Frames
    # ========================================================================
    video_fps = DEFAULT_VIDEO_FPS

    try:
        video_fps = await get_video_fps(video_path, video_url)
    except Exception as e:
        logger.warning("Failed to detect video FPS for segment analysis: %s", e)

    importance_segments_for_selection = calculate_importance_segments_for_analysis(
        stream_a_results=stream_a_results or [],
        stream_b_results=stream_b_results or [],
        visual_frames=visual_frames or [],
        video_duration=video_duration,
        video_fps=video_fps,
    )

    # Upload importance segments to GCS
    if importance_segments_for_selection:
        upload_json_to_gcs(
            data=importance_segments_for_selection,
            project_id=project_id,
            directory="analysis",
            filename="importance_segments.json",
            bucket_name=GCS_ASSETS_BUCKET,
        )
        logger.info(
            "Uploaded importance segments to GCS: %d segments",
            len(importance_segments_for_selection),
        )

    segment_analysis = select_top_segments_and_frames(
        frames_with_features=frames_with_features,
        importance_segments=importance_segments_for_selection,
        min_segments=5,
        max_segments=8,
        min_frames_per_segment=1,
        max_frames_per_segment=2,
    )

    # Upload segment analysis to GCS
    upload_json_to_gcs(
        data=segment_analysis,
        project_id=project_id,
        directory="analysis",
        filename="segment_analysis.json",
        bucket_name=GCS_ASSETS_BUCKET,
    )
    logger.info("Uploaded segment analysis to GCS")

    # Verify frames data
    logger.info("Extracted and analyzed %d frames", len(frames_with_features))

    stats["total_time"] = time.time() - start_time

    # ========================================================================
    # PERFORMANCE BREAKDOWN REPORT
    # ========================================================================
    logger.info("=" * 70)
    logger.info("⏱️  PERFORMANCE BREAKDOWN")
    logger.info("=" * 70)
    logger.info(
        "1. Audio Extraction:        %6.2fs  (%5.1f%%)",
        stats.get("audio_extraction_time", 0),
        (stats.get("audio_extraction_time", 0) / stats["total_time"] * 100),
    )
    logger.info(
        "2. Transcription:           %6.2fs  (%5.1f%%)",
        stats.get("transcription_time", 0),
        (stats.get("transcription_time", 0) / stats["total_time"] * 100),
    )
    logger.info(
        "3. Audio Analysis (A+B):    %6.2fs  (%5.1f%%)",
        stats.get("audio_analysis_time", 0),
        (stats.get("audio_analysis_time", 0) / stats["total_time"] * 100),
    )
    logger.info(
        "4. Initial Sparse Sampling: %6.2fs  (%5.1f%%)",
        stats.get("initial_sampling_time", 0),
        (stats.get("initial_sampling_time", 0) / stats["total_time"] * 100),
    )
    logger.info(
        "5. Visual Change Analysis:  %6.2fs  (%5.1f%%)",
        stats.get("visual_analysis_time", 0),
        (stats.get("visual_analysis_time", 0) / stats["total_time"] * 100),
    )
    logger.info("6. Dense Frame Extraction + Vision Analysis:")
    logger.info(
        "   └─ Total:                %6.2fs  (%5.1f%%)",
        stats.get("importance_calculation_time", 0),
        (stats.get("importance_calculation_time", 0) / stats["total_time"] * 100),
    )
    logger.info("-" * 70)
    logger.info("TOTAL PROCESSING TIME:     %6.2fs", stats["total_time"])
    logger.info("=" * 70)

    # Check if frames were extracted
    if not frames_with_features:
        raise ValueError(
            f"No frames extracted! "
            f"Sample frames analyzed: {len(visual_frames)}. "
            f"This could indicate an issue with ffmpeg or video access."
        )

    # Filter out frames without GCS URL (failed upload)
    frames_with_gcs = [f for f in frames_with_features if f.get("gcs_url")]

    if len(frames_with_gcs) < len(frames_with_features):
        skipped = len(frames_with_features) - len(frames_with_gcs)
        logger.warning("Skipped %d frames without GCS URL", skipped)

    logger.info("Returning %d frames with complete vision features", len(frames_with_gcs))

    # ========================================================================
    # CLEANUP: Deferred to after thumbnail selection
    # ========================================================================
    # Cleanup sample frames directory (sparse frames used for analysis - no longer needed)
    # BUT KEEP dense_frames directory - needed for thumbnail selection

    try:
        sample_frames_dir = (
            Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / "sample_frames"
        ).resolve()
        if sample_frames_dir.exists():
            shutil.rmtree(sample_frames_dir)
            logger.debug("Cleaned up sample frames directory: %s", sample_frames_dir)
    except Exception as e:
        logger.warning("Failed to cleanup sample frames directory: %s", e)

    # KEEP dense_frames directory - needed for thumbnail selection
    # Will be cleaned up in main.py after thumbnail selection via cleanup_local_frames()
    dense_frames_dir = (
        Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / "dense_frames"
    ).resolve()
    if dense_frames_dir.exists():
        logger.debug("Keeping dense frames local for thumbnail selection: %s", dense_frames_dir)

    return {
        "project_id": project_id,
        "frames": [f["gcs_url"] for f in frames_with_gcs],
        "extracted_frames": frames_with_gcs,  # Return ALL vision features, not just filtered metadata
        "segment_analysis": segment_analysis,
        "processing_stats": stats,
    }


def cleanup_local_frames(project_id: str) -> None:
    """
    Clean up local extracted frames after thumbnail selection.

    Call this after thumbnail selection completes to remove local files.
    Frames are already backed up in GCS.

    Args:
        project_id: Project identifier
    """
    try:
        # Clean up dense_frames directory (extracted frames for thumbnail selection)
        dense_frames_dir = (
            Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / "dense_frames"
        ).resolve()
        if dense_frames_dir.exists():
            shutil.rmtree(dense_frames_dir)
            logger.debug("Deleted dense frames directory: %s", dense_frames_dir)

        # Clean up derived-media directory (if it exists)
        derived_dir = (Path(LOCAL_MEDIA_DIR) / DERIVED_MEDIA_DIR / project_id).resolve()
        if derived_dir.exists():
            shutil.rmtree(derived_dir)
            logger.debug("Deleted derived-media directory: %s", derived_dir)

        # Clean up entire temp directory if it's now empty
        temp_dir = (Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id).resolve()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.debug("Removed empty temp directory: %s", temp_dir)
    except Exception as e:
        logger.warning("Failed to cleanup local frames: %s", e)
