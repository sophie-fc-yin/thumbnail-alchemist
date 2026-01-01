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

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.analysis.moment_importance import (
    calculate_audio_energy_delta,
    calculate_moment_importance,
    calculate_speech_emotion_delta,
    importance_to_sampling_interval,
    segment_video_by_importance,
)
from app.analysis.output import (
    build_comprehensive_analysis_json,
    merge_stream_timelines,
    save_analysis_json_to_gcs,
)
from app.audio.extraction import (
    analyze_audio_features,
    extract_audio_from_video,
    transcribe_and_analyze_audio,
)
from app.audio.saliency import detect_audio_saliency
from app.audio.speech_semantics import analyze_speech_semantics
from app.models import SourceMedia
from app.vision.extraction import generate_signed_url
from app.vision.face_analysis import FaceExpressionAnalyzer, calculate_landmark_motion

# ============================================================================
# CONSTANTS
# ============================================================================

# Pipeline Configuration
DEFAULT_MAX_FRAMES = 100
DEFAULT_MAX_DURATION_SECONDS = 1800  # 30 minutes
INITIAL_SAMPLE_RATIO = 5  # Sample 1/5 of max frames initially (min 20)
IMPORTANCE_SEGMENTATION_THRESHOLD = 0.2  # Importance change threshold for segmentation

# Dynamic max_frames calculation
FRAMES_PER_MINUTE = 10  # Target density: 1 frame every 6 seconds
MAX_VIDEO_LENGTH_MINUTES = 15  # Cap max_frames at 15 minutes worth
CALCULATED_MAX_FRAMES_CAP = FRAMES_PER_MINUTE * MAX_VIDEO_LENGTH_MINUTES  # 150 frames
MIN_FRAMES_FOR_ANALYSIS = 20  # Minimum frames for proper analysis

# GCS Buckets
GCS_TEMP_BUCKET = "clickmoment-prod-temp"
GCS_ASSETS_BUCKET = "clickmoment-prod-assets"

# Frame Extraction
DEFAULT_FRAME_INTERVAL_FALLBACK = 2.0  # seconds
FFMPEG_OUTPUT_PATTERN = "sample_%03d.jpg"
SEGMENT_FRAME_PATTERN = "frame_%03d.jpg"

# Local Storage Paths
LOCAL_MEDIA_DIR = "thumbnail-alchemist-media"
TEMP_DIR_NAME = "temp"
SAMPLE_FRAMES_DIR = "sample_frames"
DOWNLOADED_SAMPLES_DIR = "downloaded_samples"
DERIVED_MEDIA_DIR = "derived-media"
FRAMES_DIR = "frames"

# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

# Reuse FaceExpressionAnalyzer across all requests to avoid re-loading models
_face_analyzer_instance = None


def get_face_analyzer() -> FaceExpressionAnalyzer:
    """Get or create singleton FaceExpressionAnalyzer instance."""
    global _face_analyzer_instance
    if _face_analyzer_instance is None:
        _face_analyzer_instance = FaceExpressionAnalyzer()
    return _face_analyzer_instance


def calculate_max_frames_for_duration(duration_seconds: float) -> int:
    """
    Calculate optimal max_frames based on video duration.

    Uses a target density of 1 frame per 6 seconds (10 frames/minute),
    capped at 15 minutes worth (150 frames) to avoid excessive processing
    on very long videos.

    Args:
        duration_seconds: Video duration in seconds

    Returns:
        Calculated max_frames (between MIN_FRAMES_FOR_ANALYSIS and CALCULATED_MAX_FRAMES_CAP)
    """
    duration_minutes = duration_seconds / 60.0
    calculated_frames = int(duration_minutes * FRAMES_PER_MINUTE)

    # Apply min and max bounds
    return max(MIN_FRAMES_FOR_ANALYSIS, min(calculated_frames, CALCULATED_MAX_FRAMES_CAP))


async def orchestrate_adaptive_sampling(
    video_path: str,
    project_id: str,
    max_frames: int | None = None,
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
        max_frames: Maximum total frames to extract

    Returns:
        Dictionary with:
            - frames: List of GCS URLs for extracted frames
            - importance_segments: List of importance segments with metadata
            - audio_timeline: Full audio timeline
            - processing_stats: Timing and performance metrics
            - summary: Human-readable summary
    """
    start_time = time.time()
    stats = {
        "audio_time": 0.0,
        "initial_sampling_time": 0.0,
        "face_analysis_time": 0.0,
        "importance_calculation_time": 0.0,
        "adaptive_extraction_time": 0.0,
        "total_time": 0.0,
    }

    print(f"[Orchestrator] Starting adaptive sampling for {video_path}")

    # ========================================================================
    # STEP 1: Audio Breakdown
    # ========================================================================
    print("[Orchestrator] Step 1/5: Audio breakdown...")
    step_start = time.time()

    source_media = SourceMedia(video_path=video_path)
    audio_result = await extract_audio_from_video(
        content_sources=source_media,
        project_id=project_id,
        max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS,
    )

    if not audio_result:
        raise ValueError("Failed to extract audio from video")

    # Use speech lane for transcription (better accuracy)
    speech_path = audio_result["speech"]
    speech_ratio = audio_result.get("speech_ratio", 0.0)

    print(
        f"[Orchestrator] Audio extraction complete: "
        f"{speech_ratio:.1%} speech, "
        f"{len(audio_result.get('segments', []))} speech segments"
    )

    audio_analysis = await transcribe_and_analyze_audio(
        audio_path=speech_path,  # Use speech-only lane for transcription
        project_id=project_id,
        language="en",
        save_timeline=True,
    )

    audio_timeline = audio_analysis["timeline"]
    stats["audio_time"] = time.time() - step_start
    print(f"[Orchestrator] Audio breakdown complete: {len(audio_timeline)} events")

    # ========================================================================
    # STREAM A + B: Advanced Audio Analysis (BEFORE sampling!)
    # ========================================================================
    print("[Orchestrator] Running Stream A (Speech Semantics) + Stream B (Audio Saliency)...")
    stream_start = time.time()

    # Stream A: Speech Semantics Analysis
    stream_a_results = None
    try:
        if audio_analysis.get("transcript"):
            stream_a_results = await analyze_speech_semantics(
                audio_path=speech_path,
                transcript=audio_analysis["transcript"],
                transcript_segments=audio_analysis.get("segments", []),
                speech_segments=audio_result.get("segments", []),
                prosody_features={},  # Will be extracted internally
            )
            print(
                f"[Stream A] Complete: {len(stream_a_results.get('importance_timeline', []))} moments"
            )
    except Exception as e:
        print(f"[Stream A] Failed: {e}")

    # Stream B: Audio Saliency Detection
    stream_b_results = None
    try:
        full_audio_path = audio_result.get("full_audio")
        if full_audio_path:
            audio_features = await analyze_audio_features(full_audio_path)  # FIX: await
            stream_b_results = detect_audio_saliency(
                audio_features=audio_features,
                speech_segments=audio_result.get("segments"),
            )
            print(
                f"[Stream B] Complete: {len(stream_b_results.get('saliency_timeline', []))} moments"
            )
    except Exception as e:
        print(f"[Stream B] Failed: {e}")

    stats["stream_analysis_time"] = time.time() - stream_start

    # Calculate max_frames dynamically if not provided
    video_duration = audio_analysis.get("duration_seconds", 0.0)
    if max_frames is None:
        max_frames = calculate_max_frames_for_duration(video_duration)
        print(
            f"[Orchestrator] Calculated max_frames={max_frames} "
            f"for {video_duration/60:.1f}min video "
            f"(~{FRAMES_PER_MINUTE} frames/min, capped at {MAX_VIDEO_LENGTH_MINUTES}min)"
        )
    else:
        print(f"[Orchestrator] Using provided max_frames={max_frames}")

    # ========================================================================
    # STEP 2: Initial Sparse Frame Sampling
    # ========================================================================
    print("[Orchestrator] Step 2/5: Initial sparse sampling...")
    step_start = time.time()

    # Sample frames for importance analysis based on video duration
    # Keep local for analysis, upload to GCS for reference
    # Target: 1 frame every 3 seconds for good moment importance detection
    SAMPLE_INTERVAL_SECONDS = 3.0
    duration_based_count = int(video_duration / SAMPLE_INTERVAL_SECONDS)

    # Ensure reasonable bounds: 20-100 frames
    initial_sample_count = max(20, min(100, duration_based_count))
    local_sample_frames = await _extract_sample_frames_to_temp(
        content_sources=source_media,
        project_id=project_id,
        max_frames=initial_sample_count,
    )

    if not local_sample_frames:
        raise ValueError("Failed to extract sample frames")

    stats["initial_sampling_time"] = time.time() - step_start

    # ========================================================================
    # STEP 3: Face Analysis on Samples
    # ========================================================================
    print("[Orchestrator] Step 3/5: Face analysis...")
    step_start = time.time()

    # Get singleton analyzer instance (models loaded once and reused)
    analyzer = get_face_analyzer()
    face_analyses = []
    expression_deltas = []
    landmark_motions = []

    prev_landmarks = None
    prev_expression = 0.0

    for frame_path in local_sample_frames:
        analysis = analyzer.analyze_frame(frame_path)
        face_analyses.append(analysis)

        if analysis["has_face"]:
            # Calculate expression delta
            expr_delta = abs(analysis["expression_intensity"] - prev_expression)
            expression_deltas.append(expr_delta)
            prev_expression = analysis["expression_intensity"]

            # Calculate landmark motion
            if prev_landmarks:
                motion = calculate_landmark_motion(prev_landmarks, analysis["landmarks"])
                landmark_motions.append(motion)
            else:
                landmark_motions.append(0.0)

            prev_landmarks = analysis["landmarks"]
        else:
            expression_deltas.append(0.0)
            landmark_motions.append(0.0)

    stats["face_analysis_time"] = time.time() - step_start
    print(
        f"[Orchestrator] Face analysis complete: "
        f"{sum(1 for a in face_analyses if a['has_face'])}/{len(face_analyses)} faces detected"
    )

    # ========================================================================
    # STEP 4: Calculate Moment Importance Scores
    # ========================================================================
    print("[Orchestrator] Step 4/5: Importance calculation...")
    step_start = time.time()

    # Get audio signals
    audio_energy_deltas = calculate_audio_energy_delta(audio_timeline)
    speech_emotion_deltas = calculate_speech_emotion_delta(audio_timeline)

    # Calculate importance scores for each sample point
    # We need to align face analysis with audio timeline AND Stream A/B signals
    video_duration = audio_analysis["duration_seconds"]

    # Extract Stream A/B timelines
    stream_a_timeline = stream_a_results.get("importance_timeline", []) if stream_a_results else []
    stream_b_timeline = stream_b_results.get("saliency_timeline", []) if stream_b_results else []

    importance_scores = []
    timestamps = []

    for i in range(len(local_sample_frames)):
        # Calculate timestamp for this frame (evenly distributed)
        timestamp = (i / max(1, len(local_sample_frames) - 1)) * video_duration
        timestamps.append(timestamp)

        # Get closest audio signals
        audio_idx = (
            min(
                int(timestamp / video_duration * len(audio_energy_deltas)),
                len(audio_energy_deltas) - 1,
            )
            if audio_energy_deltas
            else 0
        )

        speech_idx = (
            min(
                int(timestamp / video_duration * len(speech_emotion_deltas)),
                len(speech_emotion_deltas) - 1,
            )
            if speech_emotion_deltas
            else 0
        )

        audio_delta = audio_energy_deltas[audio_idx] if audio_energy_deltas else 0.0
        speech_delta = speech_emotion_deltas[speech_idx] if speech_emotion_deltas else 0.0

        # Get audio_score from timeline (find closest segment)
        timestamp_ms = timestamp * 1000
        audio_score_value = 0.0
        min_distance = float("inf")

        for event in audio_timeline:
            if event.get("type") == "segment" and "audio_score" in event:
                event_center_ms = (event["start_ms"] + event["end_ms"]) / 2.0
                distance = abs(timestamp_ms - event_center_ms)

                if distance < min_distance:
                    min_distance = distance
                    audio_score_value = event["audio_score"]

        # Get Stream A boost (narrative moments nearby)
        stream_a_boost = 0.0
        for moment in stream_a_timeline:
            time_diff = abs(moment.get("time", 0) - timestamp)
            if time_diff < 2.0:  # Within 2 seconds
                # Closer moments have stronger influence
                proximity = 1.0 - (time_diff / 2.0)
                stream_a_boost = max(stream_a_boost, moment.get("score", 0) * proximity)

        # Get Stream B boost (audio saliency nearby)
        stream_b_boost = 0.0
        for moment in stream_b_timeline:
            time_diff = abs(moment.get("time", 0) - timestamp)
            if time_diff < 1.5:  # Within 1.5 seconds
                proximity = 1.0 - (time_diff / 1.5)
                stream_b_boost = max(stream_b_boost, moment.get("score", 0) * proximity)

        # Calculate importance score with Stream A/B boosts
        base_importance = calculate_moment_importance(
            expression_delta=expression_deltas[i],
            landmark_motion=landmark_motions[i],
            audio_energy_delta=audio_delta,
            speech_emotion_delta=speech_delta,
            audio_score=audio_score_value,
        )

        # Boost importance near important narrative/audio moments
        importance = base_importance * (1.0 + stream_a_boost * 0.3 + stream_b_boost * 0.2)
        importance_scores.append(importance)

    # Segment video by importance
    importance_segments = segment_video_by_importance(
        importance_scores=importance_scores,
        timestamps=timestamps,
        threshold=IMPORTANCE_SEGMENTATION_THRESHOLD,
        video_duration=video_duration,
    )

    stats["importance_calculation_time"] = time.time() - step_start
    print(
        f"[Orchestrator] Importance calculation complete: {len(importance_segments)} segments identified"
    )

    # Log importance segments
    for i, seg in enumerate(importance_segments):
        print(
            f"  Segment {i+1}: {seg['start_time']:.1f}s-{seg['end_time']:.1f}s "
            f"| Importance: {seg['importance_level']} ({seg['avg_importance']:.2f})"
        )

    # ========================================================================
    # STEP 5: Adaptive Frame Extraction Based on Importance
    # ========================================================================
    print("[Orchestrator] Step 5/5: Adaptive extraction + face analysis...")
    step_start = time.time()

    # Calculate target frame count per segment based on importance
    all_frames = []  # List of dicts with 'local_path', 'gcs_url', and 'face_analysis'
    frame_segment_map = {}  # Map GCS URL → segment info

    # Get face analyzer for ALL extracted frames (not just samples!)
    analyzer = get_face_analyzer()

    for segment_idx, segment in enumerate(importance_segments):
        print(
            f"[Orchestrator] Processing segment {segment_idx+1}/{len(importance_segments)}: {segment['start_time']:.1f}s-{segment['end_time']:.1f}s ({segment['importance_level']})"
        )
        try:
            # Calculate sampling interval for this segment
            interval = importance_to_sampling_interval(segment["avg_importance"])

            # Calculate number of frames for this segment
            segment_duration = segment["end_time"] - segment["start_time"]
            frame_count = max(1, int(segment_duration / interval))
            print(f"[Orchestrator]   Target: {frame_count} frames (interval: {interval:.2f}s)")

            # Extract frames for this segment using ffmpeg (returns dicts with local_path + gcs_url)
            segment_frames = await _extract_frames_for_segment(
                video_path=video_path,
                project_id=project_id,
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                frame_count=frame_count,
            )
            print(f"[Orchestrator]   Extracted {len(segment_frames)} frames from segment")
            # Run face analysis on EVERY extracted frame
            faces_detected = 0
            for idx, frame_dict in enumerate(segment_frames):
                local_path = frame_dict.get("local_path")
                gcs_url = frame_dict.get("gcs_url", "")

                # Analyze this frame - try local path first, fallback to GCS download if needed
                frame_path_for_analysis = None
                if local_path and Path(local_path).exists():
                    frame_path_for_analysis = str(local_path)
                elif gcs_url:
                    # Fallback: download from GCS if local file is missing (Cloud Run ephemeral filesystem)
                    try:
                        from google.cloud import storage  # type: ignore

                        # Download to temp location
                        temp_frame_path = (
                            Path(LOCAL_MEDIA_DIR)
                            / TEMP_DIR_NAME
                            / project_id
                            / f"frame_{idx}_temp.jpg"
                        )
                        temp_frame_path.parent.mkdir(parents=True, exist_ok=True)

                        # Parse GCS URL and download
                        if gcs_url.startswith("gs://"):
                            bucket_name = gcs_url.split("/")[2]
                            blob_path = "/".join(gcs_url.split("/")[3:])
                            client = storage.Client()
                            bucket = client.bucket(bucket_name)
                            blob = bucket.blob(blob_path)
                            blob.download_to_filename(str(temp_frame_path))
                            frame_path_for_analysis = str(temp_frame_path)
                            print(
                                f"    [DEBUG] Frame {idx+1}: Downloaded from GCS (local file missing)"
                            )
                    except Exception as e:
                        print(f"    [DEBUG] Frame {idx+1}: Failed to download from GCS: {e}")

                if frame_path_for_analysis:
                    try:
                        face_analysis = analyzer.analyze_frame(frame_path_for_analysis)
                        frame_dict["face_analysis"] = face_analysis

                        # DEBUG: Log first 3 frames to see what's being detected
                        if idx < 3:
                            print(f"    [DEBUG] Frame {idx+1} analysis:")
                            print(f"      has_face: {face_analysis.get('has_face', False)}")
                            print(
                                f"      emotion: {face_analysis.get('dominant_emotion', 'unknown')}"
                            )
                            print(
                                f"      expression_intensity: {face_analysis.get('expression_intensity', 0.0):.2f}"
                            )

                        if face_analysis.get("has_face", False):
                            faces_detected += 1
                    except Exception as e:
                        print(f"    [DEBUG] Frame {idx+1}: Face analysis failed: {e}")
                        frame_dict["face_analysis"] = {
                            "has_face": False,
                            "dominant_emotion": "unknown",
                            "expression_intensity": 0.0,
                            "eye_openness": 0.0,
                            "mouth_openness": 0.0,
                            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        }
                else:
                    # No face analysis if file missing and GCS download failed
                    frame_dict["face_analysis"] = {
                        "has_face": False,
                        "dominant_emotion": "unknown",
                        "expression_intensity": 0.0,
                        "eye_openness": 0.0,
                        "mouth_openness": 0.0,
                        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    }
                    if idx < 3:
                        print(f"    [DEBUG] Frame {idx+1}: File missing - {local_path}")

                # Track which segment each frame belongs to (using GCS URL as key)
                # CRITICAL: Map ALL frames with GCS URLs to their segment info for scoring
                if gcs_url:  # Only map frames that successfully uploaded
                    frame_segment_map[gcs_url] = {
                        "importance_level": segment["importance_level"],
                        "importance_score": segment["avg_importance"],
                        "segment_start": segment["start_time"],
                        "segment_end": segment["end_time"],
                    }
                else:
                    # Log warning if frame doesn't have GCS URL (shouldn't happen after upload)
                    print(
                        f"    [WARNING] Frame {idx+1} in segment has no GCS URL - will be skipped later"
                    )

            # Add all frames from this segment to the total collection
            all_frames.extend(segment_frames)
            print(
                f"[Orchestrator]   Added {len(segment_frames)} frames to collection (total so far: {len(all_frames)})"
            )

            print(
                f"  Segment {segment['importance_level']}: "
                f"extracted {len(segment_frames)} frames "
                f"(interval: {interval:.2f}s, {faces_detected} faces)"
            )
        except Exception as e:
            print(
                f"[Orchestrator] ⚠️  Error processing segment {segment['start_time']:.1f}s-{segment['end_time']:.1f}s: {e}"
            )
            print("[Orchestrator] Continuing with next segment...")
            import traceback

            traceback.print_exc()
            # Continue with next segment - don't break the loop

    stats["adaptive_extraction_time"] = time.time() - step_start
    stats["total_time"] = time.time() - start_time

    print(f"[Orchestrator] Adaptive extraction complete: {len(all_frames)} total frames")
    print(f"[Orchestrator] Frames mapped to segments: {len(frame_segment_map)}")
    print(f"[Orchestrator] Total processing time: {stats['total_time']:.2f}s")

    # Verify all frames have required data for scoring
    frames_with_gcs = sum(1 for f in all_frames if f.get("gcs_url"))
    frames_with_face_analysis = sum(1 for f in all_frames if f.get("face_analysis"))
    print("[Orchestrator] Quality check:")
    print(f"  Frames with GCS URL: {frames_with_gcs}/{len(all_frames)}")
    print(f"  Frames with face_analysis: {frames_with_face_analysis}/{len(all_frames)}")
    print(f"  Frames mapped to segments: {len(frame_segment_map)}/{frames_with_gcs}")

    # Check if frames were extracted
    if not all_frames:
        raise ValueError(
            f"No frames extracted! "
            f"Importance segments: {len(importance_segments)}, "
            f"Sample frames analyzed: {len(local_sample_frames)}. "
            f"This could indicate an issue with ffmpeg or video access."
        )

    # ========================================================================
    # BUILD COMPREHENSIVE ANALYSIS JSON
    # ========================================================================
    print("[Orchestrator] Building comprehensive analysis JSON...")

    # Prepare visual analysis with timestamps
    visual_analysis_with_timestamps = []
    for i, (frame_path, analysis) in enumerate(zip(local_sample_frames, face_analyses)):
        timestamp = (i / max(1, len(local_sample_frames) - 1)) * video_duration
        # Use local path for processing
        visual_analysis_with_timestamps.append(
            {
                "timestamp": timestamp,
                "frame_path": str(frame_path),
                **analysis,
            }
        )

    # Merge timelines
    stream_a_timeline = stream_a_results.get("importance_timeline", []) if stream_a_results else []
    stream_b_timeline = stream_b_results.get("saliency_timeline", []) if stream_b_results else []

    merged_timeline = merge_stream_timelines(
        stream_a_timeline=stream_a_timeline,
        stream_b_timeline=stream_b_timeline,
        visual_frames=visual_analysis_with_timestamps,
        temporal_window=0.5,
    )

    print(f"[Orchestrator] Merged timeline: {len(merged_timeline)} moment candidates")

    # Build extracted frames metadata with face analysis data
    # CRITICAL: Ensure ALL frames get complete analysis (face_analysis, moment_score, importance_score)
    extracted_frames_metadata = []
    frames_without_analysis = 0
    frames_without_scores = 0

    for frame_dict in all_frames:
        gcs_url = frame_dict.get("gcs_url", "")
        local_path = frame_dict.get("local_path", "")
        face_analysis = frame_dict.get("face_analysis", {})

        # Skip frames that failed to upload to GCS (can't be used without GCS URL)
        if not gcs_url:
            print(f"⚠️  Skipping frame (no GCS URL): {local_path}")
            continue

        # Ensure face_analysis exists and has all required fields for scoring
        # Required fields: has_face, dominant_emotion, expression_intensity, eye_openness, mouth_openness, head_pose
        if not face_analysis:
            frames_without_analysis += 1
            print(f"⚠️  Frame missing face_analysis, creating default: {local_path}")
            face_analysis = {
                "has_face": False,
                "dominant_emotion": "unknown",
                "expression_intensity": 0.0,
                "eye_openness": 0.0,
                "mouth_openness": 0.0,
                "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "emotion_probs": {},  # Optional but good to have
                "landmarks": [],  # Optional but good to have
            }
        else:
            # Validate that all required fields exist (fill in missing ones)
            required_fields = {
                "has_face": face_analysis.get("has_face", False),
                "dominant_emotion": face_analysis.get("dominant_emotion", "unknown"),
                "expression_intensity": face_analysis.get("expression_intensity", 0.0),
                "eye_openness": face_analysis.get("eye_openness", 0.0),
                "mouth_openness": face_analysis.get("mouth_openness", 0.0),
                "head_pose": face_analysis.get(
                    "head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
                ),
            }
            # Merge to ensure all fields are present
            face_analysis = {**face_analysis, **required_fields}

        # Extract timestamp from filename (e.g., "frame_5200ms.jpg")
        try:
            timestamp_ms = int(Path(local_path).stem.split("_")[1].replace("ms", ""))
            timestamp_sec = timestamp_ms / 1000.0

            # Find closest moment in merged timeline for moment_score
            closest_moment = (
                min(
                    merged_timeline,
                    key=lambda m: abs(m["time"] - timestamp_sec),
                )
                if merged_timeline
                else None
            )

            # Get segment info (keyed by GCS URL) for importance_score
            segment_info = frame_segment_map.get(gcs_url, {})

            # Calculate moment_score (from merged timeline)
            moment_score = closest_moment["score"] if closest_moment else 0.0

            # Get importance_score (from segment)
            importance_score = segment_info.get("importance_score", 0.0)
            importance_level = segment_info.get("importance_level", "unknown")

            # Ensure scores are valid numbers
            if not isinstance(moment_score, (int, float)):
                moment_score = 0.0
            if not isinstance(importance_score, (int, float)):
                importance_score = 0.0
                frames_without_scores += 1

            extracted_frames_metadata.append(
                {
                    "timestamp": timestamp_sec,
                    "frame_path": gcs_url,  # GCS URL for debugging/storage
                    "local_path": str(local_path),  # Local path for Gemini (faster)
                    "moment_score": float(moment_score),  # Ensure it's a float
                    "importance_score": float(importance_score),  # Ensure it's a float
                    "importance_level": importance_level,
                    "sources": closest_moment["sources"] if closest_moment else [],
                    "types": closest_moment["types"] if closest_moment else [],
                    "face_analysis": face_analysis,  # Include face analysis for this specific frame
                }
            )
        except (ValueError, IndexError, AttributeError) as e:
            # Fallback if timestamp parsing fails - still include frame with default scores
            print(f"⚠️  Failed to parse timestamp for {local_path}: {e}")
            segment_info = frame_segment_map.get(gcs_url, {})
            extracted_frames_metadata.append(
                {
                    "timestamp": 0.0,
                    "frame_path": gcs_url,  # GCS URL for debugging/storage
                    "local_path": str(local_path),  # Local path for Gemini (faster)
                    "moment_score": 0.0,  # Default score if parsing fails
                    "importance_score": float(segment_info.get("importance_score", 0.0)),
                    "importance_level": segment_info.get("importance_level", "unknown"),
                    "sources": [],
                    "types": [],
                    "face_analysis": face_analysis,  # Include face analysis even in fallback
                }
            )

    # Log analysis coverage
    print("[Orchestrator] Analysis coverage:")
    print(f"  Total frames: {len(extracted_frames_metadata)}")
    print(f"  Frames without face_analysis: {frames_without_analysis}")
    print(f"  Frames without importance_score: {frames_without_scores}")
    print(
        f"  Frames with complete analysis: {len(extracted_frames_metadata) - frames_without_analysis - frames_without_scores}"
    )

    # Build comprehensive analysis JSON
    comprehensive_analysis = build_comprehensive_analysis_json(
        project_id=project_id,
        video_path=video_path,
        stream_a_results=stream_a_results,
        stream_b_results=stream_b_results,
        visual_analysis=visual_analysis_with_timestamps,
        merged_timeline=merged_timeline,
        extracted_frames=extracted_frames_metadata,
        importance_segments=importance_segments,
        importance_statistics={
            "avg_importance": float(
                np.mean([seg["avg_importance"] for seg in importance_segments])
            ),
            "segment_counts": {
                "low": sum(1 for s in importance_segments if s["importance_level"] == "low"),
                "medium": sum(1 for s in importance_segments if s["importance_level"] == "medium"),
                "high": sum(1 for s in importance_segments if s["importance_level"] == "high"),
            },
            "total_segments": len(importance_segments),
        },
        processing_stats=stats,
        audio_features=audio_features if stream_b_results else None,
        transcript_data={
            "transcript": audio_analysis.get("transcript", ""),
            "segments": audio_analysis.get("segments", []),
            "duration": audio_analysis.get("duration_seconds", 0.0),
        }
        if audio_analysis
        else None,
    )

    # Save to GCS
    try:
        analysis_json_url = await save_analysis_json_to_gcs(
            analysis_json=comprehensive_analysis,
            project_id=project_id,
            bucket_name=GCS_ASSETS_BUCKET,
        )
        print(f"[Orchestrator] Saved comprehensive analysis to: {analysis_json_url}")
    except Exception as e:
        print(f"[Orchestrator] Failed to save analysis JSON: {e}")
        analysis_json_url = None

    # ========================================================================
    # CLEANUP: Deferred to after thumbnail selection
    # ========================================================================
    # NOTE: Cleanup now happens in main.py AFTER thumbnail selection completes
    # This allows Gemini to read frames directly from local disk (faster than downloading from GCS)

    # Cleanup temp directory (sample frames for analysis - no longer needed)
    try:
        temp_dir = Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"[Orchestrator] Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        print(f"[Orchestrator] Failed to cleanup temp directory: {e}")

    # KEEP derived-media directory for now (Gemini needs these files)
    # Will be cleaned up in main.py after thumbnail selection
    print("[Orchestrator] Keeping extracted frames local for thumbnail selection...")

    # ========================================================================
    # RETURN RESULTS
    # ========================================================================

    # Calculate statistics
    importance_levels = [seg["importance_level"] for seg in importance_segments]
    category_counts = {
        "low": importance_levels.count("low"),
        "medium": importance_levels.count("medium"),
        "high": importance_levels.count("high"),
    }

    avg_importance = float(np.mean([seg["avg_importance"] for seg in importance_segments]))

    # Enhanced summary with Stream A+B info
    stream_info = ""
    if stream_a_results or stream_b_results:
        stream_a_count = (
            len(stream_a_results.get("importance_timeline", [])) if stream_a_results else 0
        )
        stream_b_count = (
            len(stream_b_results.get("saliency_timeline", [])) if stream_b_results else 0
        )
        stream_info = (
            f" Stream A: {stream_a_count} moments, Stream B: {stream_b_count} moments, "
            f"Merged: {len(merged_timeline)} candidates."
        )

    summary = (
        f"Adaptive sampling complete: {len(all_frames)} frames extracted. "
        f"Importance segments: {category_counts['low']} low, "
        f"{category_counts['medium']} medium, "
        f"{category_counts['high']} high. "
        f"Average importance: {avg_importance:.2f}. "
        f"{stream_info}"
        f"Processing time: {stats['total_time']:.1f}s"
    )

    return {
        "project_id": project_id,
        "frames": [f["gcs_url"] for f in all_frames if f.get("gcs_url")],  # Return GCS URLs only
        "frame_segment_map": frame_segment_map,
        "importance_segments": importance_segments,
        "audio_timeline": audio_timeline,
        "processing_stats": stats,
        "importance_statistics": {
            "avg_importance": avg_importance,
            "segment_counts": category_counts,
            "total_segments": len(importance_segments),
        },
        # New: Stream A + B results
        "stream_a_moments": len(stream_a_results.get("importance_timeline", []))
        if stream_a_results
        else 0,
        "stream_b_moments": len(stream_b_results.get("saliency_timeline", []))
        if stream_b_results
        else 0,
        "merged_moment_candidates": len(merged_timeline),
        # New: Comprehensive analysis JSON URL
        "analysis_json_url": analysis_json_url,
        "summary": summary,
    }


async def _download_temp_frames(
    temp_gcs_urls: list[str],
    project_id: str,
) -> list[Path]:
    """
    Download frames from temp bucket for local analysis.

    Args:
        temp_gcs_urls: List of GCS URLs in temp bucket
        project_id: Project ID

    Returns:
        List of local file paths
    """
    from google.cloud import storage  # type: ignore

    client = storage.Client()
    local_frames = []

    # Create local temp directory
    local_dir = Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / DOWNLOADED_SAMPLES_DIR
    local_dir.mkdir(parents=True, exist_ok=True)

    for gcs_url in temp_gcs_urls:
        # Parse GCS URL: gs://bucket/path
        if not gcs_url.startswith("gs://"):
            # Already a local path
            local_frames.append(Path(gcs_url))
            continue

        parts = gcs_url.replace("gs://", "").split("/", 1)
        if len(parts) != 2:
            continue

        bucket_name, blob_path = parts

        # Download blob
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            local_path = local_dir / Path(blob_path).name
            blob.download_to_filename(str(local_path))
            local_frames.append(local_path)

        except Exception as e:
            print(f"Failed to download {gcs_url}: {e}")
            continue

    print(f"[Orchestrator] Downloaded {len(local_frames)} temp frames for analysis")
    return local_frames


async def _extract_sample_frames_to_temp(
    content_sources: SourceMedia,
    project_id: str,
    max_frames: int,
) -> list[Path]:
    """
    Extract sample frames for moment importance analysis.

    Frames are stored locally for processing.

    Args:
        content_sources: Video source
        project_id: Project ID
        max_frames: Number of frames to sample

    Returns:
        List of local frame paths
    """
    video_path = content_sources.video_path

    # Setup ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return []

    # Generate signed URL if needed
    if video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    # Create local output directory
    output_dir = Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id / SAMPLE_FRAMES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames evenly distributed
    output_pattern = output_dir / FFMPEG_OUTPUT_PATTERN

    # Get video duration for adaptive sampling
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        probe_process = await asyncio.create_subprocess_exec(
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
        stdout, _ = await probe_process.communicate()

        try:
            duration = float(stdout.decode().strip())
            interval = duration / max_frames
        except (ValueError, AttributeError):
            interval = DEFAULT_FRAME_INTERVAL_FALLBACK
    else:
        interval = DEFAULT_FRAME_INTERVAL_FALLBACK

    # Extract frames
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-i",
        video_url,
        "-vf",
        f"fps=1/{interval}",
        "-frames:v",
        str(max_frames),
        str(output_pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()

    # Collect frames
    local_frames = sorted(output_dir.glob("sample_*.jpg"))

    print(f"[Orchestrator] Extracted {len(local_frames)} sample frames for importance analysis")
    return local_frames


async def _extract_frames_for_segment(
    video_path: str,
    project_id: str,
    start_time: float,
    end_time: float,
    frame_count: int,
) -> list[dict[str, str]]:
    """
    Extract frames for a specific time segment and upload to GCS.

    Uses ffmpeg to extract frames from start_time to end_time.
    Frames are temporarily stored locally for ML processing, then uploaded to GCS.

    Args:
        video_path: Video source
        project_id: Project ID for organizing outputs
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        frame_count: Number of frames to extract

    Returns:
        List of dicts with 'local_path' and 'gcs_url' for each frame
    """
    # Setup ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return []

    # Generate signed URL if needed
    if video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(video_path)
    else:
        video_url = video_path

    # Create output directory (use absolute path for Cloud Run compatibility)
    output_dir = (
        Path(LOCAL_MEDIA_DIR)
        / DERIVED_MEDIA_DIR
        / project_id
        / FRAMES_DIR
        / f"segment_{int(start_time)}_{int(end_time)}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate interval
    duration = end_time - start_time
    interval = duration / max(1, frame_count)

    # Extract frames using ffmpeg with timestamp seek
    output_pattern = output_dir / SEGMENT_FRAME_PATTERN

    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-ss",
        str(start_time),  # Seek to start
        "-t",
        str(duration),  # Duration
        "-i",
        video_url,
        "-vf",
        f"fps=1/{interval}",  # Sample at calculated interval
        "-frames:v",
        str(frame_count),
        str(output_pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    await process.communicate()

    # Collect extracted frames
    local_frames = sorted(output_dir.glob("frame_*.jpg"))

    # Rename frames with millisecond timestamps and upload to GCS
    frames_with_urls = []
    upload_failures = 0

    # Initialize GCS client
    from google.cloud import storage  # type: ignore

    client = storage.Client()
    bucket = client.bucket(GCS_ASSETS_BUCKET)

    print(f"[Orchestrator] Uploading {len(local_frames)} frames to gs://{GCS_ASSETS_BUCKET}/...")

    for idx, frame_path in enumerate(local_frames):
        timestamp_ms = int((start_time + idx * interval) * 1000)
        new_name = output_dir / f"frame_{timestamp_ms}ms.jpg"
        frame_path.rename(new_name)

        # Upload to GCS
        try:
            blob_path = f"projects/{project_id}/frames/segment_{int(start_time)}_{int(end_time)}/frame_{timestamp_ms}ms.jpg"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(new_name), content_type="image/jpeg")

            gcs_url = f"gs://{GCS_ASSETS_BUCKET}/{blob_path}"
            # Store absolute path as string to ensure it works across different contexts
            frames_with_urls.append(
                {
                    "local_path": str(new_name.resolve()),
                    "gcs_url": gcs_url,
                }
            )
        except Exception as e:
            upload_failures += 1
            print(f"[Orchestrator] ❌ Failed to upload frame {idx+1}/{len(local_frames)}: {e}")
            # Still include with local path as fallback (will be skipped later)
            frames_with_urls.append(
                {
                    "local_path": str(new_name.resolve()),
                    "gcs_url": "",
                }
            )

    successful_uploads = len(frames_with_urls) - upload_failures
    print(f"[Orchestrator] ✅ Uploaded {successful_uploads}/{len(frames_with_urls)} frames to GCS")

    if upload_failures > 0:
        print(
            f"[Orchestrator] ⚠️  {upload_failures} frames failed to upload - these will be skipped"
        )
    return frames_with_urls


def cleanup_local_frames(project_id: str) -> None:
    """
    Clean up local extracted frames after thumbnail selection.

    Call this after thumbnail selection completes to remove local files.
    Frames are already backed up in GCS.

    Args:
        project_id: Project identifier
    """
    try:
        derived_dir = Path(LOCAL_MEDIA_DIR) / DERIVED_MEDIA_DIR / project_id
        if derived_dir.exists():
            shutil.rmtree(derived_dir)
            print(f"[Cleanup] ✅ Deleted local frames: {derived_dir}")
        else:
            print(f"[Cleanup] No local frames to clean: {derived_dir}")
    except Exception as e:
        print(f"[Cleanup] ❌ Failed to cleanup local frames: {e}")
