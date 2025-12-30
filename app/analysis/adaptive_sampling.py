"""Adaptive frame sampling orchestrator.

Coordinates the full pipeline:
1. Audio breakdown → timeline
2. Initial sparse sampling → sample frames
3. Face analysis → expression & motion
4. Pace calculation → pace scores & segments
5. Adaptive extraction → dense where needed
6. Storage → upload frames to GCS

This module orchestrates the order of operations and manages data flow.
"""

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from google.cloud import storage

from app.analysis.output import (
    build_comprehensive_analysis_json,
    merge_stream_timelines,
    save_analysis_json_to_gcs,
)
from app.analysis.pace_analysis import (
    calculate_audio_energy_delta,
    calculate_pace_score,
    calculate_speech_emotion_delta,
    pace_to_sampling_interval,
    segment_video_by_pace,
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
PACE_SEGMENTATION_THRESHOLD = 0.2  # Pace change threshold for segmentation

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


async def orchestrate_adaptive_sampling(
    video_path: str,
    project_id: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    upload_to_gcs: bool = True,
) -> dict[str, Any]:
    """
    Orchestrate adaptive frame sampling pipeline.

    This is the main entry point that coordinates:
    - Audio analysis
    - Face analysis
    - Pace calculation
    - Adaptive frame extraction
    - Storage upload

    Args:
        video_path: GCS URL or local path to video
        project_id: Project identifier for organizing outputs
        max_frames: Maximum total frames to extract
        upload_to_gcs: Whether to upload frames to GCS

    Returns:
        Dictionary with:
            - frames: List of frame paths (GCS URLs or local)
            - pace_segments: List of pace segments with metadata
            - audio_timeline: Full audio timeline
            - processing_stats: Timing and performance metrics
            - summary: Human-readable summary
    """
    start_time = time.time()
    stats = {
        "audio_time": 0.0,
        "initial_sampling_time": 0.0,
        "face_analysis_time": 0.0,
        "pace_calculation_time": 0.0,
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
    # STEP 2: Initial Sparse Frame Sampling
    # ========================================================================
    print("[Orchestrator] Step 2/5: Initial sparse sampling...")
    step_start = time.time()

    # Sample ~20 frames evenly distributed for pace analysis
    # Upload to temp bucket for analysis (not final storage)
    initial_sample_count = min(20, max_frames // INITIAL_SAMPLE_RATIO)
    sample_frames = await _extract_sample_frames_to_temp(
        content_sources=source_media,
        project_id=project_id,
        max_frames=initial_sample_count,
    )

    if not sample_frames:
        raise ValueError("Failed to extract sample frames")

    stats["initial_sampling_time"] = time.time() - step_start
    print(f"[Orchestrator] Extracted {len(sample_frames)} sample frames")

    # ========================================================================
    # STEP 3: Face Analysis on Samples
    # ========================================================================
    print("[Orchestrator] Step 3/5: Face analysis...")
    step_start = time.time()

    analyzer = FaceExpressionAnalyzer()
    face_analyses = []
    expression_deltas = []
    landmark_motions = []

    prev_landmarks = None
    prev_expression = 0.0

    # Download temp frames for local analysis
    local_sample_frames = await _download_temp_frames(sample_frames, project_id)

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
    # STEP 4: Calculate Pace Scores
    # ========================================================================
    print("[Orchestrator] Step 4/5: Pace calculation...")
    step_start = time.time()

    # Get audio signals
    audio_energy_deltas = calculate_audio_energy_delta(audio_timeline)
    speech_emotion_deltas = calculate_speech_emotion_delta(audio_timeline)

    # Calculate pace scores for each sample point
    # We need to align face analysis with audio timeline
    video_duration = audio_analysis["duration_seconds"]

    pace_scores = []
    timestamps = []

    for i in range(len(sample_frames)):
        # Calculate timestamp for this frame (evenly distributed)
        timestamp = (i / max(1, len(sample_frames) - 1)) * video_duration
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

        # Calculate pace score with new audio_score component
        pace = calculate_pace_score(
            expression_delta=expression_deltas[i],
            landmark_motion=landmark_motions[i],
            audio_energy_delta=audio_delta,
            speech_emotion_delta=speech_delta,
            audio_score=audio_score_value,
        )
        pace_scores.append(pace)

    # Segment video by pace
    pace_segments = segment_video_by_pace(
        pace_scores=pace_scores,
        timestamps=timestamps,
        threshold=PACE_SEGMENTATION_THRESHOLD,
    )

    stats["pace_calculation_time"] = time.time() - step_start
    print(f"[Orchestrator] Pace calculation complete: " f"{len(pace_segments)} segments identified")

    # Log pace segments
    for i, seg in enumerate(pace_segments):
        print(
            f"  Segment {i+1}: {seg['start_time']:.1f}s-{seg['end_time']:.1f}s "
            f"| Pace: {seg['pace_category']} ({seg['avg_pace']:.2f})"
        )

    # ========================================================================
    # STEP 5: Adaptive Frame Extraction Based on Pace
    # ========================================================================
    print("[Orchestrator] Step 5/5: Adaptive extraction...")
    step_start = time.time()

    # Calculate target frame count per segment based on pace
    all_frames = []
    frame_segment_map = {}  # Map frame → segment info

    for segment in pace_segments:
        # Calculate sampling interval for this segment
        interval = pace_to_sampling_interval(segment["avg_pace"])

        # Calculate number of frames for this segment
        segment_duration = segment["end_time"] - segment["start_time"]
        frame_count = max(1, int(segment_duration / interval))

        # Extract frames for this segment using ffmpeg
        segment_frames = await _extract_frames_for_segment(
            video_path=video_path,
            project_id=project_id,
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            frame_count=frame_count,
            upload_to_gcs=upload_to_gcs,
        )

        # Track which segment each frame belongs to
        for frame in segment_frames:
            frame_segment_map[str(frame)] = {
                "pace_category": segment["pace_category"],
                "pace_score": segment["avg_pace"],
                "segment_start": segment["start_time"],
                "segment_end": segment["end_time"],
            }

        all_frames.extend(segment_frames)

        print(
            f"  Segment {segment['pace_category']}: "
            f"extracted {len(segment_frames)} frames "
            f"(interval: {interval:.2f}s)"
        )

    stats["adaptive_extraction_time"] = time.time() - step_start
    stats["total_time"] = time.time() - start_time

    print(f"[Orchestrator] Adaptive extraction complete: {len(all_frames)} total frames")
    print(f"[Orchestrator] Total processing time: {stats['total_time']:.2f}s")

    # ========================================================================
    # STREAM A + B: Advanced Audio Analysis
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
            audio_features = analyze_audio_features(full_audio_path)
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

    # ========================================================================
    # BUILD COMPREHENSIVE ANALYSIS JSON
    # ========================================================================
    print("[Orchestrator] Building comprehensive analysis JSON...")

    # Prepare visual analysis with timestamps
    visual_analysis_with_timestamps = []
    for i, (frame_path, analysis) in enumerate(zip(local_sample_frames, face_analyses)):
        timestamp = (i / max(1, len(sample_frames) - 1)) * video_duration
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

    # Build extracted frames metadata
    extracted_frames_metadata = []
    for frame_path_str in all_frames:
        # Extract timestamp from filename (e.g., "frame_5200ms.jpg")
        try:
            timestamp_ms = int(Path(frame_path_str).stem.split("_")[1].replace("ms", ""))
            timestamp_sec = timestamp_ms / 1000.0

            # Find closest moment in merged timeline
            closest_moment = (
                min(
                    merged_timeline,
                    key=lambda m: abs(m["time"] - timestamp_sec),
                )
                if merged_timeline
                else None
            )

            # Get segment info
            segment_info = frame_segment_map.get(frame_path_str, {})

            extracted_frames_metadata.append(
                {
                    "timestamp": timestamp_sec,
                    "frame_path": frame_path_str,
                    "moment_score": closest_moment["score"] if closest_moment else 0.0,
                    "pace_score": segment_info.get("pace_score", 0.0),
                    "pace_category": segment_info.get("pace_category", "unknown"),
                    "sources": closest_moment["sources"] if closest_moment else [],
                    "types": closest_moment["types"] if closest_moment else [],
                }
            )
        except (ValueError, IndexError, AttributeError):
            # Fallback if timestamp parsing fails
            extracted_frames_metadata.append(
                {
                    "timestamp": 0.0,
                    "frame_path": frame_path_str,
                    "moment_score": 0.0,
                    "pace_score": frame_segment_map.get(frame_path_str, {}).get("pace_score", 0.0),
                    "pace_category": frame_segment_map.get(frame_path_str, {}).get(
                        "pace_category", "unknown"
                    ),
                    "sources": [],
                    "types": [],
                }
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
        pace_segments=pace_segments,
        pace_statistics={
            "avg_pace": float(np.mean([seg["avg_pace"] for seg in pace_segments])),
            "segment_counts": {
                "low": sum(1 for s in pace_segments if s["pace_category"] == "low"),
                "medium": sum(1 for s in pace_segments if s["pace_category"] == "medium"),
                "high": sum(1 for s in pace_segments if s["pace_category"] == "high"),
            },
            "total_segments": len(pace_segments),
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
    # CLEANUP: Delete local temp files
    # ========================================================================
    print("[Orchestrator] Cleaning up local temp files...")
    try:
        temp_dir = Path(LOCAL_MEDIA_DIR) / TEMP_DIR_NAME / project_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"[Orchestrator] Cleaned up {temp_dir}")
    except Exception as e:
        print(f"[Orchestrator] Failed to cleanup temp files: {e}")

    # ========================================================================
    # RETURN RESULTS
    # ========================================================================

    # Calculate statistics
    pace_categories = [seg["pace_category"] for seg in pace_segments]
    category_counts = {
        "low": pace_categories.count("low"),
        "medium": pace_categories.count("medium"),
        "high": pace_categories.count("high"),
    }

    avg_pace = float(np.mean([seg["avg_pace"] for seg in pace_segments]))

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
        f"Pace segments: {category_counts['low']} low, "
        f"{category_counts['medium']} medium, "
        f"{category_counts['high']} high. "
        f"Average pace: {avg_pace:.2f}. "
        f"{stream_info}"
        f"Processing time: {stats['total_time']:.1f}s"
    )

    return {
        "project_id": project_id,
        "frames": [str(f) for f in all_frames],
        "frame_segment_map": frame_segment_map,
        "pace_segments": pace_segments,
        "audio_timeline": audio_timeline,
        "processing_stats": stats,
        "pace_statistics": {
            "avg_pace": avg_pace,
            "segment_counts": category_counts,
            "total_segments": len(pace_segments),
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
) -> list[str]:
    """
    Extract sample frames for pace analysis to temporary bucket.

    Uploads to clickmoment-prod-temp bucket for analysis,
    separate from final frame storage.

    Args:
        content_sources: Video source
        project_id: Project ID
        max_frames: Number of frames to sample

    Returns:
        List of GCS URLs in temp bucket
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

    # Upload to TEMP bucket
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_TEMP_BUCKET)

        temp_urls = []
        for frame_path in local_frames:
            # Upload to temp bucket with project organization
            blob_path = f"projects/{project_id}/sample_frames/{frame_path.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(frame_path))

            # Set lifecycle to auto-delete after 1 day
            blob.metadata = {"temp": "true", "project_id": project_id}
            blob.patch()

            temp_url = f"gs://{GCS_TEMP_BUCKET}/{blob_path}"
            temp_urls.append(temp_url)

            # Delete local file
            frame_path.unlink()

        print(f"[Orchestrator] Uploaded {len(temp_urls)} sample frames to temp bucket")
        return temp_urls

    except Exception as e:
        print(f"Failed to upload sample frames to temp bucket: {e}")
        # Fallback to local paths if upload fails
        return [str(f) for f in local_frames]


async def _extract_frames_for_segment(
    video_path: str,
    project_id: str,
    start_time: float,
    end_time: float,
    frame_count: int,
    upload_to_gcs: bool = True,
) -> list[Path | str]:
    """
    Extract frames for a specific time segment.

    Uses ffmpeg to extract frames from start_time to end_time.

    Args:
        video_path: Video source
        project_id: Project ID for organizing outputs
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        frame_count: Number of frames to extract
        upload_to_gcs: Whether to upload to GCS

    Returns:
        List of frame paths (GCS URLs or local paths)
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

    # Create output directory
    output_dir = (
        Path(LOCAL_MEDIA_DIR)
        / DERIVED_MEDIA_DIR
        / project_id
        / FRAMES_DIR
        / f"segment_{int(start_time)}_{int(end_time)}"
    )
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

    # Rename frames with millisecond timestamps
    renamed_frames = []
    for idx, frame_path in enumerate(local_frames):
        timestamp_ms = int((start_time + idx * interval) * 1000)
        new_name = output_dir / f"frame_{timestamp_ms}ms.jpg"
        frame_path.rename(new_name)
        renamed_frames.append(new_name)

    if upload_to_gcs:
        # Upload to GCS
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_ASSETS_BUCKET)

            gcs_frames = []
            for frame_path in renamed_frames:
                blob_path = f"projects/{project_id}/signals/frames/{frame_path.name}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(frame_path))

                gcs_url = f"gs://{GCS_ASSETS_BUCKET}/{blob_path}"
                gcs_frames.append(gcs_url)

                # Delete local file
                frame_path.unlink()

            return gcs_frames
        except Exception as e:
            print(f"Failed to upload segment frames to GCS: {e}")
            return renamed_frames
    else:
        return renamed_frames
