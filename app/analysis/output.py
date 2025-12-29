"""Build and save comprehensive analysis JSON output.

This module creates a complete JSON artifact containing all timelines,
features, scores, and extracted frames from adaptive sampling.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from google.cloud import storage


def build_comprehensive_analysis_json(
    project_id: str,
    video_path: str,
    stream_a_results: dict[str, Any] | None,
    stream_b_results: dict[str, Any] | None,
    visual_analysis: list[dict[str, Any]],
    merged_timeline: list[dict[str, Any]],
    extracted_frames: list[dict[str, Any]],
    pace_segments: list[dict[str, Any]],
    pace_statistics: dict[str, Any],
    processing_stats: dict[str, Any],
    audio_features: dict[str, Any] | None = None,
    transcript_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build comprehensive analysis JSON with all timelines and features.

    Args:
        project_id: Project identifier
        video_path: Source video path
        stream_a_results: Speech semantics analysis results
        stream_b_results: Audio saliency analysis results
        visual_analysis: Face analysis results for each sample frame
        merged_timeline: Combined timeline from all streams
        extracted_frames: Final extracted frames with metadata
        pace_segments: Pace segmentation results
        pace_statistics: Pace statistics summary
        processing_stats: Processing time statistics
        audio_features: Raw audio features (optional)
        transcript_data: Transcription data (optional)

    Returns:
        Complete analysis dictionary ready for JSON serialization
    """
    output = {
        # Metadata
        "project_id": project_id,
        "video_path": video_path,
        "processing_timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
        # Stream A: Speech Semantics
        "stream_a": {
            "enabled": stream_a_results is not None,
            "toned_segments": stream_a_results.get("toned_segments", [])
            if stream_a_results
            else [],
            "style_changes": stream_a_results.get("style_changes", []) if stream_a_results else [],
            "narrative_moments": (
                stream_a_results.get("narrative_moments", []) if stream_a_results else []
            ),
            "timeline": (
                stream_a_results.get("importance_timeline", []) if stream_a_results else []
            ),
            "total_moments": (
                len(stream_a_results.get("importance_timeline", [])) if stream_a_results else 0
            ),
        },
        # Stream B: Audio Saliency
        "stream_b": {
            "enabled": stream_b_results is not None,
            "energy_peaks": stream_b_results.get("energy_peaks", []) if stream_b_results else [],
            "spectral_changes": (
                stream_b_results.get("spectral_changes", []) if stream_b_results else []
            ),
            "silence_to_impact": (
                stream_b_results.get("silence_to_impact", []) if stream_b_results else []
            ),
            "non_speech_sounds": (
                stream_b_results.get("non_speech_sounds", []) if stream_b_results else []
            ),
            "timeline": stream_b_results.get("saliency_timeline", []) if stream_b_results else [],
            "total_moments": (
                len(stream_b_results.get("saliency_timeline", [])) if stream_b_results else 0
            ),
        },
        # Visual Analysis
        "visual_analysis": {
            "sample_frames": visual_analysis,
            "total_analyzed": len(visual_analysis),
            "faces_detected": sum(1 for frame in visual_analysis if frame.get("has_face", False)),
        },
        # Merged Timeline (all streams combined)
        "merged_timeline": merged_timeline,
        # Final Extracted Frames
        "extracted_frames": extracted_frames,
        # Pace Analysis (current method)
        "pace_analysis": {
            "segments": pace_segments,
            "statistics": pace_statistics,
        },
        # Processing Statistics
        "processing_stats": processing_stats,
        # Optional: Raw audio features
        "audio_features": audio_features if audio_features else None,
        # Optional: Transcript data
        "transcript": transcript_data if transcript_data else None,
    }

    return output


async def save_analysis_json_to_gcs(
    analysis_json: dict[str, Any],
    project_id: str,
    bucket_name: str = "clickmoment-prod-assets",
) -> str:
    """
    Save comprehensive analysis JSON to GCS.

    Args:
        analysis_json: Complete analysis dictionary
        project_id: Project identifier
        bucket_name: GCS bucket name

    Returns:
        GCS URL of saved JSON file
    """
    try:
        # Convert to JSON string
        json_str = json.dumps(analysis_json, indent=2, ensure_ascii=False)

        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blob_path = f"projects/{project_id}/analysis/adaptive_sampling_analysis.json"
        blob = bucket.blob(blob_path)

        # Upload with metadata
        blob.upload_from_string(
            json_str,
            content_type="application/json",
        )

        blob.metadata = {
            "project_id": project_id,
            "timestamp": analysis_json["processing_timestamp"],
            "version": analysis_json.get("version", "1.0"),
        }
        blob.patch()

        gcs_url = f"gs://{bucket_name}/{blob_path}"
        print(f"[Analysis Output] Saved comprehensive analysis to {gcs_url}")

        return gcs_url

    except Exception as e:
        print(f"[Analysis Output] Failed to save analysis JSON: {e}")
        # Fallback: save locally
        try:
            local_path = Path(f"analysis_{project_id}.json")
            local_path.write_text(json_str, encoding="utf-8")
            print(f"[Analysis Output] Saved analysis locally to {local_path}")
            return str(local_path)
        except Exception as local_error:
            print(f"[Analysis Output] Failed to save locally: {local_error}")
            raise


def merge_stream_timelines(
    stream_a_timeline: list[dict[str, Any]] | None,
    stream_b_timeline: list[dict[str, Any]] | None,
    visual_frames: list[dict[str, Any]],
    temporal_window: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Merge Stream A + B timelines with visual analysis.

    Groups moments within temporal_window and combines scores.

    Args:
        stream_a_timeline: Speech semantics timeline
        stream_b_timeline: Audio saliency timeline
        visual_frames: Visual analysis results with timestamps
        temporal_window: Time window (seconds) for grouping moments

    Returns:
        Merged timeline sorted by time with combined scores
    """
    # Collect all moments
    all_moments = []

    if stream_a_timeline:
        all_moments.extend(stream_a_timeline)

    if stream_b_timeline:
        all_moments.extend(stream_b_timeline)

    # Sort by time
    all_moments.sort(key=lambda x: x["time"])

    # Group moments within temporal window
    merged = []
    i = 0

    while i < len(all_moments):
        current = all_moments[i]
        cluster = [current]

        # Find nearby moments within window
        j = i + 1
        while (
            j < len(all_moments) and (all_moments[j]["time"] - current["time"]) <= temporal_window
        ):
            cluster.append(all_moments[j])
            j += 1

        # Calculate combined score (weighted average)
        total_score = sum(m["score"] for m in cluster)
        avg_score = total_score / len(cluster)

        # Boost score if multiple sources agree
        source_bonus = 0.1 * (len(cluster) - 1)  # +0.1 for each additional source
        combined_score = min(avg_score + source_bonus, 1.0)

        # Find closest visual frame
        avg_time = sum(m["time"] for m in cluster) / len(cluster)
        closest_visual = min(
            visual_frames,
            key=lambda f: abs(f["timestamp"] - avg_time),
            default=None,
        )

        # Build merged moment
        merged_moment = {
            "time": avg_time,
            "score": combined_score,
            "sources": [m["source"] for m in cluster],
            "types": [m["type"] for m in cluster],
            "num_signals": len(cluster),
            "features": {
                "speech": [m for m in cluster if m["source"] == "speech"],
                "audio": [m for m in cluster if m["source"] == "audio"],
                "visual": closest_visual if closest_visual else {},
            },
            "metadata": {
                "cluster_size": len(cluster),
                "cluster_moments": cluster,
            },
        }

        merged.append(merged_moment)
        i = j  # Skip to next unprocessed moment

    # Sort by score (highest first)
    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged
