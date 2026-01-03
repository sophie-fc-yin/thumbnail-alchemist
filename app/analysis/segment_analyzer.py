"""Segment analyzer for ranking segments and selecting top frames."""

from __future__ import annotations

from typing import Any


def _segment_signal_agreement(segment: dict[str, Any]) -> float:
    """Compute how strongly signals agree for a segment (0-1)."""
    triggers = segment.get("triggers", {}) or {}
    values = []
    for key in ("stream_a", "stream_b", "visual"):
        value = triggers.get(key, 0.0)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)

    if not values:
        return 0.0

    active_count = sum(1 for v in values if v >= 0.5)
    active_ratio = active_count / 3.0
    weighted_ratio = sum(values) / 3.0

    # Favor explicit agreement while still rewarding stronger trigger magnitudes.
    return min(1.0, (active_ratio * 0.6) + (weighted_ratio * 0.4))


def _segment_score(segment: dict[str, Any]) -> float:
    """Blend importance and signal agreement into a single segment score."""
    avg_importance = float(segment.get("avg_importance", 0.0))
    agreement = _segment_signal_agreement(segment)
    return (avg_importance * 0.7) + (agreement * 0.3)


def _frame_visual_score(frame: dict[str, Any]) -> float | None:
    """Compute a visual quality score from available frame metrics."""
    aesthetics = frame.get("aesthetics", {}) or {}
    composition = frame.get("composition", {}) or {}
    technical = frame.get("technical_quality", {}) or {}
    face_quality = frame.get("face_quality", {}) or {}
    editability = frame.get("editability", {}) or {}

    values = [
        aesthetics.get("score"),
        composition.get("overall_score"),
        technical.get("overall_score"),
        face_quality.get("overall_score"),
        editability.get("overall_editability"),
    ]
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _frame_score(frame: dict[str, Any]) -> float:
    """Score a frame using importance plus visual quality."""
    importance_score = float(frame.get("importance_score", 0.0))
    visual_score = _frame_visual_score(frame)
    if visual_score is None:
        return importance_score
    return (importance_score * 0.6) + (visual_score * 0.4)


def select_top_segments_and_frames(
    frames_with_features: list[dict[str, Any]],
    importance_segments: list[dict[str, Any]],
    min_segments: int = 4,
    max_segments: int = 6,
    min_frames_per_segment: int = 1,
    max_frames_per_segment: int = 2,
) -> dict[str, Any]:
    """
    Rank segments by importance + signal agreement and select top frames per segment.

    Args:
        frames_with_features: Extracted frames with vision features and segment metadata
        importance_segments: Segments with triggers/importance values
        min_segments: Minimum number of segments to return (if available)
        max_segments: Maximum number of segments to return
        min_frames_per_segment: Minimum frames to select per segment (if available)
        max_frames_per_segment: Maximum frames to select per segment

    Returns:
        Dictionary with selected segments and frames.
    """
    segments = []
    for idx, segment in enumerate(importance_segments):
        segment_index = segment.get("segment_index", idx)
        score = _segment_score(segment)
        agreement = _segment_signal_agreement(segment)
        segments.append(
            {
                "segment_index": segment_index,
                "start_time": segment.get("start_time", 0.0),
                "end_time": segment.get("end_time", 0.0),
                "importance_level": segment.get("importance_level", "low"),
                "avg_importance": float(segment.get("avg_importance", 0.0)),
                "signal_agreement": agreement,
                "segment_score": score,
                "triggers": segment.get("triggers", {}),
            }
        )

    segments.sort(
        key=lambda s: (s["segment_score"], s["avg_importance"], -s["segment_index"]),
        reverse=True,
    )

    available_segments = len(segments)
    target_count = min(max_segments, available_segments)
    if target_count < min_segments:
        target_count = available_segments

    selected_segments = segments[:target_count]
    selected_indices = {seg["segment_index"] for seg in selected_segments}

    frames_by_segment: dict[int, list[dict[str, Any]]] = {}
    for frame in frames_with_features:
        segment_index = frame.get("segment_index")
        if segment_index is None or segment_index not in selected_indices:
            continue
        frames_by_segment.setdefault(int(segment_index), []).append(frame)

    selected_frames = []
    for segment in selected_segments:
        segment_index = segment["segment_index"]
        segment_frames = frames_by_segment.get(segment_index, [])
        if not segment_frames:
            continue

        scored_frames = []
        for frame in segment_frames:
            scored_frames.append(
                {
                    **frame,
                    "frame_score": _frame_score(frame),
                }
            )

        scored_frames.sort(
            key=lambda f: (f["frame_score"], f.get("importance_score", 0.0)),
            reverse=True,
        )

        take_count = min(max_frames_per_segment, len(scored_frames))
        if take_count < min_frames_per_segment:
            take_count = min(len(scored_frames), min_frames_per_segment)

        selected_frames.extend(scored_frames[:take_count])

    return {
        "selected_segments": selected_segments,
        "selected_frames": selected_frames,
        "selection_stats": {
            "segments_considered": available_segments,
            "segments_selected": len(selected_segments),
            "frames_selected": len(selected_frames),
            "frames_with_features": len(frames_with_features),
        },
    }
