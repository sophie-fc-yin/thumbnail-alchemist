"""Pace analysis for adaptive frame sampling.

Fuses multiple signals (facial expression, audio, speech) to calculate video pace.
Higher pace = more frames needed to capture key moments.
"""

from typing import Any

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

# Pace Score Weights
DEFAULT_WEIGHT_EXPRESSION = 0.25  # Visual emotion (key for thumbnails)
DEFAULT_WEIGHT_LANDMARK = 0.15  # Head movement, gestures
DEFAULT_WEIGHT_AUDIO = 0.2  # Music, excitement
DEFAULT_WEIGHT_SPEECH = 0.15  # Vocal emphasis
DEFAULT_WEIGHT_AUDIO_SCORE = 0.25  # Comprehensive audio analysis

# Pace Categorization Thresholds
PACE_LOW_THRESHOLD = 0.3  # Below this = low pace
PACE_MEDIUM_THRESHOLD = 0.7  # Below this = medium pace, above = high pace

# Sampling Intervals (seconds)
DEFAULT_MIN_INTERVAL = 0.1  # Minimum sampling interval (high pace)
DEFAULT_MAX_INTERVAL = 2.0  # Maximum sampling interval (low pace)
LOW_PACE_MIN_INTERVAL = 1.5  # Low pace minimum interval
LOW_PACE_MAX_INTERVAL = 2.0  # Low pace maximum interval
MEDIUM_PACE_MIN_INTERVAL = 0.5  # Medium pace minimum interval
MEDIUM_PACE_MAX_INTERVAL = 1.5  # Medium pace maximum interval
HIGH_PACE_MAX_INTERVAL = 0.5  # High pace maximum interval

# Segmentation
DEFAULT_SEGMENTATION_THRESHOLD = 0.2  # Pace change threshold

# Signal Processing
DEFAULT_SMOOTHING_WINDOW_SIZE = 5  # Window size for moving average

# Time Conversion
MS_TO_SECONDS = 1000.0  # Milliseconds to seconds conversion factor


def calculate_pace_score(
    expression_delta: float = 0.0,
    landmark_motion: float = 0.0,
    audio_energy_delta: float = 0.0,
    speech_emotion_delta: float = 0.0,
    audio_score: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Calculate pace score by fusing multiple signals.

    This is NOT machine learning - it's signal fusion.
    Each signal is normalized to [0, 1] and weighted.

    Args:
        expression_delta: Change in facial expression intensity [0, 1]
        landmark_motion: Facial landmark movement [0, 1]
        audio_energy_delta: Change in audio energy [0, 1]
        speech_emotion_delta: Change in speech prosody [0, 1]
        audio_score: Comprehensive audio score (speech_gate × text_importance × emphasis × bgm_penalty) [0, 1]
        weights: Optional custom weights. Defaults:
            - expression: 0.25 (visual emotion is key for thumbnails)
            - landmark: 0.15 (head movement, gestures)
            - audio: 0.2 (catches music, excitement)
            - speech: 0.15 (vocal emphasis)
            - audio_score: 0.25 (comprehensive audio analysis - NEW)

    Returns:
        Pace score [0, 1] where:
            - 0.0-0.3: Low pace (calm talking, slow scenes)
            - 0.3-0.7: Medium pace (normal engagement)
            - 0.7-1.0: High pace (emotional peaks, action)
    """
    if weights is None:
        weights = {
            "expression": 0.25,
            "landmark": 0.15,
            "audio": 0.2,
            "speech": 0.15,
            "audio_score": 0.25,  # New comprehensive audio signal
        }

    # Weighted sum
    pace = (
        weights["expression"] * expression_delta
        + weights["landmark"] * landmark_motion
        + weights["audio"] * audio_energy_delta
        + weights["speech"] * speech_emotion_delta
        + weights["audio_score"] * audio_score
    )

    # Clamp to [0, 1]
    return float(min(max(pace, 0.0), 1.0))


def pace_to_sampling_interval(
    pace_score: float,
    min_interval: float = 0.1,
    max_interval: float = 2.0,
) -> float:
    """
    Convert pace score to frame sampling interval.

    Adaptive rule:
        - Low pace (0.0-0.3) → 1.5-2.0s intervals (few frames)
        - Medium pace (0.3-0.7) → 0.5-1.0s intervals (normal density)
        - High pace (0.7-1.0) → 0.1-0.25s intervals (dense capture)

    This mirrors how humans scrub video - slow through calm parts,
    dense sampling through emotional/action moments.

    Args:
        pace_score: Pace score [0, 1]
        min_interval: Minimum sampling interval in seconds (default: 0.1s)
        max_interval: Maximum sampling interval in seconds (default: 2.0s)

    Returns:
        Sampling interval in seconds
    """
    if pace_score < 0.3:
        # Low pace: calm talking, slow scenes
        # Linear interpolation: 1.5s → 2.0s as pace decreases
        return 1.5 + (0.3 - pace_score) / 0.3 * 0.5

    elif pace_score < 0.7:
        # Medium pace: normal engagement
        # Linear interpolation: 0.5s → 1.5s as pace decreases
        return 0.5 + (0.7 - pace_score) / 0.4 * 1.0

    else:
        # High pace: emotional peaks, action
        # Linear interpolation: 0.1s → 0.5s as pace decreases
        return min_interval + (1.0 - pace_score) / 0.3 * 0.4


def segment_video_by_pace(
    pace_scores: list[float],
    timestamps: list[float],
    threshold: float = 0.2,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """
    Segment video into pace regions.

    Groups consecutive similar-pace sections together.
    Important: Segment BEFORE sampling to avoid missing transitions.

    Args:
        pace_scores: List of pace scores for each time point
        timestamps: Corresponding timestamps in seconds
        threshold: Pace change threshold to trigger new segment
        video_duration: Total video duration (optional, for proper end time)

    Returns:
        List of segments with:
            - start_time: Segment start in seconds
            - end_time: Segment end in seconds
            - avg_pace: Average pace score for segment
            - pace_category: "low", "medium", or "high"
    """
    if not pace_scores or not timestamps:
        return []

    # Special case: very few samples - create full-duration segment
    if len(timestamps) <= 2:
        avg_pace = float(np.mean(pace_scores))
        end_time = video_duration if video_duration else timestamps[-1]
        # Ensure minimum duration
        if end_time <= timestamps[0]:
            end_time = timestamps[0] + 10.0  # Minimum 10 seconds

        return [
            {
                "start_time": timestamps[0],
                "end_time": end_time,
                "avg_pace": avg_pace,
                "pace_category": _categorize_pace(avg_pace),
            }
        ]

    segments = []
    current_start = timestamps[0]
    current_paces = [pace_scores[0]]

    for i in range(1, len(pace_scores)):
        pace = pace_scores[i]
        prev_pace = pace_scores[i - 1]

        # Check if pace changed significantly
        if abs(pace - prev_pace) > threshold:
            # End current segment at midpoint between samples
            midpoint = (timestamps[i - 1] + timestamps[i]) / 2.0
            avg_pace = float(np.mean(current_paces))
            segments.append(
                {
                    "start_time": current_start,
                    "end_time": midpoint,
                    "avg_pace": avg_pace,
                    "pace_category": _categorize_pace(avg_pace),
                }
            )

            # Start new segment at midpoint
            current_start = midpoint
            current_paces = [pace]
        else:
            # Continue current segment
            current_paces.append(pace)

    # Add final segment - extend to video duration if available
    if current_paces:
        avg_pace = float(np.mean(current_paces))
        end_time = video_duration if video_duration else timestamps[-1]
        # Ensure end_time is after start
        if end_time <= current_start:
            end_time = current_start + max(10.0, (timestamps[-1] - timestamps[0]) / len(timestamps))

        segments.append(
            {
                "start_time": current_start,
                "end_time": end_time,
                "avg_pace": avg_pace,
                "pace_category": _categorize_pace(avg_pace),
            }
        )

    return segments


def _categorize_pace(pace_score: float) -> str:
    """Categorize pace score into low/medium/high."""
    if pace_score < 0.3:
        return "low"
    elif pace_score < 0.7:
        return "medium"
    else:
        return "high"


def calculate_audio_energy_delta(
    audio_timeline: list[dict[str, Any]],
    window_size: int = 5,
) -> list[float]:
    """
    Calculate audio energy change rate from timeline.

    Args:
        audio_timeline: Timeline from transcribe_and_analyze_audio
        window_size: Number of time points to average over

    Returns:
        List of normalized energy deltas [0, 1]
    """
    if not audio_timeline:
        return []

    # Extract energy values from timeline (from energy peaks and segments)
    energy_values = []
    timestamps = []

    for event in audio_timeline:
        if event["type"] == "energy_peak":
            energy_values.append(event["energy"])
            timestamps.append(event["time_ms"] / 1000.0)  # Convert ms to seconds
        elif event["type"] == "segment":
            energy_values.append(event.get("avg_energy", 0.0))
            timestamps.append(event["start_ms"] / 1000.0)  # Convert ms to seconds

    if not energy_values:
        return []

    # Calculate deltas
    deltas = []
    for i in range(len(energy_values)):
        if i == 0:
            deltas.append(0.0)
        else:
            delta = abs(energy_values[i] - energy_values[i - 1])
            deltas.append(delta)

    # Smooth with moving average
    smoothed = []
    for i in range(len(deltas)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(deltas), i + window_size // 2 + 1)
        avg_delta = np.mean(deltas[start_idx:end_idx])
        smoothed.append(avg_delta)

    # Normalize to [0, 1]
    if max(smoothed) > 0:
        normalized = [d / max(smoothed) for d in smoothed]
    else:
        normalized = smoothed

    return normalized


def calculate_speech_emotion_delta(
    audio_timeline: list[dict[str, Any]],
    window_size: int = 5,
) -> list[float]:
    """
    Calculate speech emotion change rate from timeline.

    Uses pitch variance and energy as proxies for emotional intensity.

    Args:
        audio_timeline: Timeline from transcribe_and_analyze_audio
        window_size: Number of time points to average over

    Returns:
        List of normalized emotion deltas [0, 1]
    """
    if not audio_timeline:
        return []

    # Extract emotion proxies (pitch + energy) from segments
    emotion_values = []
    timestamps = []

    for event in audio_timeline:
        if event["type"] == "segment":
            # Combine pitch and energy as emotion proxy
            pitch = event.get("avg_pitch", 0.0)
            energy = event.get("avg_energy", 0.0)
            emotion = (pitch + energy) / 2.0
            emotion_values.append(emotion)
            timestamps.append(event["start_ms"] / 1000.0)  # Convert ms to seconds

    if not emotion_values:
        return []

    # Calculate deltas
    deltas = []
    for i in range(len(emotion_values)):
        if i == 0:
            deltas.append(0.0)
        else:
            delta = abs(emotion_values[i] - emotion_values[i - 1])
            deltas.append(delta)

    # Smooth with moving average
    smoothed = []
    for i in range(len(deltas)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(deltas), i + window_size // 2 + 1)
        avg_delta = np.mean(deltas[start_idx:end_idx])
        smoothed.append(avg_delta)

    # Normalize to [0, 1]
    if max(smoothed) > 0:
        normalized = [d / max(smoothed) for d in smoothed]
    else:
        normalized = smoothed

    return normalized
