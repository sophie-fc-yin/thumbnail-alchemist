"""Analysis and orchestration modules.

Handles adaptive sampling, pace analysis, and output formatting.
"""

from app.analysis.adaptive_sampling import orchestrate_adaptive_sampling
from app.analysis.pace_analysis import (
    calculate_audio_energy_delta,
    calculate_pace_score,
    calculate_speech_emotion_delta,
    pace_to_sampling_interval,
    segment_video_by_pace,
)

__all__ = [
    "orchestrate_adaptive_sampling",
    "calculate_pace_score",
    "calculate_audio_energy_delta",
    "calculate_speech_emotion_delta",
    "pace_to_sampling_interval",
    "segment_video_by_pace",
]
