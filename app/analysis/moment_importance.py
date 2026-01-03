"""Moment importance analysis for adaptive frame sampling.

DEPRECATED: This module's functions have been replaced by the new adaptive sampling system.

The old system used formula-based importance calculation with fixed intervals.
The new system uses trigger-based multi-signal fusion with adaptive intervals.

See: app/analysis/processing.py for the new implementation
"""


# ============================================================================
# CONSTANTS (Kept for reference, may be removed in future cleanup)
# ============================================================================

# Importance Score Weights
DEFAULT_WEIGHT_EXPRESSION = 0.25  # Visual emotion (key for thumbnails)
DEFAULT_WEIGHT_LANDMARK = 0.15  # Head movement, gestures
DEFAULT_WEIGHT_AUDIO = 0.2  # Music, excitement
DEFAULT_WEIGHT_SPEECH = 0.15  # Vocal emphasis
DEFAULT_WEIGHT_AUDIO_SCORE = 0.25  # Comprehensive audio analysis

# Importance Categorization Thresholds
IMPORTANCE_LOW_THRESHOLD = 0.3  # Below this = low importance
IMPORTANCE_MEDIUM_THRESHOLD = 0.7  # Below this = medium, above = high

# Sampling Intervals (seconds) - inverse of importance
DEFAULT_MIN_INTERVAL = 0.1  # Minimum sampling interval (high importance)
DEFAULT_MAX_INTERVAL = 2.0  # Maximum sampling interval (low importance)
LOW_IMPORTANCE_MIN_INTERVAL = 1.5  # Low importance minimum interval
LOW_IMPORTANCE_MAX_INTERVAL = 2.0  # Low importance maximum interval
MEDIUM_IMPORTANCE_MIN_INTERVAL = 0.5  # Medium importance minimum interval
MEDIUM_IMPORTANCE_MAX_INTERVAL = 1.5  # Medium importance maximum interval
HIGH_IMPORTANCE_MAX_INTERVAL = 0.5  # High importance maximum interval

# Segmentation
DEFAULT_SEGMENTATION_THRESHOLD = 0.2  # Importance change threshold

# Signal Processing
DEFAULT_SMOOTHING_WINDOW_SIZE = 5  # Window size for moving average

# Time Conversion
MS_TO_SECONDS = 1000.0  # Milliseconds to seconds conversion factor


# ============================================================================
# DEPRECATED: Legacy Importance Calculation Functions
# ============================================================================
# All functions in this module have been replaced by the new adaptive sampling system.
#
# Deleted functions (formerly lines 45-360):
# - calculate_moment_importance() → Replaced by trigger-based weighting
# - importance_to_sampling_interval() → Replaced by get_adaptive_intervals()
# - segment_video_by_importance() → Replaced by _calculate_importance_segments()
# - calculate_audio_energy_delta() → No longer needed
# - calculate_speech_emotion_delta() → No longer needed
# - _categorize_importance() → No longer needed
#
# New system location:
# - app/analysis/processing.py::get_adaptive_intervals()
# - app/analysis/processing.py::_calculate_importance_segments()
# - app/analysis/processing.py::_assign_sampling_intervals()
#
# See git history for the old implementation if needed for reference.
# This file can be deleted entirely once all references are confirmed removed.
# ============================================================================
