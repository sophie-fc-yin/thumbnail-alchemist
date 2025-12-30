"""Audio processing modules.

Handles audio extraction, speech detection, saliency analysis, and semantics.
"""

from app.audio.extraction import (
    analyze_audio_features,
    extract_audio_from_video,
    transcribe_and_analyze_audio,
)
from app.audio.saliency import detect_audio_saliency
from app.audio.speech_detection import detect_speech_in_audio
from app.audio.speech_semantics import analyze_speech_semantics

__all__ = [
    "extract_audio_from_video",
    "transcribe_and_analyze_audio",
    "analyze_audio_features",
    "detect_speech_in_audio",
    "detect_audio_saliency",
    "analyze_speech_semantics",
]
