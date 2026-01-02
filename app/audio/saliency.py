"""Stream B: Full Audio Saliency Detection

Detects perceptually impactful moments in full audio (speech + music + effects)
regardless of semantic content.

Key Functions:
- Energy peak detection (sudden loudness spikes)
- Spectral change detection (timbre shifts, sound effects)
- Silence-to-impact detection (quiet → loud dramatic moments)
- Non-speech sound detection (music drops, explosions, etc.)

Timeline Format:
All detections return unified format compatible with Stream A:
{
    "time": float,       # Timestamp in seconds
    "type": str,         # Event type (e.g., "energy_peak")
    "score": float,      # Saliency score 0.0-1.0
    "source": str,       # Always "audio" for Stream B
    "metadata": dict     # Additional details
}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AudioSaliencyDetector:
    """
    Detect perceptually salient moments in audio.

    Stream B focuses on WHAT HITS emotionally/perceptually:
    - Energy spikes
    - Sound effects
    - Music drops
    - Dramatic silences
    """

    def __init__(
        self,
        energy_peak_threshold: float = 0.15,
        spectral_change_threshold: float = 200.0,
        silence_threshold: float = 0.05,
        impact_threshold: float = 0.20,
    ):
        """
        Initialize saliency detector.

        Args:
            energy_peak_threshold: Minimum RMS delta for energy peak
            spectral_change_threshold: Minimum spectral centroid delta (Hz)
            silence_threshold: Maximum RMS for silence
            impact_threshold: Minimum RMS for impact moment
        """
        self.energy_peak_threshold = energy_peak_threshold
        self.spectral_change_threshold = spectral_change_threshold
        self.silence_threshold = silence_threshold
        self.impact_threshold = impact_threshold

    def detect_energy_peaks(
        self, audio_features: list[dict[str, Any]], percentile: float = 90.0
    ) -> list[dict[str, Any]]:
        """
        Detect sudden energy spikes (loudness jumps).

        Finds moments where audio suddenly gets much louder:
        - Music drops
        - Sound effects
        - Sudden sounds

        Args:
            audio_features: Output from analyze_audio_features()
            percentile: Percentile threshold for peaks (default: top 10%)

        Returns:
            List of energy peak moments:
            [
                {
                    "time": 5.2,
                    "start": 5.1,
                    "end": 5.3,
                    "type": "energy_peak",
                    "score": 0.95,
                    "source": "audio",
                    "metadata": {
                        "energy": 0.25,
                        "energy_delta": 0.18,
                        "peak_ratio": 3.2
                    }
                },
                ...
            ]
        """
        segments = audio_features
        energy = np.array([seg["energy"] for seg in segments])

        # Calculate energy delta (change rate)
        energy_delta = np.abs(np.diff(energy, prepend=energy[0]))

        # Find peaks above threshold
        threshold = np.percentile(energy_delta, percentile)
        threshold = max(threshold, self.energy_peak_threshold)

        peak_indices = np.where(energy_delta > threshold)[0]

        peaks = []
        for idx in peak_indices:
            seg = segments[idx]
            # Skip if too close to previous peak (within 0.5s)
            if peaks and (seg["start"] - peaks[-1]["time"]) < 0.5:
                continue

            # Calculate peak ratio (current / previous)
            if idx > 0 and energy[idx - 1] > 0:
                peak_ratio = energy[idx] / energy[idx - 1]
            else:
                peak_ratio = 2.0

            # Normalize score [0, 1]
            score = min(energy_delta[idx] / 0.3, 1.0)

            peaks.append(
                {
                    "time": float(seg["start"]),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "type": "energy_peak",
                    "score": float(score),
                    "source": "audio",
                    "metadata": {
                        "energy": float(energy[idx]),
                        "energy_delta": float(energy_delta[idx]),
                        "peak_ratio": float(peak_ratio),
                    },
                }
            )

        return peaks

    def detect_spectral_changes(
        self, audio_features: list[dict[str, Any]], percentile: float = 85.0
    ) -> list[dict[str, Any]]:
        """
        Detect timbre/frequency content changes.

        Finds moments where the character of sound changes:
        - Music genre shifts
        - New instruments entering
        - Sound effect triggers
        - Scene changes

        Args:
            audio_features: Output from analyze_audio_features()
            percentile: Percentile threshold for changes (default: top 15%)

        Returns:
            List of spectral change moments:
            [
                {
                    "time": 12.8,
                    "start": 12.7,
                    "end": 12.9,
                    "type": "spectral_change",
                    "score": 0.82,
                    "source": "audio",
                    "metadata": {
                        "centroid_delta": 250.5,
                        "brightness_change": "darker"
                    }
                },
                ...
            ]
        """
        segments = audio_features
        spectral_brightness = np.array([seg["spectral_brightness"] for seg in segments])

        # Calculate spectral flux (change in brightness)
        spectral_delta = np.abs(np.diff(spectral_brightness, prepend=spectral_brightness[0]))

        # Find changes above threshold
        threshold = np.percentile(spectral_delta, percentile)
        threshold = max(threshold, self.spectral_change_threshold)

        change_indices = np.where(spectral_delta > threshold)[0]

        changes = []
        for idx in change_indices:
            seg = segments[idx]
            # Skip if too close to previous change (within 0.5s)
            if changes and (seg["start"] - changes[-1]["time"]) < 0.5:
                continue

            # Determine brightness direction
            if idx > 0:
                brightness_change = (
                    "brighter"
                    if spectral_brightness[idx] > spectral_brightness[idx - 1]
                    else "darker"
                )
            else:
                brightness_change = "neutral"

            # Normalize score [0, 1]
            score = min(spectral_delta[idx] / 500.0, 1.0)

            changes.append(
                {
                    "time": float(seg["start"]),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "type": "spectral_change",
                    "score": float(score),
                    "source": "audio",
                    "metadata": {
                        "centroid_delta": float(spectral_delta[idx]),
                        "brightness_change": brightness_change,
                        "centroid_before": float(spectral_brightness[max(0, idx - 1)]),
                        "centroid_after": float(spectral_brightness[idx]),
                    },
                }
            )

        return changes

    def detect_silence_to_impact(
        self, audio_features: list[dict[str, Any]], silence_window: int = 10
    ) -> list[dict[str, Any]]:
        """
        Detect dramatic silence → loud moments.

        Finds patterns of quiet build-up followed by sudden impact:
        - Tension → release
        - Dramatic pauses → reveals
        - Quiet moments → explosions/drops

        Args:
            audio_features: Output from analyze_audio_features()
            silence_window: Number of frames to check for silence (default: 10 ≈ 0.5s)

        Returns:
            List of silence-to-impact moments:
            [
                {
                    "time": 23.4,
                    "start": 23.3,
                    "end": 23.5,
                    "type": "silence_to_impact",
                    "score": 0.88,
                    "source": "audio",
                    "metadata": {
                        "silence_duration": 1.2,
                        "impact_energy": 0.28,
                        "contrast_ratio": 5.6
                    }
                },
                ...
            ]
        """
        segments = audio_features
        energy = np.array([seg["energy"] for seg in segments])

        impacts = []

        for idx in range(silence_window, len(segments)):
            # Check if current moment is loud
            if energy[idx] < self.impact_threshold:
                continue

            # Check if previous window was quiet
            silence_window_energy = energy[idx - silence_window : idx]
            if np.mean(silence_window_energy) > self.silence_threshold:
                continue

            # Calculate silence duration
            silence_start_idx = idx - silence_window
            for i in range(idx - 1, max(0, idx - 50), -1):  # Look back up to 2.5s
                if energy[i] > self.silence_threshold:
                    silence_start_idx = i + 1
                    break

            silence_duration = segments[idx]["start"] - segments[silence_start_idx]["start"]

            # Skip very short silences (< 0.3s)
            if silence_duration < 0.3:
                continue

            # Calculate contrast ratio
            silence_avg = np.mean(energy[silence_start_idx:idx])
            if silence_avg > 0:
                contrast_ratio = energy[idx] / silence_avg
            else:
                contrast_ratio = 10.0

            # Skip if already detected nearby impact
            if impacts and (segments[idx]["start"] - impacts[-1]["time"]) < 1.0:
                continue

            # Normalize score [0, 1]
            # Higher score for longer silence + bigger contrast
            silence_score = min(silence_duration / 2.0, 1.0)
            contrast_score = min(contrast_ratio / 10.0, 1.0)
            score = 0.6 * contrast_score + 0.4 * silence_score

            impacts.append(
                {
                    "time": float(segments[idx]["start"]),
                    "start": float(segments[idx]["start"]),
                    "end": float(segments[idx]["end"]),
                    "type": "silence_to_impact",
                    "score": float(score),
                    "source": "audio",
                    "metadata": {
                        "silence_duration": float(silence_duration),
                        "impact_energy": float(energy[idx]),
                        "contrast_ratio": float(contrast_ratio),
                        "silence_start": float(segments[silence_start_idx]["start"]),
                    },
                }
            )

        return impacts

    def detect_non_speech_sounds(
        self, audio_features: list[dict[str, Any]], speech_segments: list[dict[str, float]]
    ) -> list[dict[str, Any]]:
        """
        Detect high-impact sounds outside of speech segments.

        Identifies impactful non-voice sounds:
        - Music crescendos
        - Percussive hits
        - Explosions
        - Sound effects

        Args:
            audio_features: Output from analyze_audio_features()
            speech_segments: Speech segments from VAD (to exclude)

        Returns:
            List of non-speech sound moments:
            [
                {
                    "time": 8.7,
                    "start": 8.6,
                    "end": 8.8,
                    "type": "non_speech_sound",
                    "score": 0.75,
                    "source": "audio",
                    "metadata": {
                        "sound_type": "percussive",
                        "energy": 0.22,
                        "zcr": 0.15
                    }
                },
                ...
            ]
        """
        segments = audio_features
        energy = np.array([seg["energy"] for seg in segments])
        zcr = np.array([seg["zero_crossing_rate"] for seg in segments])

        # Create speech mask
        speech_mask = np.zeros(len(segments), dtype=bool)
        for speech_seg in speech_segments:
            for i, seg in enumerate(segments):
                # Check if segment overlaps with speech segment
                if not (seg["end"] <= speech_seg["start"] or seg["start"] >= speech_seg["end"]):
                    speech_mask[i] = True

        non_speech_sounds = []

        # Find high-energy non-speech moments
        high_energy_threshold = np.percentile(energy, 75)

        for idx, seg in enumerate(segments):
            # Skip speech segments
            if speech_mask[idx]:
                continue

            # Skip low energy
            if energy[idx] < high_energy_threshold:
                continue

            # Skip if too close to previous detection
            if non_speech_sounds and (seg["start"] - non_speech_sounds[-1]["time"]) < 0.5:
                continue

            # Classify sound type based on features
            sound_type = "unknown"
            if zcr[idx] > 0.1:  # High ZCR = percussive/noisy
                sound_type = "percussive"
            elif energy[idx] > 0.2:  # Very high energy
                sound_type = "impact"
            else:
                sound_type = "tonal"

            # Normalize score [0, 1]
            score = min(energy[idx] / 0.3, 1.0)

            non_speech_sounds.append(
                {
                    "time": float(seg["start"]),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "type": "non_speech_sound",
                    "score": float(score),
                    "source": "audio",
                    "metadata": {
                        "sound_type": sound_type,
                        "energy": float(energy[idx]),
                        "zcr": float(zcr[idx]),
                    },
                }
            )

        return non_speech_sounds


def detect_audio_saliency(
    audio_features: list[dict[str, Any]], speech_segments: list[dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    """
    Complete Stream B analysis: perceptual saliency detection.

    Detects all types of impactful audio moments and combines them into segments.

    Args:
        audio_features: Output from analyze_audio_features() - list of segments with start/end and features
        speech_segments: Optional speech segments from VAD (for non-speech detection)

    Returns:
        List of segments with all saliency features combined (same format as speech_segments):
        [
            {
                "start": 0.0,
                "end": 0.1,
                "pitch": 120.5,
                "pitch_variance": 10.2,
                "energy": 0.15,
                "zero_crossing_rate": 0.05,
                "spectral_brightness": 1500.0,
                "spectral_rolloff": 3000.0,
                "has_energy_peak": True,
                "energy_peak_score": 0.95,
                "has_spectral_change": False,
                "has_silence_to_impact": False,
                "has_non_speech_sound": True,
                "non_speech_sound_type": "percussive",
                "non_speech_sound_score": 0.75,
                "saliency_score": 0.85,  # Combined saliency score
            },
            ...
        ]
    """
    detector = AudioSaliencyDetector()

    # 1. Detect energy peaks
    logger.debug("Detecting energy peaks...")
    energy_peaks = detector.detect_energy_peaks(audio_features)

    # 2. Detect spectral changes
    logger.debug("Detecting spectral changes...")
    spectral_changes = detector.detect_spectral_changes(audio_features)

    # 3. Detect silence-to-impact moments
    logger.debug("Detecting silence-to-impact moments...")
    silence_impacts = detector.detect_silence_to_impact(audio_features)

    # 4. Detect non-speech sounds (if speech segments provided)
    non_speech_sounds = []
    if speech_segments:
        logger.debug("Detecting non-speech sounds...")
        non_speech_sounds = detector.detect_non_speech_sounds(audio_features, speech_segments)

    # 5. Combine all detections into segments (same format as speech_segments)
    # Start with base audio_features segments and add saliency features
    saliency_segments = []

    # Create lookup maps for efficient matching
    energy_peaks_by_segment = {}
    for peak in energy_peaks:
        # Find which audio_features segment this peak belongs to
        for i, seg in enumerate(audio_features):
            if seg["start"] <= peak["start"] < seg["end"]:
                if i not in energy_peaks_by_segment:
                    energy_peaks_by_segment[i] = []
                energy_peaks_by_segment[i].append(peak)
                break

    spectral_changes_by_segment = {}
    for change in spectral_changes:
        for i, seg in enumerate(audio_features):
            if seg["start"] <= change["start"] < seg["end"]:
                if i not in spectral_changes_by_segment:
                    spectral_changes_by_segment[i] = []
                spectral_changes_by_segment[i].append(change)
                break

    silence_impacts_by_segment = {}
    for impact in silence_impacts:
        for i, seg in enumerate(audio_features):
            if seg["start"] <= impact["start"] < seg["end"]:
                if i not in silence_impacts_by_segment:
                    silence_impacts_by_segment[i] = []
                silence_impacts_by_segment[i].append(impact)
                break

    non_speech_by_segment = {}
    for sound in non_speech_sounds:
        for i, seg in enumerate(audio_features):
            if seg["start"] <= sound["start"] < seg["end"]:
                if i not in non_speech_by_segment:
                    non_speech_by_segment[i] = []
                non_speech_by_segment[i].append(sound)
                break

    # Build unified segments with all features
    for i, seg in enumerate(audio_features):
        # Start with base segment features
        saliency_seg = {
            "start": seg["start"],
            "end": seg["end"],
            "pitch": seg["pitch"],
            "pitch_variance": seg["pitch_variance"],
            "energy": seg["energy"],
            "zero_crossing_rate": seg["zero_crossing_rate"],
            "spectral_brightness": seg["spectral_brightness"],
            "spectral_rolloff": seg["spectral_rolloff"],
        }

        # Add energy peak features
        if i in energy_peaks_by_segment:
            peaks = energy_peaks_by_segment[i]
            saliency_seg["has_energy_peak"] = True
            saliency_seg["energy_peak_score"] = max(p["score"] for p in peaks)
            saliency_seg["energy_peak_count"] = len(peaks)
        else:
            saliency_seg["has_energy_peak"] = False
            saliency_seg["energy_peak_score"] = 0.0
            saliency_seg["energy_peak_count"] = 0

        # Add spectral change features
        if i in spectral_changes_by_segment:
            changes = spectral_changes_by_segment[i]
            saliency_seg["has_spectral_change"] = True
            saliency_seg["spectral_change_score"] = max(c["score"] for c in changes)
            saliency_seg["spectral_change_count"] = len(changes)
        else:
            saliency_seg["has_spectral_change"] = False
            saliency_seg["spectral_change_score"] = 0.0
            saliency_seg["spectral_change_count"] = 0

        # Add silence-to-impact features
        if i in silence_impacts_by_segment:
            impacts = silence_impacts_by_segment[i]
            saliency_seg["has_silence_to_impact"] = True
            saliency_seg["silence_to_impact_score"] = max(imp["score"] for imp in impacts)
            saliency_seg["silence_to_impact_count"] = len(impacts)
        else:
            saliency_seg["has_silence_to_impact"] = False
            saliency_seg["silence_to_impact_score"] = 0.0
            saliency_seg["silence_to_impact_count"] = 0

        # Add non-speech sound features
        if i in non_speech_by_segment:
            sounds = non_speech_by_segment[i]
            saliency_seg["has_non_speech_sound"] = True
            saliency_seg["non_speech_sound_score"] = max(s["score"] for s in sounds)
            saliency_seg["non_speech_sound_count"] = len(sounds)
            # Use the most common sound type
            sound_types = [s["metadata"]["sound_type"] for s in sounds]
            saliency_seg["non_speech_sound_type"] = max(set(sound_types), key=sound_types.count)
        else:
            saliency_seg["has_non_speech_sound"] = False
            saliency_seg["non_speech_sound_score"] = 0.0
            saliency_seg["non_speech_sound_count"] = 0
            saliency_seg["non_speech_sound_type"] = None

        # Calculate combined saliency score (weighted average of all detection scores)
        saliency_scores = []
        if saliency_seg["has_energy_peak"]:
            saliency_scores.append(saliency_seg["energy_peak_score"])
        if saliency_seg["has_spectral_change"]:
            saliency_scores.append(saliency_seg["spectral_change_score"])
        if saliency_seg["has_silence_to_impact"]:
            saliency_scores.append(saliency_seg["silence_to_impact_score"])
        if saliency_seg["has_non_speech_sound"]:
            saliency_scores.append(saliency_seg["non_speech_sound_score"])

        if saliency_scores:
            saliency_seg["saliency_score"] = float(sum(saliency_scores) / len(saliency_scores))
        else:
            saliency_seg["saliency_score"] = 0.0

        saliency_segments.append(saliency_seg)

    logger.info(
        "Stream B saliency detection complete: %d segments, %d energy peaks, %d spectral changes, %d silence→impact, %d non-speech sounds",
        len(saliency_segments),
        len(energy_peaks),
        len(spectral_changes),
        len(silence_impacts),
        len(non_speech_sounds),
    )

    return saliency_segments
