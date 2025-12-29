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

from typing import Any

import numpy as np


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
        self, audio_features: dict[str, Any], percentile: float = 90.0
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
        energy = np.array(audio_features["energy"])
        times = np.array(audio_features["times"])

        # Calculate energy delta (change rate)
        energy_delta = np.abs(np.diff(energy, prepend=energy[0]))

        # Find peaks above threshold
        threshold = np.percentile(energy_delta, percentile)
        threshold = max(threshold, self.energy_peak_threshold)

        peak_indices = np.where(energy_delta > threshold)[0]

        peaks = []
        for idx in peak_indices:
            # Skip if too close to previous peak (within 0.5s)
            if peaks and (times[idx] - peaks[-1]["time"]) < 0.5:
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
                    "time": float(times[idx]),
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
        self, audio_features: dict[str, Any], percentile: float = 85.0
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
        spectral_brightness = np.array(audio_features["spectral_brightness"])
        times = np.array(audio_features["times"])

        # Calculate spectral flux (change in brightness)
        spectral_delta = np.abs(np.diff(spectral_brightness, prepend=spectral_brightness[0]))

        # Find changes above threshold
        threshold = np.percentile(spectral_delta, percentile)
        threshold = max(threshold, self.spectral_change_threshold)

        change_indices = np.where(spectral_delta > threshold)[0]

        changes = []
        for idx in change_indices:
            # Skip if too close to previous change (within 0.5s)
            if changes and (times[idx] - changes[-1]["time"]) < 0.5:
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
                    "time": float(times[idx]),
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
        self, audio_features: dict[str, Any], silence_window: int = 10
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
        energy = np.array(audio_features["energy"])
        times = np.array(audio_features["times"])

        impacts = []

        for idx in range(silence_window, len(energy)):
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

            silence_duration = times[idx] - times[silence_start_idx]

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
            if impacts and (times[idx] - impacts[-1]["time"]) < 1.0:
                continue

            # Normalize score [0, 1]
            # Higher score for longer silence + bigger contrast
            silence_score = min(silence_duration / 2.0, 1.0)
            contrast_score = min(contrast_ratio / 10.0, 1.0)
            score = 0.6 * contrast_score + 0.4 * silence_score

            impacts.append(
                {
                    "time": float(times[idx]),
                    "type": "silence_to_impact",
                    "score": float(score),
                    "source": "audio",
                    "metadata": {
                        "silence_duration": float(silence_duration),
                        "impact_energy": float(energy[idx]),
                        "contrast_ratio": float(contrast_ratio),
                        "silence_start": float(times[silence_start_idx]),
                    },
                }
            )

        return impacts

    def detect_non_speech_sounds(
        self, audio_features: dict[str, Any], speech_segments: list[dict[str, float]]
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
        energy = np.array(audio_features["energy"])
        zcr = np.array(audio_features["zero_crossing_rate"])
        times = np.array(audio_features["times"])

        # Create speech mask
        speech_mask = np.zeros(len(times), dtype=bool)
        for segment in speech_segments:
            start_idx = np.searchsorted(times, segment["start"])
            end_idx = np.searchsorted(times, segment["end"])
            speech_mask[start_idx:end_idx] = True

        non_speech_sounds = []

        # Find high-energy non-speech moments
        high_energy_threshold = np.percentile(energy, 75)

        for idx in range(len(times)):
            # Skip speech segments
            if speech_mask[idx]:
                continue

            # Skip low energy
            if energy[idx] < high_energy_threshold:
                continue

            # Skip if too close to previous detection
            if non_speech_sounds and (times[idx] - non_speech_sounds[-1]["time"]) < 0.5:
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
                    "time": float(times[idx]),
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
    audio_features: dict[str, Any], speech_segments: list[dict[str, float]] | None = None
) -> dict[str, Any]:
    """
    Complete Stream B analysis: perceptual saliency detection.

    Detects all types of impactful audio moments and creates unified timeline.

    Args:
        audio_features: Output from analyze_audio_features()
        speech_segments: Optional speech segments from VAD (for non-speech detection)

    Returns:
        Stream B analysis results:
        {
            "energy_peaks": [...],         # Sudden loudness spikes
            "spectral_changes": [...],     # Timbre/frequency shifts
            "silence_to_impact": [...],    # Dramatic quiet → loud
            "non_speech_sounds": [...],    # Music/effects outside speech
            "saliency_timeline": [...]     # Unified timeline (compatible with Stream A)
        }
    """
    detector = AudioSaliencyDetector()

    # 1. Detect energy peaks
    print("[Stream B] Detecting energy peaks...")
    energy_peaks = detector.detect_energy_peaks(audio_features)

    # 2. Detect spectral changes
    print("[Stream B] Detecting spectral changes...")
    spectral_changes = detector.detect_spectral_changes(audio_features)

    # 3. Detect silence-to-impact moments
    print("[Stream B] Detecting silence-to-impact moments...")
    silence_impacts = detector.detect_silence_to_impact(audio_features)

    # 4. Detect non-speech sounds (if speech segments provided)
    non_speech_sounds = []
    if speech_segments:
        print("[Stream B] Detecting non-speech sounds...")
        non_speech_sounds = detector.detect_non_speech_sounds(audio_features, speech_segments)

    # 5. Create unified saliency timeline (compatible with Stream A)
    saliency_timeline = []

    # Add all detections to timeline
    saliency_timeline.extend(energy_peaks)
    saliency_timeline.extend(spectral_changes)
    saliency_timeline.extend(silence_impacts)
    saliency_timeline.extend(non_speech_sounds)

    # Sort by time
    saliency_timeline.sort(key=lambda x: x["time"])

    print(
        f"[Stream B] Saliency detection complete: "
        f"{len(energy_peaks)} energy peaks, "
        f"{len(spectral_changes)} spectral changes, "
        f"{len(silence_impacts)} silence→impact, "
        f"{len(non_speech_sounds)} non-speech sounds"
    )

    return {
        "energy_peaks": energy_peaks,
        "spectral_changes": spectral_changes,
        "silence_to_impact": silence_impacts,
        "non_speech_sounds": non_speech_sounds,
        "saliency_timeline": saliency_timeline,  # UNIFIED FORMAT
    }
