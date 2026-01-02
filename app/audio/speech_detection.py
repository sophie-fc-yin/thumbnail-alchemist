"""Speech detection and voice activity detection using Silero VAD.

This module provides speech/non-speech separation:
- Detects spoken voice segments (creator talking)
- Filters out music, singing, and background noise
- Uses Silero VAD for accurate voice activity detection
- Uses pitch analysis to distinguish speaking from singing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.constants import (
    DEFAULT_SAMPLE_RATE,
    MIN_PITCH_ANALYSIS_DURATION_MS,
    PITCH_ANALYSIS_FMAX,
    PITCH_ANALYSIS_FMIN,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_SPEECH_PAD_MS,
)


class SpeechDetector:
    """
    Detect speech segments in audio using Silero VAD.

    Separates spoken voice from music, singing, and background noise.
    """

    def __init__(self):
        """Initialize Silero VAD model."""
        # Load Silero VAD model from torch hub
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        # Extract utility functions
        (self.get_speech_timestamps, _, _, _, _) = utils

        # Set to evaluation mode
        self.model.eval()

    def detect_speech_segments(
        self,
        audio_path: Path | str,
    ) -> list[dict[str, float]]:
        """
        Detect speech segments in audio file.

        Args:
            audio_path: Path to audio file (WAV format)

        Returns:
            List of speech segments with start/end times in seconds:
            [
                {"start": 0.5, "end": 3.2},
                {"start": 5.1, "end": 8.7},
            ]
        """
        # Load audio - always resample to target sample rate for consistency
        wav, file_sample_rate = torchaudio.load(str(audio_path))

        # Silero VAD expects 16kHz mono - always resample to ensure consistency
        if file_sample_rate != DEFAULT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(file_sample_rate, DEFAULT_SAMPLE_RATE)
            wav = resampler(wav)

        sample_rate = DEFAULT_SAMPLE_RATE  # Always use target sample rate

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Get speech timestamps using configurable VAD parameters
        speech_timestamps = self.get_speech_timestamps(
            wav.squeeze(),
            self.model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
            return_seconds=True,
        )

        # Silero VAD already returns timestamps in seconds with start/end fields
        return speech_timestamps

    def filter_singing_segments(
        self,
        audio_path: Path | str,
        speech_segments: list[dict[str, float]],
        pitch_variance_threshold: float = 30.0,
    ) -> list[dict[str, float]]:
        """
        Filter out singing from speech segments using pitch analysis.

        Speaking has high pitch variance (natural speech melody).
        Singing has low pitch variance (sustained musical notes).

        Args:
            audio_path: Path to audio file
            speech_segments: Speech segments from VAD
            pitch_variance_threshold: Threshold for speaking vs singing (Hz std dev)

        Returns:
            Filtered segments containing only speaking (no singing)
        """
        # Load audio with librosa for pitch analysis
        # Use DEFAULT_SAMPLE_RATE for consistency (pitch analysis works fine at 16kHz)
        # Full audio is already mono from extraction, but specify mono=True for consistency
        y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)

        speaking_segments = []

        for segment in speech_segments:
            # Extract segment audio
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            segment_audio = y[start_sample:end_sample]

            # Skip very short segments (pitch analysis needs minimum duration)
            min_samples = int(sr * MIN_PITCH_ANALYSIS_DURATION_MS / 1000.0)
            if len(segment_audio) < min_samples:
                continue

            # Analyze pitch within human voice range
            pitches, magnitudes = librosa.piptrack(
                y=segment_audio, sr=sr, fmin=PITCH_ANALYSIS_FMIN, fmax=PITCH_ANALYSIS_FMAX
            )

            # Extract dominant pitch over time
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Only non-zero pitches
                    pitch_values.append(pitch)

            if len(pitch_values) < 5:  # Not enough pitch data
                continue

            # Calculate pitch variance
            pitch_variance = float(np.std(pitch_values))

            # Speaking: high variance (40-80 Hz typical)
            # Singing: low variance (5-20 Hz typical)
            if pitch_variance > pitch_variance_threshold:
                segment["pitch_variance"] = pitch_variance
                segment["is_speaking"] = True
                speaking_segments.append(segment)
            else:
                # This is likely singing, skip it
                continue

        return speaking_segments

    def create_speech_only_audio(
        self,
        audio_path: Path | str,
        speech_segments: list[dict[str, float]],
        output_path: Path | str,
    ) -> Path:
        """
        Create speech-only audio file by extracting and concatenating speech segments.

        Args:
            audio_path: Source audio file
            speech_segments: List of speech segments with start/end times
            output_path: Output path for speech-only audio

        Returns:
            Path to created speech-only audio file
        """
        # Load full audio
        y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)

        # Extract and concatenate speech segments
        speech_audio_segments = []
        for segment in speech_segments:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            segment_audio = y[start_sample:end_sample]
            speech_audio_segments.append(segment_audio)

        if not speech_audio_segments:
            # No speech detected, create silent file
            speech_audio = np.zeros(int(sr * 0.1))  # 100ms silence
        else:
            # Concatenate all speech segments
            speech_audio = np.concatenate(speech_audio_segments)

        # Save speech-only audio
        sf.write(str(output_path), speech_audio, sr)

        return Path(output_path)


def detect_speech_in_audio(
    audio_path: Path | str,
    output_speech_path: Path | str | None = None,
    filter_singing: bool = True,
) -> dict[str, Any]:
    """
    Detect speech in audio and optionally create speech-only file.

    Args:
        audio_path: Path to input audio file
        output_speech_path: Optional path for speech-only output
        filter_singing: Whether to filter out singing segments

    Returns:
        Dictionary with:
            - segments: List of speech segments
            - speech_ratio: Ratio of speech to total audio duration
            - speech_path: Path to speech-only file (if output_speech_path provided)
    """
    detector = SpeechDetector()

    # Detect speech segments
    segments = detector.detect_speech_segments(audio_path)

    # Filter singing if requested
    if filter_singing and segments:
        segments = detector.filter_singing_segments(audio_path, segments)

    # Calculate speech ratio
    # Always use DEFAULT_SAMPLE_RATE for consistency - don't trust file metadata
    # Full audio is already mono from extraction, but specify mono=True for consistency
    y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)
    total_duration = len(y) / sr

    speech_duration = sum(seg["end"] - seg["start"] for seg in segments)
    speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

    result = {
        "segments": segments,
        "speech_ratio": speech_ratio,
        "total_duration": total_duration,
        "speech_duration": speech_duration,
    }

    # Create speech-only file if output path provided
    if output_speech_path and segments:
        speech_path = detector.create_speech_only_audio(audio_path, segments, output_speech_path)
        result["speech_path"] = speech_path

    return result
