"""Audio extraction and transcription from video sources."""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from openai import OpenAI

from app.models import SourceMedia
from app.vision_media import MediaValidationError, generate_signed_url


async def extract_audio_from_video(
    content_sources: SourceMedia,
    project_id: str,
    max_duration_seconds: int = 600,
    output_dir: Path | None = None,
    ffmpeg_binary: str | None = None,
) -> Path | None:
    """
    Extract audio from video without loading the entire file.

    Uses ffmpeg streaming to extract audio progressively. For long videos,
    extracts only the first N seconds to avoid processing the entire file.

    Args:
        content_sources: SourceMedia containing video_path (GCS URL, other URL, or local path)
        project_id: Unique identifier for the project (used to organize output files)
        max_duration_seconds: Maximum duration to extract in seconds (default: 600 = 10 minutes)
        output_dir: Custom output directory for extracted audio (overrides default structure)
        ffmpeg_binary: Path to ffmpeg binary (auto-detected if None)

    Returns:
        Path to extracted audio file (WAV format), or None if extraction failed
    """
    video_source = content_sources.video_path
    ffmpeg_path = ffmpeg_binary or shutil.which("ffmpeg")
    if not video_source or not ffmpeg_path:
        return None

    # Check if video is local file (validate existence) or URL (generate signed URL if GCS)
    if not video_source.startswith(("http://", "https://", "gs://")):
        # Local file path - validate existence
        video_path_obj = Path(video_source)
        if not video_path_obj.exists():
            return None
    else:
        # GCS URL - generate signed URL for private access
        video_source = generate_signed_url(video_source)

    if output_dir:
        target_dir = output_dir
    else:
        # Default to structured storage: clickmoment-prod-assets/projects/{project_id}/signals/audio
        target_dir = Path("clickmoment-prod-assets") / "projects" / project_id / "signals" / "audio"

    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "audio.wav"

    # Extract audio using ffmpeg streaming
    # -t limits duration to max_duration_seconds
    # -vn removes video stream (audio only)
    # -acodec pcm_s16le for WAV format (16-bit PCM)
    # -ac 1 converts to mono (reduces file size, sufficient for speech)
    # -ar 16000 sets sample rate to 16kHz (standard for speech recognition)
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_source,  # Signed URL for GCS, or direct path/URL
        "-t",
        str(max_duration_seconds),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # WAV codec (16-bit PCM)
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16kHz sample rate
        str(output_path),
        "-y",  # Overwrite output file
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        print(f"ffmpeg failed with return code {process.returncode}: {error_msg}")
        return None

    if output_path.exists():
        return output_path

    print(f"ffmpeg succeeded but output file not found at {output_path}")
    return None


async def analyze_audio_features(audio_path: Path) -> dict[str, Any]:
    """
    Extract audio features for prosody and music analysis.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with frame-by-frame audio features
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Frame parameters (analyze in small windows)
    frame_length = int(0.1 * sr)  # 100ms frames
    hop_length = int(0.05 * sr)  # 50ms hop (50% overlap)

    # Extract features
    # Pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(float(pitch) if pitch > 0 else 0.0)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Zero crossing rate (voice breaks, noisy vs tonal)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[
        0
    ]

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]

    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    # Create timeline from frames
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    # Calculate pitch variance (in windows)
    window_size = 20  # ~1 second windows
    pitch_variance = []
    for i in range(len(pitch_values)):
        start = max(0, i - window_size // 2)
        end = min(len(pitch_values), i + window_size // 2)
        window_pitches = [p for p in pitch_values[start:end] if p > 0]
        variance = float(np.var(window_pitches)) if window_pitches else 0.0
        pitch_variance.append(variance)

    return {
        "times": times.tolist(),
        "pitch": pitch_values,
        "pitch_variance": pitch_variance,
        "energy": rms.tolist(),
        "zero_crossing_rate": zcr.tolist(),
        "spectral_brightness": spectral_centroids.tolist(),
        "spectral_rolloff": spectral_rolloff.tolist(),
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "sample_rate": sr,
    }


def save_timeline_json(timeline_data: dict[str, Any], output_path: Path) -> None:
    """
    Save timeline data to JSON file.

    Args:
        timeline_data: Timeline dictionary from transcribe_and_analyze_audio
        output_path: Path to save JSON file
    """

    # Convert any Path objects to strings for JSON serialization
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    serializable_data = convert_paths(timeline_data)

    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=2)


async def transcribe_and_analyze_audio(
    audio_path: Path,
    project_id: str,
    language: str = "en",
    save_timeline: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive audio understanding using GPT-4o Transcribe Diarize + Librosa.

    Pipeline:
    1. GPT-4o Transcribe Diarize - Automatic transcription with speaker diarization
    2. Librosa - Low-level audio features (pitch, energy, tempo, spectral analysis)
    3. Timeline assembly - Merge all data into unified KEY MOMENTS timeline

    The timeline focuses on key moments for thumbnail selection:
    - Segments (sentences) with avg energy/pitch and speaker labels
    - Speaker turns (transitions between speakers)
    - Energy peaks (high-energy moments, top 10%)
    - Significant pauses (>1 second)
    - Music sections (>3 seconds, detected from spectral features)

    Args:
        audio_path: Path to audio file (WAV format, 16kHz mono recommended)
        project_id: Project identifier for organizing output files
        language: Language code for transcription (default: "en")
        save_timeline: If True, save timeline to JSON file (default: True)

    Returns:
        Dictionary containing comprehensive audio timeline with:
            - timeline: Time-aligned key events (20-50 events, not word-level)
                       All timestamps in milliseconds (start_ms, end_ms, time_ms)
            - transcript: Full text transcript
            - speakers: List of detected speakers with IDs
            - duration_seconds: Total audio duration
            - speech_tone: Overall speech characteristics (pitch, energy, tempo)
            - music_tone: Music characteristics (tempo, loudness, brightness)
            - audio_features: Raw librosa features (times, pitch, energy, etc.)
            - timeline_path: Path to saved audio_timeline.json (if save_timeline=True)

    Raises:
        MediaValidationError: If transcription/analysis fails
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # STEP 1: GPT-4o Transcribe Diarize - Transcription + Speaker Diarization
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",  # Required for audio > 30 seconds
                language=language,
            )

        # STEP 2: Extract audio features (prosody, tone, music)
        audio_features = await analyze_audio_features(audio_path)

        # STEP 3: Extract speakers from transcription
        # The diarized_json format includes speaker info directly in segments
        speakers = set()
        if hasattr(transcription, "segments") and transcription.segments:
            for segment in transcription.segments:
                if hasattr(segment, "speaker") and segment.speaker:
                    speakers.add(segment.speaker)

        # STEP 4: Build unified timeline (KEY MOMENTS ONLY)
        timeline = []

        # Add segment-level events (sentences) with enriched context
        if hasattr(transcription, "segments") and transcription.segments:
            for segment in transcription.segments:
                # Get average energy and pitch for this segment
                segment_features = _get_segment_features(segment.start, segment.end, audio_features)

                segment_event = {
                    "type": "segment",
                    "start_ms": int(segment.start * 1000),
                    "end_ms": int(segment.end * 1000),
                    "text": segment.text,
                    "avg_energy": segment_features["avg_energy"],
                    "avg_pitch": segment_features["avg_pitch"],
                }

                # Add speaker info from diarization (if available)
                if hasattr(segment, "speaker") and segment.speaker:
                    segment_event["speaker"] = segment.speaker

                timeline.append(segment_event)

        # Add speaker turn events (transitions between speakers)
        if hasattr(transcription, "segments") and transcription.segments:
            prev_speaker = None
            for segment in transcription.segments:
                current_speaker = segment.speaker if hasattr(segment, "speaker") else None
                if current_speaker and current_speaker != prev_speaker and prev_speaker is not None:
                    timeline.append(
                        {
                            "type": "speaker_turn",
                            "time_ms": int(segment.start * 1000),
                            "from_speaker": prev_speaker,
                            "to_speaker": current_speaker,
                        }
                    )
                prev_speaker = current_speaker

        # Add SIGNIFICANT pauses only (> 1 second)
        silence_threshold = 0.02
        in_silence = False
        silence_start = 0.0

        for time, energy in zip(audio_features["times"], audio_features["energy"]):
            if energy < silence_threshold and not in_silence:
                silence_start = time
                in_silence = True
            elif energy >= silence_threshold and in_silence:
                if time - silence_start > 1.0:  # Only significant pauses
                    timeline.append(
                        {
                            "type": "pause",
                            "start_ms": int(silence_start * 1000),
                            "end_ms": int(time * 1000),
                            "duration_ms": int((time - silence_start) * 1000),
                        }
                    )
                in_silence = False

        # Add energy peaks (excitement/emphasis moments)
        energy_peaks = _detect_energy_peaks(audio_features)
        for peak in energy_peaks:
            timeline.append(
                {
                    "type": "energy_peak",
                    "time_ms": int(peak["time"] * 1000),
                    "energy": peak["energy"],
                    "context": "high_energy_moment",
                }
            )

        # Add music sections (detected from spectral analysis)
        music_sections = _detect_music_sections(audio_features)
        for section in music_sections:
            timeline.append(
                {
                    "type": "music_section",
                    "start_ms": int(section["start"] * 1000),
                    "end_ms": int(section["end"] * 1000),
                    "intensity": section["intensity"],
                }
            )

        # Sort timeline by start time
        timeline.sort(key=lambda x: x.get("start_ms", x.get("time_ms", 0)))

        # Calculate overall tone characteristics
        speech_tone = {
            "avg_pitch": float(
                sum(p for p in audio_features["pitch"] if p > 0)
                / max(1, sum(1 for p in audio_features["pitch"] if p > 0))
            ),
            "avg_energy": float(sum(audio_features["energy"]) / len(audio_features["energy"])),
            "pitch_variance": float(
                sum(audio_features["pitch_variance"]) / len(audio_features["pitch_variance"])
            ),
            "tempo": audio_features["tempo"],
        }

        music_tone = {
            "tempo": audio_features["tempo"],
            "avg_loudness": float(sum(audio_features["energy"]) / len(audio_features["energy"])),
            "avg_brightness": float(
                sum(audio_features["spectral_brightness"])
                / len(audio_features["spectral_brightness"])
            ),
            "harmonic_ratio": 1.0
            - float(
                sum(audio_features["zero_crossing_rate"])
                / len(audio_features["zero_crossing_rate"])
            ),
        }

        duration = audio_features["times"][-1] if audio_features["times"] else 0.0

        result = {
            "timeline": timeline,
            "transcript": transcription.text,
            "speakers": [{"id": spk, "label": spk} for spk in sorted(speakers)],
            "duration_seconds": duration,
            "speech_tone": speech_tone,
            "music_tone": music_tone,
            "audio_features": audio_features,
        }

        # Save timeline to JSON if requested
        if save_timeline:
            timeline_dir = (
                Path("clickmoment-prod-assets") / "projects" / project_id / "signals" / "audio"
            )
            timeline_dir.mkdir(parents=True, exist_ok=True)
            timeline_path = timeline_dir / "audio_timeline.json"
            save_timeline_json(result, timeline_path)
            result["timeline_path"] = str(timeline_path)

        return result

    except Exception as e:
        raise MediaValidationError(f"Failed to transcribe and analyze audio: {e}") from e


def _calculate_speaking_rate(words: list, center_time: float, window_seconds: float = 2.0) -> float:
    """
    Calculate speaking rate (words per second) in a window around center_time.

    Args:
        words: List of word objects with start/end times
        center_time: Center of analysis window
        window_seconds: Window size in seconds

    Returns:
        Words per second in the window
    """
    window_start = center_time - window_seconds / 2
    window_end = center_time + window_seconds / 2

    words_in_window = [w for w in words if window_start <= w.start <= window_end]

    if len(words_in_window) < 2:
        return 0.0

    time_span = words_in_window[-1].end - words_in_window[0].start
    if time_span <= 0:
        return 0.0

    return len(words_in_window) / time_span


def _get_segment_features(start: float, end: float, audio_features: dict) -> dict:
    """
    Get average audio features for a time segment.

    Args:
        start: Segment start time
        end: Segment end time
        audio_features: Audio features dict from analyze_audio_features

    Returns:
        Dictionary with average features for the segment
    """
    # Find all feature frames within this segment
    times = audio_features["times"]
    indices = [i for i, t in enumerate(times) if start <= t <= end]

    if not indices:
        return {"avg_energy": 0.0, "avg_pitch": 0.0}

    # Calculate averages
    energies = [audio_features["energy"][i] for i in indices]
    pitches = [audio_features["pitch"][i] for i in indices if audio_features["pitch"][i] > 0]

    return {
        "avg_energy": float(sum(energies) / len(energies)) if energies else 0.0,
        "avg_pitch": float(sum(pitches) / len(pitches)) if pitches else 0.0,
    }


def _detect_energy_peaks(audio_features: dict, threshold_percentile: float = 90) -> list[dict]:
    """
    Detect energy peaks (high-energy moments) in audio.

    Args:
        audio_features: Audio features dict
        threshold_percentile: Energy percentile to consider as peak (default: 90)

    Returns:
        List of energy peak events
    """
    import numpy as np

    energies = np.array(audio_features["energy"])
    times = audio_features["times"]

    # Calculate threshold (top 10% by default)
    threshold = np.percentile(energies, threshold_percentile)

    # Find peaks (local maxima above threshold)
    peaks = []
    window_size = 20  # ~1 second window

    for i in range(window_size, len(energies) - window_size):
        if energies[i] > threshold:
            # Check if it's a local maximum
            if energies[i] == max(energies[i - window_size : i + window_size]):
                # Avoid duplicate peaks too close together
                if not peaks or (times[i] - peaks[-1]["time"]) > 2.0:
                    peaks.append(
                        {
                            "time": times[i],
                            "energy": float(energies[i]),
                        }
                    )

    return peaks


def _detect_music_sections(audio_features: dict, energy_threshold: float = 0.1) -> list[dict]:
    """
    Detect sections with background music.

    Uses spectral brightness and harmonic content to identify music.

    Args:
        audio_features: Audio features dict
        energy_threshold: Minimum energy to consider

    Returns:
        List of music section events
    """
    import numpy as np

    times = audio_features["times"]
    energies = np.array(audio_features["energy"])
    brightness = np.array(audio_features["spectral_brightness"])
    zcr = np.array(audio_features["zero_crossing_rate"])

    # Music typically has: high brightness, low ZCR (harmonic), sustained energy
    harmonic_ratio = 1.0 - zcr

    # Combine indicators
    music_score = (brightness / brightness.max()) * harmonic_ratio * (energies > energy_threshold)

    # Find continuous sections with high music score
    sections = []
    in_section = False
    section_start = 0.0

    threshold = 0.5
    min_duration = 3.0  # Minimum 3 seconds

    for time, score in zip(times, music_score):
        if score > threshold and not in_section:
            section_start = time
            in_section = True
        elif score <= threshold and in_section:
            if time - section_start > min_duration:
                # Calculate average intensity for this section
                section_indices = [j for j, t in enumerate(times) if section_start <= t <= time]
                avg_intensity = float(np.mean([energies[j] for j in section_indices]))

                sections.append(
                    {
                        "start": section_start,
                        "end": time,
                        "intensity": avg_intensity,
                    }
                )
            in_section = False

    return sections
