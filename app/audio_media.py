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
    max_duration_seconds: int = 1800,
    output_dir: Path | None = None,
    ffmpeg_binary: str | None = None,
) -> dict[str, Path] | None:
    """
    Extract audio from video with speech/music lane separation.

    Creates two audio lanes:
    1. Speech lane: Isolated spoken voice (creator talking) for ASR/transcription
       - Uses Silero VAD to detect speech segments
       - Filters out singing using pitch analysis
       - Only contains spoken dialogue/narration
    2. Music lane: Full spectrum audio for music/energy analysis
       - Contains everything (music, singing, effects, speech)
       - Normalized for consistent measurements

    Args:
        content_sources: SourceMedia containing video_path (GCS URL, other URL, or local path)
        project_id: Unique identifier for the project (used to organize output files)
        max_duration_seconds: Maximum duration to extract in seconds (default: 1800 = 30 minutes)
        output_dir: Custom output directory for extracted audio (overrides default structure)
        ffmpeg_binary: Path to ffmpeg binary (auto-detected if None)

    Returns:
        Dictionary with audio paths, or None if extraction failed:
        {
            "speech": Path("audio_speech.wav"),  # Speech-only lane
            "music": Path("audio_music.wav"),    # Full audio lane
            "speech_ratio": 0.42,                # % of video that's speech
            "segments": [...]                     # Speech segment timestamps
        }
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
    music_path = target_dir / "audio_music.wav"
    speech_path = target_dir / "audio_speech.wav"

    # STEP 1: Extract full audio (music lane) using ffmpeg streaming
    # -t limits duration to max_duration_seconds
    # -vn removes video stream (audio only)
    # -af loudnorm normalizes audio volume for consistent analysis
    # -acodec pcm_s16le for WAV format (16-bit PCM)
    # -ac 1 converts to mono (reduces file size, sufficient for analysis)
    # -ar 16000 sets sample rate to 16kHz (standard for VAD)
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
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",  # EBU R128 loudness normalization
        "-acodec",
        "pcm_s16le",  # WAV codec (16-bit PCM)
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16kHz sample rate (required for Silero VAD)
        str(music_path),
        "-y",  # Overwrite output file
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        print(f"ffmpeg failed with return code {process.returncode}: {error_msg}")
        return None

    if not music_path.exists():
        print(f"ffmpeg succeeded but output file not found at {music_path}")
        return None

    # STEP 2: Detect speech segments and create speech-only lane
    from app.speech_detection import detect_speech_in_audio

    try:
        speech_result = detect_speech_in_audio(
            audio_path=music_path,
            output_speech_path=speech_path,
            filter_singing=True,  # Filter out singing vocals
        )

        print(
            f"[Audio Extraction] Speech detection complete: "
            f"{len(speech_result['segments'])} segments, "
            f"{speech_result['speech_ratio']:.1%} speech ratio"
        )

        return {
            "speech": speech_path,
            "music": music_path,
            "speech_ratio": speech_result["speech_ratio"],
            "segments": speech_result["segments"],
            "total_duration": speech_result["total_duration"],
            "speech_duration": speech_result["speech_duration"],
        }

    except Exception as e:
        print(f"Speech detection failed: {e}")
        # Fallback: return music lane only, use it for both
        return {
            "speech": music_path,  # Use full audio as fallback
            "music": music_path,
            "speech_ratio": 1.0,
            "segments": [],
            "total_duration": 0.0,
            "speech_duration": 0.0,
        }


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
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=300.0,  # 5 minutes for large audio files
        )

        # STEP 1: GPT-4o Transcribe Diarize - Transcription + Speaker Diarization
        # Retry logic for network issues
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe-diarize",
                        file=audio_file,
                        response_format="diarized_json",
                        chunking_strategy="auto",  # Required for audio > 30 seconds
                        language=language,
                    )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"Transcription attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise  # Re-raise on final attempt

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

                # Calculate audio score for this segment
                audio_scores = calculate_audio_score(
                    segment=segment_event,
                    audio_features=audio_features,
                )

                # Add audio score components to segment
                segment_event.update(audio_scores)

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


# ============================================================================
# AUDIO SCORING COMPONENTS
# ============================================================================


def calculate_speech_gate(
    segment: dict,
    speech_confidence_threshold: float = 0.5,
) -> float:
    """
    Calculate speech gate: binary gate based on speech confidence.

    Formula:
        speech_gate = 1.0 if speech_confidence ≥ threshold
                      0.0 otherwise

    Speech confidence is derived from:
    - Text presence (transcription exists)
    - Speaker assignment (diarization confidence)

    Args:
        segment: Timeline segment with text and speaker fields
        speech_confidence_threshold: Confidence threshold (default: 0.5)

    Returns:
        Binary speech gate: 1.0 if speech present, 0.0 otherwise
    """
    # Speech confidence from text presence
    has_text = bool(segment.get("text", "").strip())

    # Higher confidence if speaker is assigned (diarization worked)
    has_speaker = bool(segment.get("speaker"))

    # Combine confidence signals
    # - Text present = 0.8 confidence
    # - Text + speaker = 1.0 confidence
    if has_text and has_speaker:
        speech_confidence = 1.0
    elif has_text:
        speech_confidence = 0.8
    else:
        speech_confidence = 0.0

    # Binary gate
    return 1.0 if speech_confidence >= speech_confidence_threshold else 0.0


def calculate_text_importance(
    segment: dict,
    claim_keywords: list[str] | None = None,
    emphasis_keywords: list[str] | None = None,
    filler_keywords: list[str] | None = None,
) -> float:
    """
    Calculate text importance based on categorical content analysis.

    Formula:
        text_importance = 1.0  (strong claim / reveal)
                          0.7  (emphasis / contrast)
                          0.4  (neutral explanation)
                          0.1  (filler)

    Args:
        segment: Timeline segment with text field
        claim_keywords: Keywords indicating strong claims/reveals
        emphasis_keywords: Keywords indicating emphasis/contrast
        filler_keywords: Keywords indicating filler content

    Returns:
        Text importance score: 1.0, 0.7, 0.4, or 0.1
    """
    text = segment.get("text", "").strip()

    if not text:
        return 0.1  # Empty text = filler

    text_lower = text.lower()

    # Default keyword lists
    if claim_keywords is None:
        claim_keywords = [
            "discovered",
            "revealed",
            "found",
            "proved",
            "demonstrated",
            "breakthrough",
            "amazing",
            "incredible",
            "shocking",
            "secret",
            "truth",
            "fact",
            "evidence",
            "research shows",
            "study found",
            "this is",
            "here's why",
            "the reason",
            "turns out",
        ]

    if emphasis_keywords is None:
        emphasis_keywords = [
            "but",
            "however",
            "although",
            "instead",
            "actually",
            "important",
            "key",
            "crucial",
            "critical",
            "essential",
            "remember",
            "note",
            "pay attention",
            "listen",
            "focus",
            "versus",
            "compared to",
            "unlike",
            "difference",
        ]

    if filler_keywords is None:
        filler_keywords = [
            "um",
            "uh",
            "like",
            "you know",
            "i mean",
            "sort of",
            "kind of",
            "basically",
            "literally",
            "just",
        ]

    # Check for strong claims/reveals (highest priority)
    has_claim = any(kw in text_lower for kw in claim_keywords)
    has_reveal_markers = any(marker in text for marker in ["!", "?", "...", "—"])

    if has_claim or (has_reveal_markers and len(text.split()) > 10):
        return 1.0  # Strong claim / reveal

    # Check for emphasis/contrast
    has_emphasis = any(kw in text_lower for kw in emphasis_keywords)
    has_caps = any(word.isupper() and len(word) > 2 for word in text.split())

    if has_emphasis or has_caps:
        return 0.7  # Emphasis / contrast

    # Check for filler
    has_filler = any(kw in text_lower for kw in filler_keywords)
    is_short = len(text.split()) < 5

    if has_filler or is_short:
        return 0.1  # Filler

    # Default: neutral explanation
    return 0.4


def calculate_emphasis_score(
    segment: dict,
    audio_features: dict,
    window_seconds: float = 5.0,
) -> float:
    """
    Calculate emphasis score using local energy deviation from rolling baseline.

    Formula:
        emphasis_score = clamp(
            (local_energy - rolling_baseline) / rolling_std,
            0, 1
        )

    This measures how much the segment's energy stands out from the
    surrounding audio context.

    Args:
        segment: Timeline segment with start_ms, end_ms
        audio_features: Audio features dict from analyze_audio_features
        window_seconds: Window size for rolling baseline (default: 5.0s)

    Returns:
        Emphasis score [0, 1] where:
            - 1.0: Energy significantly above baseline (strong emphasis)
            - 0.5: Energy moderately above baseline
            - 0.0: Energy at or below baseline
    """
    start_s = segment.get("start_ms", 0) / 1000.0
    end_s = segment.get("end_ms", 0) / 1000.0

    # Get segment features
    segment_features = _get_segment_features(start_s, end_s, audio_features)
    local_energy = segment_features["avg_energy"]

    # Calculate rolling baseline and std using window around segment
    times = np.array(audio_features["times"])
    energies = np.array(audio_features["energy"])

    # Find center of segment
    segment_center = (start_s + end_s) / 2.0

    # Get window around segment for baseline calculation
    window_start = segment_center - window_seconds / 2.0
    window_end = segment_center + window_seconds / 2.0

    # Get energies in window
    window_mask = (times >= window_start) & (times <= window_end)
    window_energies = energies[window_mask]

    if len(window_energies) < 2:
        # Not enough data for baseline, use global baseline
        rolling_baseline = float(np.mean(energies))
        rolling_std = float(np.std(energies))
    else:
        rolling_baseline = float(np.mean(window_energies))
        rolling_std = float(np.std(window_energies))

    # Avoid division by zero
    if rolling_std < 1e-6:
        rolling_std = 1e-6

    # Calculate normalized deviation
    emphasis = (local_energy - rolling_baseline) / rolling_std

    # Clamp to [0, 1]
    return max(0.0, min(emphasis, 1.0))


def calculate_bgm_penalty(
    segment: dict,
    audio_features: dict,
) -> float:
    """
    Calculate background music penalty based on energy variance.

    Formula:
        bgm_penalty = clamp(1 - energy_variance_normalized, 0.2, 1.0)

    Background music typically has high energy variance (rhythm, beats).
    Speech has more consistent energy (smoother envelope).

    Args:
        segment: Timeline segment with start_ms, end_ms
        audio_features: Audio features dict from analyze_audio_features

    Returns:
        BGM penalty multiplier [0.2, 1.0] where:
            - 1.0: Low variance (clean speech, no music)
            - 0.5-1.0: Moderate variance (light music or animated speech)
            - 0.2: High variance (strong background music)
    """
    start_s = segment.get("start_ms", 0) / 1000.0
    end_s = segment.get("end_ms", 0) / 1000.0

    # Get energies in segment
    times = audio_features["times"]
    indices = [i for i, t in enumerate(times) if start_s <= t <= end_s]

    if not indices:
        return 1.0  # No data, assume no penalty

    energies = [audio_features["energy"][i] for i in indices]

    if len(energies) < 2:
        return 1.0  # Not enough data for variance

    # Calculate energy variance
    energy_variance = float(np.var(energies))

    # Normalize variance to [0, 1] using typical ranges
    # Typical speech: variance ~ 0.001-0.01
    # Music: variance ~ 0.01-0.1+
    max_variance = 0.05  # Threshold for max penalty
    energy_variance_normalized = min(energy_variance / max_variance, 1.0)

    # Calculate penalty
    penalty = 1.0 - energy_variance_normalized

    # Clamp to [0.2, 1.0]
    return max(0.2, min(penalty, 1.0))


def calculate_audio_score(
    segment: dict,
    audio_features: dict,
    claim_keywords: list[str] | None = None,
    emphasis_keywords: list[str] | None = None,
    filler_keywords: list[str] | None = None,
) -> dict[str, float]:
    """
    Calculate comprehensive audio score using all components.

    Formula:
        audio_score = speech_gate × text_importance × emphasis_score × bgm_penalty

    Where:
        - speech_gate: Binary gate (1.0 if speech, 0.0 otherwise)
        - text_importance: Categorical (1.0=claim, 0.7=emphasis, 0.4=neutral, 0.1=filler)
        - emphasis_score: Energy deviation from baseline [0, 1]
        - bgm_penalty: Music penalty based on variance [0.2, 1.0]

    Args:
        segment: Timeline segment with start_ms, end_ms, text
        audio_features: Audio features dict from analyze_audio_features
        claim_keywords: Keywords for strong claims/reveals
        emphasis_keywords: Keywords for emphasis/contrast
        filler_keywords: Keywords for filler content

    Returns:
        Dictionary with all component scores and final audio_score:
            - speech_gate: Binary [0, 1]
            - text_importance: Categorical [0.1, 0.4, 0.7, 1.0]
            - emphasis_score: [0, 1]
            - bgm_penalty: [0.2, 1.0]
            - audio_score: Product of all components
    """
    # Calculate individual components
    speech_gate = calculate_speech_gate(segment)
    text_importance = calculate_text_importance(
        segment,
        claim_keywords=claim_keywords,
        emphasis_keywords=emphasis_keywords,
        filler_keywords=filler_keywords,
    )
    emphasis_score = calculate_emphasis_score(segment, audio_features)
    bgm_penalty = calculate_bgm_penalty(segment, audio_features)

    # Final score (multiplicative)
    audio_score = speech_gate * text_importance * emphasis_score * bgm_penalty

    return {
        "speech_gate": speech_gate,
        "text_importance": text_importance,
        "emphasis_score": emphasis_score,
        "bgm_penalty": bgm_penalty,
        "audio_score": audio_score,
    }
