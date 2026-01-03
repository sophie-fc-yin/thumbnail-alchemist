"""Audio extraction and transcription from video sources."""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from openai import OpenAI

from app.audio.speech_detection import detect_speech_in_audio
from app.constants import (
    AUDIO_FRAME_LENGTH_SECONDS,
    AUDIO_HOP_LENGTH_SECONDS,
    BGM_MAX_PENALTY,
    BGM_MAX_VARIANCE,
    BGM_MIN_PENALTY,
    DEFAULT_CLAIM_KEYWORDS,
    DEFAULT_EMPHASIS_KEYWORDS,
    DEFAULT_FILLER_KEYWORDS,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    EMPHASIS_MIN_STD,
    EMPHASIS_WINDOW_SECONDS,
    ENERGY_PEAK_MIN_SPACING,
    ENERGY_PEAK_PERCENTILE,
    ENERGY_PEAK_WINDOW_SIZE,
    INITIAL_RETRY_DELAY,
    LOCAL_MEDIA_DIR,
    LOUDNORM_I,
    LOUDNORM_LRA,
    LOUDNORM_TP,
    MAX_TRANSCRIPTION_RETRIES,
    MUSIC_ENERGY_THRESHOLD,
    MUSIC_MIN_DURATION,
    MUSIC_SCORE_THRESHOLD,
    PITCH_VARIANCE_WINDOW_SIZE,
    PROJECT_ASSETS_BUCKET,
    SPEECH_CONFIDENCE_THRESHOLD,
    TEXT_IMPORTANCE_CLAIM,
    TEXT_IMPORTANCE_EMPHASIS,
    TEXT_IMPORTANCE_FILLER,
    TEXT_IMPORTANCE_MIN_CLAIM_WORDS,
    TEXT_IMPORTANCE_NEUTRAL,
    TRANSCRIPTION_TIMEOUT,
)
from app.models import SourceMedia
from app.utils.storage import upload_json_to_gcs
from app.vision.extraction import MediaValidationError, generate_signed_url

logger = logging.getLogger(__name__)


async def extract_audio_from_video(
    content_sources: SourceMedia,
    project_id: str,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    output_dir: Path | None = None,
    ffmpeg_binary: str | None = None,
    video_url: str | None = None,
) -> dict[str, Path] | None:
    """
    Extract audio from video with speech/full-audio separation.

    Creates two audio files:
    1. Speech: Isolated spoken voice (creator talking) for ASR/transcription
       - Uses Silero VAD to detect speech segments
       - Filters out singing using pitch analysis
       - Only contains spoken dialogue/narration
    2. Full Audio: Complete audio track for music/energy analysis
       - Contains everything (speech, music, singing, effects)
       - Normalized for consistent measurements

    Args:
        content_sources: SourceMedia containing video_path (GCS URL, other URL, or local path)
        project_id: Unique identifier for the project (used to organize output files)
        max_duration_seconds: Maximum duration to extract in seconds (default: 1800 = 30 minutes)
        output_dir: Custom output directory for extracted audio (overrides default structure)
        ffmpeg_binary: Path to ffmpeg binary (auto-detected if None)
        video_url: Optional pre-generated signed URL (avoids duplicate signing if already done)

    Returns:
        Dictionary with audio paths, or None if extraction failed:
        {
            "speech": Path("audio_speech.wav") | None,  # Speech-only (None if no speech detected)
            "full_audio": Path("audio_full.wav"),      # Complete audio
            "speech_ratio": 0.42,                        # % of video that's speech
            "segments": [...]                            # Speech segment timestamps
        }
    """
    video_source = content_sources.video_path
    ffmpeg_path = ffmpeg_binary or shutil.which("ffmpeg")
    if not video_source or not ffmpeg_path:
        return None

    # Use pre-generated signed URL if provided, otherwise generate one
    if video_url:
        # Reuse pre-generated signed URL (avoids duplicate GCS API calls)
        video_source = video_url
    elif not video_source.startswith(("http://", "https://", "gs://")):
        # Local file path - validate existence
        video_path_obj = Path(video_source)
        if not video_path_obj.exists():
            return None
        # video_source is already correct (local path)
    else:
        # GCS URL - generate signed URL for private access (fallback if not pre-generated)
        video_source = generate_signed_url(video_source)

    if output_dir:
        target_dir = Path(output_dir).resolve()
    else:
        # Default to local structured storage: thumbnail-alchemist-media/projects/{project_id}/signals/audio
        target_dir = Path(LOCAL_MEDIA_DIR).resolve() / "projects" / project_id / "signals" / "audio"

    target_dir.mkdir(parents=True, exist_ok=True)
    full_audio_path = target_dir / "audio_full.wav"
    speech_path = target_dir / "audio_speech.wav"

    # Ensure paths are absolute for ffmpeg
    full_audio_path = full_audio_path.resolve()
    speech_path = speech_path.resolve()

    # Verify directory exists and is writable
    if not target_dir.exists():
        logger.error("Target directory does not exist: %s", target_dir)
        return None
    if not os.access(target_dir, os.W_OK):
        logger.error("Target directory is not writable: %s", target_dir)
        return None

    logger.info("Extracting full audio to: %s", full_audio_path)

    # STEP 1: Extract full audio using ffmpeg streaming
    # Note: Duration validation should be done BEFORE calling this function
    # The -t flag is kept as a safety fallback, but videos exceeding max_duration
    # should be rejected earlier in the pipeline
    # -vn removes video stream (audio only)
    # -af loudnorm normalizes audio volume for consistent analysis
    # -acodec pcm_s16le for WAV format (16-bit PCM)
    # -ac 1 converts to mono (reduces file size, sufficient for analysis)
    # -ar sets sample rate (standard for VAD)

    # Build ffmpeg command (keep -t as safety fallback, but validation should happen earlier)
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_source,  # Signed URL for GCS, or direct path/URL
        "-t",
        str(max_duration_seconds),  # Safety fallback - should not be needed if validation works
        "-vn",  # No video
        "-af",
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}",  # EBU R128 loudness normalization
        "-acodec",
        "pcm_s16le",  # WAV codec (16-bit PCM)
        "-ac",
        "1",  # Mono
        "-ar",
        str(DEFAULT_SAMPLE_RATE),  # Sample rate (required for Silero VAD)
        str(full_audio_path),
        "-y",  # Overwrite output file
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        logger.error("ffmpeg failed with return code %d: %s", process.returncode, error_msg)
        return None

    # Check if file exists (with small retry for filesystem flush)
    max_retries = 3
    for attempt in range(max_retries):
        if full_audio_path.exists():
            break
        if attempt < max_retries - 1:
            await asyncio.sleep(0.1)  # Small delay for filesystem flush
        else:
            # Final check - log detailed error
            logger.error(
                "ffmpeg succeeded (returncode=0) but output file not found at %s (absolute: %s)",
                full_audio_path,
                full_audio_path.resolve(),
            )
            if stderr:
                logger.error("ffmpeg stderr: %s", stderr.decode())
            # Check if file exists in current directory (common ffmpeg issue)
            cwd_file = Path.cwd() / full_audio_path.name
            if cwd_file.exists():
                logger.error("Found file in current directory instead: %s", cwd_file)
            return None

    # STEP 2: Detect speech segments and create speech-only lane
    speech_result = detect_speech_in_audio(
        audio_path=full_audio_path,
        output_speech_path=speech_path,
        filter_singing=True,  # Filter out singing vocals
    )

    # Determine speech file path: use speech_path if speech was detected, None otherwise
    # The function returns the same path we passed in, so we can check segments instead
    speech_file_path = speech_path if speech_result.get("segments") else None

    # Log if no speech was detected (valid result, not an error)
    if not speech_file_path:
        logger.info(
            "No speech detected: speech_ratio=%.1f%%, segments=%d. "
            "No speech file created - downstream code should handle this case.",
            speech_result["speech_ratio"] * 100,
            len(speech_result["segments"]),
        )

    return {
        "speech": speech_file_path,  # None if no speech detected
        "full_audio": full_audio_path,
        "speech_ratio": speech_result["speech_ratio"],
        "segments": speech_result["segments"],
        "total_duration": speech_result["total_duration"],
        "speech_duration": speech_result["speech_duration"],
    }


async def analyze_audio_features(audio_path: Path) -> list[dict[str, Any]]:
    """
    Extract audio features for prosody and music analysis.

    Args:
        audio_path: Path to audio file

    Returns:
        List of audio segments with start/end and features (same format as speech segments):
        [
            {
                "start": float,
                "end": float,
                "pitch": float,
                "pitch_variance": float,
                "energy": float,
                "zero_crossing_rate": float,
                "spectral_brightness": float,
                "spectral_rolloff": float,
            },
            ...
        ]
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)

    # Frame parameters (analyze in small windows)
    frame_length = int(AUDIO_FRAME_LENGTH_SECONDS * sr)
    hop_length = int(AUDIO_HOP_LENGTH_SECONDS * sr)

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

    # Create timeline from analysis windows - format as list of segments (same as speech segments)
    segment_starts = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    segment_duration = frame_length / sr  # Duration of each analysis window in seconds
    segment_ends = segment_starts + segment_duration

    # Calculate pitch variance (in windows)
    pitch_variance = []
    for i in range(len(pitch_values)):
        start = max(0, i - PITCH_VARIANCE_WINDOW_SIZE // 2)
        end = min(len(pitch_values), i + PITCH_VARIANCE_WINDOW_SIZE // 2)
        window_pitches = [p for p in pitch_values[start:end] if p > 0]
        variance = float(np.var(window_pitches)) if window_pitches else 0.0
        pitch_variance.append(variance)

    # Format as list of segments (same format as speech segments)
    segments = []
    for i in range(len(rms)):
        segments.append(
            {
                "start": float(segment_starts[i]),
                "end": float(segment_ends[i]),
                "pitch": pitch_values[i],
                "pitch_variance": pitch_variance[i],
                "energy": float(rms[i]),
                "zero_crossing_rate": float(zcr[i]),
                "spectral_brightness": float(spectral_centroids[i]),
                "spectral_rolloff": float(spectral_rolloff[i]),
            }
        )

    return segments


def save_timeline_json(timeline_data: dict[str, Any], output_path: Path) -> None:
    """
    Save timeline data to JSON file.

    Args:
        timeline_data: Timeline dictionary from transcribe_speech_audio
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


async def transcribe_speech_audio(
    audio_path: Path,
    project_id: str,
    language: str | None = None,
    save_timeline: bool = True,
) -> dict[str, Any]:
    """
    Transcribe speech audio with speaker diarization.

    This function handles transcription and diarization of speech-only audio

    Note: Function is async because of retry logic with asyncio.sleep().
    The OpenAI API call itself is synchronous but wrapped in async for retry handling.

    Args:
        audio_path: Path to audio file (WAV format, 16kHz mono recommended)
        project_id: Project identifier for organizing output files
        language: Language code for transcription (e.g., "en", "es", "fr").
                  If None, OpenAI will auto-detect the language (default: None)
        save_timeline: If True, save transcript to JSON file (default: True)

    Returns:
        Dictionary containing:
            - transcript: Full text transcript
            - speakers: List of detected speakers with IDs
            - segments: List of transcript segments with start, end, text, and speaker
            - duration_seconds: Total audio duration (from last segment end time)
            - timeline_path: Path to saved transcript JSON (if save_timeline=True)

    Raises:
        MediaValidationError: If transcription fails
    """
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=TRANSCRIPTION_TIMEOUT,
        )

        # STEP 1: GPT-4o Transcribe Diarize - Transcription + Speaker Diarization
        # Retry logic for network issues
        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_TRANSCRIPTION_RETRIES):
            try:
                with open(audio_path, "rb") as audio_file:
                    # Build transcription params - only include language if specified
                    transcription_params = {
                        "model": "gpt-4o-transcribe-diarize",
                        "file": audio_file,
                        "response_format": "diarized_json",
                        "chunking_strategy": "auto",  # Required for audio > 30 seconds
                    }
                    # Only add language parameter if specified (None = auto-detect)
                    if language is not None:
                        transcription_params["language"] = language

                    transcription = client.audio.transcriptions.create(**transcription_params)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < MAX_TRANSCRIPTION_RETRIES - 1:
                    logger.warning(
                        "Transcription attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        e,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise  # Re-raise on final attempt

        # Extract speakers from transcription
        # The diarized_json format includes speaker info directly in segments
        # Note: OpenAI returns segment.start and segment.end in seconds (float)
        speakers = set()
        segments = []
        duration = 0.0

        if hasattr(transcription, "segments") and transcription.segments:
            for segment in transcription.segments:
                if hasattr(segment, "speaker") and segment.speaker:
                    speakers.add(segment.speaker)

                segments.append(
                    {
                        "start": float(segment.start),  # seconds (float from OpenAI)
                        "end": float(segment.end),  # seconds (float from OpenAI)
                        "text": segment.text,
                        "speaker": getattr(segment, "speaker", None),
                    }
                )

                # Track duration from last segment
                if segment.end > duration:
                    duration = segment.end

        result = {
            "transcript": transcription.text,
            "speakers": [{"id": spk, "label": spk} for spk in sorted(speakers)],
            "segments": segments,
            "duration_seconds": duration,
        }

        # Save speech audio transcript to JSON if requested
        if save_timeline:
            # Upload directly to GCS from memory
            gcs_url = upload_json_to_gcs(
                data=result,
                project_id=project_id,
                directory="signals/audio",
                filename="speech_audio_transcript.json",
                bucket_name=PROJECT_ASSETS_BUCKET,
            )
            if gcs_url:
                result["timeline_path"] = gcs_url

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
        audio_features: List of audio segments from analyze_audio_features

    Returns:
        Dictionary with average features for the segment
    """
    # Find all segments that overlap with this segment
    matching_segments = [
        seg
        for seg in audio_features
        if not (seg["end"] <= start or seg["start"] >= end)  # Overlap check
    ]

    if not matching_segments:
        return {"avg_energy": 0.0, "avg_pitch": 0.0}

    # Calculate averages
    energies = [seg["energy"] for seg in matching_segments]
    pitches = [seg["pitch"] for seg in matching_segments if seg["pitch"] > 0]

    return {
        "avg_energy": float(sum(energies) / len(energies)) if energies else 0.0,
        "avg_pitch": float(sum(pitches) / len(pitches)) if pitches else 0.0,
    }


def _detect_energy_peaks(
    audio_features: dict,
    threshold_percentile: float = ENERGY_PEAK_PERCENTILE,
) -> list[dict]:
    """
    Detect energy peaks (high-energy moments) in audio.

    Args:
        audio_features: List of audio segments from analyze_audio_features
        threshold_percentile: Energy percentile to consider as peak

    Returns:
        List of energy peak events
    """
    segments = audio_features
    energies = np.array([seg["energy"] for seg in segments])

    # Calculate threshold
    threshold = np.percentile(energies, threshold_percentile)

    # Find peaks (local maxima above threshold)
    peaks = []

    for i in range(ENERGY_PEAK_WINDOW_SIZE, len(segments) - ENERGY_PEAK_WINDOW_SIZE):
        if energies[i] > threshold:
            # Check if it's a local maximum
            if energies[i] == max(
                energies[i - ENERGY_PEAK_WINDOW_SIZE : i + ENERGY_PEAK_WINDOW_SIZE]
            ):
                seg = segments[i]
                # Avoid duplicate peaks too close together
                if not peaks or (seg["start"] - peaks[-1]["time"]) > ENERGY_PEAK_MIN_SPACING:
                    peaks.append(
                        {
                            "time": seg["start"],
                            "energy": float(energies[i]),
                        }
                    )

    return peaks


def _detect_music_sections(
    audio_features: dict,
    energy_threshold: float = MUSIC_ENERGY_THRESHOLD,
) -> list[dict]:
    """
    Detect sections with background music.

    Uses spectral brightness and harmonic content to identify music.

    Args:
        audio_features: List of audio segments from analyze_audio_features
        energy_threshold: Minimum energy to consider

    Returns:
        List of music section events
    """
    segments = audio_features
    energies = np.array([seg["energy"] for seg in segments])
    brightness = np.array([seg["spectral_brightness"] for seg in segments])
    zcr = np.array([seg["zero_crossing_rate"] for seg in segments])

    # Music typically has: high brightness, low ZCR (harmonic), sustained energy
    harmonic_ratio = 1.0 - zcr

    # Combine indicators
    music_score = (brightness / brightness.max()) * harmonic_ratio * (energies > energy_threshold)

    # Find continuous sections with high music score
    sections = []
    in_section = False
    section_start = 0.0

    for seg, score in zip(segments, music_score):
        if score > MUSIC_SCORE_THRESHOLD and not in_section:
            section_start = seg["start"]
            in_section = True
        elif score <= MUSIC_SCORE_THRESHOLD and in_section:
            if seg["start"] - section_start > MUSIC_MIN_DURATION:
                # Calculate average intensity for this section
                section_segments = [
                    s for s in segments if section_start <= s["start"] <= seg["start"]
                ]
                avg_intensity = float(np.mean([s["energy"] for s in section_segments]))

                sections.append(
                    {
                        "start": section_start,
                        "end": seg["start"],
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
    speech_confidence_threshold: float = SPEECH_CONFIDENCE_THRESHOLD,
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
        return TEXT_IMPORTANCE_FILLER  # Empty text = filler

    text_lower = text.lower()

    # Use default keyword lists if not provided
    if claim_keywords is None:
        claim_keywords = DEFAULT_CLAIM_KEYWORDS

    if emphasis_keywords is None:
        emphasis_keywords = DEFAULT_EMPHASIS_KEYWORDS

    if filler_keywords is None:
        filler_keywords = DEFAULT_FILLER_KEYWORDS

    # Check for strong claims/reveals (highest priority)
    has_claim = any(kw in text_lower for kw in claim_keywords)
    has_reveal_markers = any(marker in text for marker in ["!", "?", "...", "—"])

    if has_claim or (has_reveal_markers and len(text.split()) > TEXT_IMPORTANCE_MIN_CLAIM_WORDS):
        return TEXT_IMPORTANCE_CLAIM

    # Check for emphasis/contrast
    has_emphasis = any(kw in text_lower for kw in emphasis_keywords)
    has_caps = any(word.isupper() and len(word) > 2 for word in text.split())

    if has_emphasis or has_caps:
        return TEXT_IMPORTANCE_EMPHASIS

    # Check for filler
    has_filler = any(kw in text_lower for kw in filler_keywords)
    is_short = len(text.split()) < 5

    if has_filler or is_short:
        return TEXT_IMPORTANCE_FILLER

    # Default: neutral explanation
    return TEXT_IMPORTANCE_NEUTRAL


def calculate_emphasis_score(
    segment: dict,
    audio_features: dict,
    window_seconds: float = EMPHASIS_WINDOW_SECONDS,
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
        audio_features: List of audio segments from analyze_audio_features
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
    segments = audio_features

    # Find center of segment
    segment_center = (start_s + end_s) / 2.0

    # Get window around segment for baseline calculation
    window_start = segment_center - window_seconds / 2.0
    window_end = segment_center + window_seconds / 2.0

    # Get energies in window
    window_segments = [
        seg for seg in segments if not (seg["end"] <= window_start or seg["start"] >= window_end)
    ]
    window_energies = np.array([seg["energy"] for seg in window_segments])
    all_energies = np.array([seg["energy"] for seg in segments])

    if len(window_energies) < 2:
        # Not enough data for baseline, use global baseline
        rolling_baseline = float(np.mean(all_energies))
        rolling_std = float(np.std(all_energies))
    else:
        rolling_baseline = float(np.mean(window_energies))
        rolling_std = float(np.std(window_energies))

    # Avoid division by zero
    if rolling_std < EMPHASIS_MIN_STD:
        rolling_std = EMPHASIS_MIN_STD

    # Calculate normalized deviation
    emphasis = (local_energy - rolling_baseline) / rolling_std

    # Clamp to [0, 1]
    return max(0.0, min(emphasis, 1.0))


def calculate_bgm_penalty(
    segment: dict,
    audio_features: list[dict[str, Any]],
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
    segments = audio_features
    segment_segments = [
        seg for seg in segments if not (seg["end"] <= start_s or seg["start"] >= end_s)
    ]

    if not segment_segments:
        return 1.0  # No data, assume no penalty

    energies = [seg["energy"] for seg in segment_segments]

    if len(energies) < 2:
        return 1.0  # Not enough data for variance

    # Calculate energy variance
    energy_variance = float(np.var(energies))

    # Normalize variance to [0, 1] using typical ranges
    # Typical speech: variance ~ 0.001-0.01
    # Music: variance ~ 0.01-0.1+
    energy_variance_normalized = min(energy_variance / BGM_MAX_VARIANCE, 1.0)

    # Calculate penalty
    penalty = 1.0 - energy_variance_normalized

    # Clamp to [BGM_MIN_PENALTY, BGM_MAX_PENALTY]
    return max(BGM_MIN_PENALTY, min(penalty, BGM_MAX_PENALTY))


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
        audio_features: List of audio segments from analyze_audio_features
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
