"""Audio and video processing functions.

This module contains reusable processing functions for audio analysis,
video analysis, and other processing tasks that can be used across
different endpoints and orchestration functions.
"""

import asyncio
import logging
import time
from typing import Any

from app.audio.extraction import (
    analyze_audio_features,
    extract_audio_from_video,
    transcribe_speech_audio,
)
from app.audio.saliency import detect_audio_saliency
from app.audio.speech_semantics import analyze_speech
from app.constants import DEFAULT_MAX_DURATION_SECONDS
from app.models import SourceMedia

logger = logging.getLogger(__name__)


async def process_audio_analysis(
    video_path: str,
    project_id: str,
    video_url: str | None = None,
) -> dict[str, Any]:
    """
    Process complete audio analysis pipeline: extraction, transcription, and Stream A/B analysis.

    This function handles:
    1. Audio extraction from video
    2. Speech transcription and diarization
    3. Stream A: Speech semantics analysis (tone, emotion, narrative context)
    4. Stream B: Audio saliency detection (energy peaks, spectral changes, etc.)

    Args:
        video_path: Path to video file (GCS URL or local path)
        project_id: Project identifier for organizing outputs
        video_url: Optional pre-generated signed URL (if None, will be generated)

    Returns:
        Dictionary containing:
            - audio_result: Audio extraction results (speech, full_audio paths, segments)
            - transcription_result: Transcription results (transcript, segments, speakers)
            - stream_a_results: Speech semantics analysis (list of segments with tone, emotion, narrative)
            - stream_b_results: Audio saliency analysis (list of segments with saliency features)
            - audio_features: Raw audio features (list of segments with pitch, energy, etc.)
            - stats: Timing statistics:
                - audio_extraction_time: Time for audio extraction
                - transcription_time: Time for transcription
                - audio_breakdown_time: Total time (extraction + transcription)
                - audio_analysis_time: Time for Stream A + B analysis
    """
    stats = {
        "audio_extraction_time": 0.0,
        "transcription_time": 0.0,
        "audio_breakdown_time": 0.0,
        "audio_analysis_time": 0.0,
    }

    # ========================================================================
    # STAGE 1: Audio Breakdown
    # ========================================================================
    logger.info("Step 1/5: Audio Extraction...")
    step_start = time.time()

    source_media = SourceMedia(video_path=video_path)
    audio_result = await extract_audio_from_video(
        content_sources=source_media,
        project_id=project_id,
        max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS,
        video_url=video_url,
    )

    if not audio_result:
        raise ValueError("Failed to extract audio from video")

    stats["audio_extraction_time"] = time.time() - step_start
    logger.info("Audio extraction complete in %.2fs", stats["audio_extraction_time"])

    # Use speech lane for transcription (better accuracy)
    speech_path = audio_result.get("speech")  # May be None if no speech detected

    logger.info("Step 2/5: Speech Audio Transcription and Diarization...")
    transcription_start = time.time()
    # Only transcribe if speech was detected
    transcription_result = None
    if speech_path:
        transcription_result = await transcribe_speech_audio(
            audio_path=speech_path,  # Use speech-only lane for transcription
            project_id=project_id,
            language=None,  # Auto-detect language
            save_timeline=True,
        )
        stats["transcription_time"] = time.time() - transcription_start
        logger.info(
            "Transcription complete: %d segments, %d speakers (%.2fs)",
            len(transcription_result.get("segments", [])),
            len(transcription_result.get("speakers", [])),
            stats["transcription_time"],
        )
    else:
        stats["transcription_time"] = time.time() - transcription_start
        logger.info("No speech detected - skipping transcription")

    # Combined audio breakdown time (extraction + transcription)
    stats["audio_breakdown_time"] = stats["audio_extraction_time"] + stats["transcription_time"]

    # ========================================================================
    # STAGE 2: Stream A + B: Audio Analysis (run in parallel)
    # ========================================================================
    logger.info("Step 3/5: Running Stream A (Speech Semantics) + Stream B (Audio Saliency)...")

    audio_analysis_start = time.time()

    # Store audio_features outside async scope for later use
    audio_features: list[dict[str, Any]] | None = None

    # Define async tasks for parallel execution
    async def run_stream_a():
        """Stream A: Speech Semantics Analysis"""
        if transcription_result and transcription_result.get("transcript") and speech_path:
            try:
                # Use transcript_segments for both - they're the same segments now
                transcript_segs = transcription_result.get("segments", [])

                speech_segments = await analyze_speech(
                    audio_path=speech_path,
                    segments=transcript_segs,  # Same segments used for both tone detection and narrative analysis
                )
                return speech_segments
            except Exception as e:
                logger.error("Stream A failed: %s", e, exc_info=True)
                return None
        else:
            logger.info("Stream A skipped: No speech detected or transcription failed")
            return None

    async def run_stream_b():
        """Stream B: Audio Saliency Detection"""
        nonlocal audio_features
        full_audio_path = audio_result.get("full_audio")
        if full_audio_path:
            try:
                audio_features = await analyze_audio_features(full_audio_path)
                audio_saliency_results = detect_audio_saliency(
                    audio_features=audio_features,
                    speech_segments=audio_result.get("segments"),
                )
                return audio_saliency_results
            except Exception as e:
                logger.error("Stream B failed: %s", e, exc_info=True)
                return None
        else:
            logger.info("Stream B skipped: No full audio available")
            return None

    # Run Stream A and Stream B in parallel
    stream_a_results, stream_b_results = await asyncio.gather(
        run_stream_a(),
        run_stream_b(),
    )

    stats["audio_analysis_time"] = time.time() - audio_analysis_start

    return {
        "audio_result": audio_result,
        "transcription_result": transcription_result,
        "stream_a_results": stream_a_results,
        "stream_b_results": stream_b_results,
        "audio_features": audio_features,
        "stats": stats,
    }
