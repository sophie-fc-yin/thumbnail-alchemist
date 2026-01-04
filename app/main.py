import logging
import os
import re
import time
import warnings
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analysis.adaptive_sampling import cleanup_local_frames, orchestrate_adaptive_sampling
from app.analysis.processing import (
    identify_important_moments,
    process_audio_analysis,
    process_initial_vision_analysis,
)
from app.models.thumbnail import ClickMomentPhase1, Phase1MomentInsight, Phase1Pillars
from app.thumbnail_agent import ThumbnailSelector
from app.vision.extraction import generate_signed_url

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Set root logger to WARNING to avoid noise from third-party libraries
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Enable DEBUG logging for our app modules only
app_log_level = getattr(logging, LOG_LEVEL, logging.DEBUG)
for logger_name in ["app", "uvicorn.access"]:
    logging.getLogger(logger_name).setLevel(app_log_level)

# Suppress NNPACK warnings from PyTorch (harmless, just noise)
warnings.filterwarnings("ignore", message=".*NNPACK.*")
logging.getLogger("torch").setLevel(logging.ERROR)

from app.constants import DEFAULT_MAX_DURATION_SECONDS, GCS_ASSETS_BUCKET  # noqa: E402
from app.models import (  # noqa: E402
    AudioBreakdownRequest,
    AudioBreakdownResponse,
    FrameWithFeatures,
    ProcessImportantMomentsRequest,
    ProcessImportantMomentsResponse,
    SignedUrlRequest,
    SignedUrlResponse,
    ThumbnailRequest,
    ThumbnailResponse,
    VideoUploadError,
    VideoUploadResponse,
    VideoUrlRequest,
    VideoUrlResponse,
    VisionBreakdownRequest,
    VisionBreakdownResponse,
)
from app.utils.storage import (  # noqa: E402
    StorageError,
    generate_signed_download_url,
    generate_signed_upload_url,
    upload_audio_file_to_gcs,
    upload_file_to_gcs,
    upload_json_to_gcs,
    upload_project_file_to_gcs,
)
from app.vision.extraction import (  # noqa: E402
    MediaValidationError,
    VideoDurationExceededError,
    get_video_duration,
    validate_video_duration,
)

logger = logging.getLogger(__name__)
logger.info("Logging configured - App: %s, Root: WARNING", LOG_LEVEL)

app = FastAPI(title="Thumbnail Alchemist API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://clickmoment.vercel.app",  # Your Vercel production domain
        "http://localhost:3000",  # For local frontend development
        "http://localhost:8000",  # For local backend development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Video Upload Configuration
MAX_VIDEO_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
ALLOWED_VIDEO_MIMETYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "video/x-flv",
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.

    - Remove directory traversal attempts (../)
    - Remove special characters except alphanumeric, dots, hyphens, underscores
    - Preserve file extension

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for storage
    """
    # Remove directory components
    filename = Path(filename).name

    # Split name and extension
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()

    # Remove unsafe characters from stem (keep alphanumeric, dash, underscore)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)

    # Limit length to 100 chars
    safe_stem = safe_stem[:100]

    return f"{safe_stem}{suffix}"


@app.get("/")
async def root():
    """Root endpoint - health check for Cloud Run."""
    return {
        "service": "Thumbnail Alchemist API",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


@app.post("/audio/breakdown", response_model=AudioBreakdownResponse)
async def breakdown_audio(payload: AudioBreakdownRequest) -> AudioBreakdownResponse:
    """
    Break down audio from video into components.

    Returns transcript, speaker diarization, prosody features, and timeline.
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    # Generate signed URL once (reuse for validation and processing)
    video_url = None
    if payload.video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(payload.video_path)

    # Validate video duration before processing
    try:
        await validate_video_duration(
            payload.video_path,
            max_duration_seconds=payload.max_duration_seconds,
            video_url=video_url,  # Reuse signed URL
        )
    except VideoDurationExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video duration exceeds limit",
                "detail": str(e),
                "max_duration_seconds": payload.max_duration_seconds,
            },
        ) from e
    except MediaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video validation failed",
                "detail": str(e),
            },
        ) from e

    # Process complete audio analysis pipeline
    try:
        audio_analysis_result = await process_audio_analysis(
            video_path=payload.video_path,
            project_id=project_id,
            video_url=video_url,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Audio analysis failed",
                "detail": str(e),
            },
        ) from e

    # Extract results
    audio_result = audio_analysis_result["audio_result"]
    transcription_result = audio_analysis_result["transcription_result"]
    stream_a_results = audio_analysis_result["stream_a_results"]
    stream_b_results = audio_analysis_result["stream_b_results"]
    audio_features = audio_analysis_result["audio_features"]

    # Upload audio files to GCS (extract_audio_from_video returns local paths)
    gcs_speech_url = ""
    speech_path = audio_result.get("speech")
    if speech_path:
        gcs_speech_url = (
            upload_audio_file_to_gcs(
                file_path=speech_path,
                project_id=project_id,
                directory="signals/audio",
                filename="audio_speech.wav",
                bucket_name="clickmoment-prod-assets",
                cleanup_local=True,
            )
            or ""
        )

    full_audio_path = audio_result.get("full_audio")
    if full_audio_path:
        upload_audio_file_to_gcs(
            file_path=full_audio_path,
            project_id=project_id,
            directory="signals/audio",
            filename="audio_full.wav",
            bucket_name="clickmoment-prod-assets",
            cleanup_local=True,
        )

    # Upload Stream A and Stream B results as JSON
    if stream_a_results:
        upload_json_to_gcs(
            data=stream_a_results,
            project_id=project_id,
            directory="signals/audio",
            filename="stream_a_results.json",
            bucket_name="clickmoment-prod-assets",
        )
        logger.info("Uploaded Stream A results to GCS")

    if stream_b_results:
        upload_json_to_gcs(
            data=stream_b_results,
            project_id=project_id,
            directory="signals/audio",
            filename="stream_b_results.json",
            bucket_name="clickmoment-prod-assets",
        )
        logger.info("Uploaded Stream B results to GCS")

    # Handle case where no speech was detected
    if not transcription_result:
        return AudioBreakdownResponse(
            project_id=project_id,
            audio_path=gcs_speech_url,
            transcript="",
            duration_seconds=0.0,
            stream_a_results=stream_a_results,
            stream_b_results=stream_b_results,
            audio_features=audio_features,
        )

    return AudioBreakdownResponse(
        project_id=project_id,
        audio_path=gcs_speech_url,
        transcript=transcription_result.get("transcript", ""),
        duration_seconds=transcription_result.get("duration_seconds", 0.0),
        stream_a_results=stream_a_results,
        stream_b_results=stream_b_results,
        audio_features=audio_features,
    )


@app.post("/vision/breakdown", response_model=VisionBreakdownResponse)
async def breakdown_vision(payload: VisionBreakdownRequest) -> VisionBreakdownResponse:
    """
    Perform initial vision analysis: sparse frame sampling and visual change detection.

    This endpoint:
    1. Validates video duration
    2. Calculates adaptive sampling interval based on video duration
    3. Extracts sparse sample frames at adaptive intervals
    4. Analyzes visual changes (shot/layout changes and motion spikes) on sample frames

    Returns sample frames with visual change analysis results.
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    # Generate signed URL once (reuse for validation and processing)
    video_url = None
    if payload.video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(payload.video_path)

    # Validate video duration before processing
    try:
        await validate_video_duration(payload.video_path, video_url=video_url)
    except VideoDurationExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video duration exceeds limit",
                "detail": str(e),
                "max_duration_seconds": DEFAULT_MAX_DURATION_SECONDS,
            },
        ) from e
    except MediaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video validation failed",
                "detail": str(e),
            },
        ) from e

    # Get video duration for adaptive sampling interval calculation
    try:
        video_duration = await get_video_duration(payload.video_path, video_url=video_url)
    except MediaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Failed to get video duration",
                "detail": str(e),
            },
        ) from e

    # Process initial vision analysis
    try:
        vision_result = await process_initial_vision_analysis(
            video_path=payload.video_path,
            project_id=project_id,
            video_duration=video_duration,
            video_url=video_url,
        )
    except Exception as e:
        logger.error("Initial vision analysis failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Initial vision analysis failed",
                "detail": str(e),
            },
        ) from e

    visual_frames = vision_result["visual_frames"]
    motion_spikes_count = sum(1 for frame in visual_frames if frame.get("motion_spike", False))

    response = VisionBreakdownResponse(
        project_id=project_id,
        frames=visual_frames,
        sample_interval=vision_result["sample_interval"],
        stats=vision_result["stats"],
    )

    logger.info(
        "Initial vision analysis complete: %d sample frames, %d motion spikes detected",
        len(visual_frames),
        motion_spikes_count,
    )

    return response


@app.post("/vision/important-moments", response_model=ProcessImportantMomentsResponse)
async def process_important_moments(
    payload: ProcessImportantMomentsRequest,
) -> ProcessImportantMomentsResponse:
    """
    Process important moments: identify, extract frames, and compute vision features.

    This endpoint performs three steps:
    1. Identify important moments by combining Stream A (speech semantics), Stream B (audio saliency), and visual signals
    2. Extract dense frames in parallel for the identified important moments
    3. Compute comprehensive vision features for each extracted frame (face analysis, aesthetics, editability, composition, technical quality)

    **Prerequisites**: This endpoint requires pre-computed audio and vision analysis results:
    - Stream A results: `/audio/breakdown` endpoint must be called first, or results must exist in GCS at `projects/{project_id}/signals/audio/stream_a_results.json`
    - Stream B results: `/audio/breakdown` endpoint must be called first, or results must exist in GCS at `projects/{project_id}/signals/audio/stream_b_results.json`
    - Visual frames: `/vision/breakdown` endpoint must be called first, or results must exist in GCS at `projects/{project_id}/signals/vision/initial_sample_visual_frames.json`

    If results are not provided in the request and not found in GCS, the endpoint will return empty results.

    Args:
        payload: Request containing video_path, optional project_id, and optional pre-computed results

    Returns:
        ProcessImportantMomentsResponse with importance segments, extracted frames, and all vision features
    """
    start_time = time.time()
    stats = {
        "importance_calculation_time": 0.0,
        "extraction_time": 0.0,
        "vision_analysis_time": 0.0,
        "total_time": 0.0,
    }

    # Use provided project_id or generate new one
    # Treat placeholder values as None
    if payload.project_id and payload.project_id.lower() not in ["string", "none", "null", ""]:
        project_id = payload.project_id
    else:
        project_id = str(uuid4())
        if payload.project_id:
            logger.warning(
                "project_id '%s' appears to be a placeholder, generated new UUID: %s",
                payload.project_id,
                project_id,
            )

    # Generate signed URL if needed
    video_url = None
    if payload.video_path.startswith(("gs://", "http://", "https://")):
        video_url = generate_signed_url(payload.video_path)

    # Get video duration
    try:
        video_duration = await get_video_duration(payload.video_path, video_url=video_url)
    except MediaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Failed to get video duration",
                "detail": str(e),
            },
        ) from e

    # Call identify_important_moments which now handles everything:
    # 1. Identifies important moments
    # 2. Extracts dense frames
    # 3. Computes vision features

    # Log input data status and structure
    logger.info(
        "Processing important moments: video_duration=%.2fs, stream_a=%s, stream_b=%s, visual_frames=%s, project_id=%s",
        video_duration,
        "provided" if payload.stream_a_results else "will load from GCS",
        "provided" if payload.stream_b_results else "will load from GCS",
        "provided" if payload.visual_frames else "will load from GCS",
        project_id,
    )

    # Debug: Log actual payload structure
    if payload.stream_a_results:
        logger.info(
            "DEBUG payload.stream_a_results: type=%s, length=%d",
            type(payload.stream_a_results).__name__,
            len(payload.stream_a_results)
            if isinstance(payload.stream_a_results, (list, dict))
            else 0,
        )
        if isinstance(payload.stream_a_results, list) and len(payload.stream_a_results) > 0:
            first = payload.stream_a_results[0]
            logger.info(
                "  First item type: %s, keys: %s",
                type(first).__name__,
                list(first.keys()) if isinstance(first, dict) else "not a dict",
            )
            if isinstance(first, dict):
                logger.info(
                    "  First item sample: start=%s, end=%s, importance=%s",
                    first.get("start"),
                    first.get("end"),
                    first.get("importance"),
                )

    if payload.stream_b_results:
        logger.info(
            "DEBUG payload.stream_b_results: type=%s, length=%d",
            type(payload.stream_b_results).__name__,
            len(payload.stream_b_results)
            if isinstance(payload.stream_b_results, (list, dict))
            else 0,
        )
        if isinstance(payload.stream_b_results, list) and len(payload.stream_b_results) > 0:
            first = payload.stream_b_results[0]
            logger.info(
                "  First item type: %s, keys: %s",
                type(first).__name__,
                list(first.keys()) if isinstance(first, dict) else "not a dict",
            )

    if payload.visual_frames:
        logger.info(
            "DEBUG payload.visual_frames: type=%s, length=%d",
            type(payload.visual_frames).__name__,
            len(payload.visual_frames) if isinstance(payload.visual_frames, (list, dict)) else 0,
        )

    try:
        frames = await identify_important_moments(
            video_duration=video_duration,
            stream_a_results=payload.stream_a_results,
            stream_b_results=payload.stream_b_results,
            visual_frames=payload.visual_frames,
            project_id=project_id,
            video_path=payload.video_path,
            video_url=video_url,
            niche=payload.niche,
        )
    except Exception as e:
        logger.error("Processing important moments failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Processing important moments failed",
                "detail": str(e),
            },
        ) from e

    stats["total_time"] = time.time() - start_time

    logger.info(
        "Processed %d frames extracted and analyzed (%.2fs)",
        len(frames),
        stats["total_time"],
    )

    # Check if we have any frames
    if not frames:
        logger.warning(
            "No frames extracted. This usually means no signals were found. "
            "Ensure that audio analysis (Stream A/B) and vision analysis results are available "
            "either in the request or in GCS at projects/%s/signals/",
            project_id,
        )
        # Return with diagnostic info
        diagnostic = {
            "message": "No frames extracted. This usually means no signals (Stream A, Stream B, or visual frames) were available.",
            "signals_loaded": {
                "stream_a_provided": payload.stream_a_results is not None,
                "stream_b_provided": payload.stream_b_results is not None,
                "visual_frames_provided": payload.visual_frames is not None,
                "project_id": project_id,
                "gcs_paths_checked": [
                    f"projects/{project_id}/signals/audio/stream_a_results.json",
                    f"projects/{project_id}/signals/audio/stream_b_results.json",
                    f"projects/{project_id}/signals/vision/initial_sample_visual_frames.json",
                ],
            },
            "recommendation": "Call /audio/breakdown and /vision/breakdown endpoints first to generate the required analysis results, or provide them in the request.",
        }
        return ProcessImportantMomentsResponse(
            project_id=project_id,
            importance_segments=[],
            frames=[],
            stats=stats,
            diagnostic=diagnostic,
        )

    # Convert frame dictionaries to FrameWithFeatures objects
    all_frames = []
    for frame_dict in frames:
        frame_with_features = FrameWithFeatures(
            local_path=frame_dict.get("local_path", ""),
            gcs_url=frame_dict.get("gcs_url", ""),
            time=frame_dict.get("time", 0.0),
            segment_index=frame_dict.get("segment_index", 0),
            importance_level=frame_dict.get("importance_level", "low"),
            importance_score=frame_dict.get("importance_score", 0.0),
            segment_start=frame_dict.get("segment_start", 0.0),
            segment_end=frame_dict.get("segment_end", 0.0),
            face_analysis=frame_dict.get("face_analysis", {}),
            visual_analysis={
                "aesthetics": frame_dict.get("aesthetics", {}),
                "editability": frame_dict.get("editability", {}),
                "composition": frame_dict.get("composition", {}),
                "technical_quality": frame_dict.get("technical_quality", {}),
                "face_quality": frame_dict.get("face_quality", {}),
            },
        )
        all_frames.append(frame_with_features)

    # Upload important moments analysis to GCS (convert Pydantic objects to dicts)
    frames_data = [frame.model_dump() for frame in all_frames]
    upload_json_to_gcs(
        data=frames_data,
        project_id=project_id,
        directory="analysis",
        filename="important_moments_analysis.json",
    )
    logger.info("Uploaded important moments analysis to GCS: %d frames", len(all_frames))

    return ProcessImportantMomentsResponse(
        project_id=project_id,
        importance_segments=[],  # Not returned by identify_important_moments anymore
        frames=all_frames,
        stats=stats,
        diagnostic=None,  # No diagnostic needed if we have results
    )


@app.post("/thumbnails/generate", response_model=ThumbnailResponse)
async def generate_thumbnail(payload: ThumbnailRequest) -> ThumbnailResponse:
    """
    Generate thumbnail using adaptive sampling + AI selection agent.

    Pipeline:
    1. Adaptive sampling → Extract candidate frames based on moment importance
    2. Thumbnail selection agent → AI picks best frame with detailed reasoning
    3. Return selected frame + composition suggestions

    Cost: ~$0.0023 per generation (Gemini 2.5 Flash)
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    logger.info("=" * 70)
    logger.info("THUMBNAIL GENERATION - Project %s", project_id)
    logger.info("=" * 70)

    # Extract video path
    video_path = payload.content_sources.video_path
    if not video_path:
        # Image-only mode: allow callers to provide pre-extracted frames/screenshots.
        # This keeps the API usable without running the heavy video pipeline.
        # No duration validation needed for image-only mode
        image_paths = payload.content_sources.image_paths
        if image_paths:
            first = image_paths[0]
            return ThumbnailResponse(
                project_id=project_id,
                status="draft",
                title_hint=payload.creative_direction.title_hint or "Untitled",
                thumbnail_url=first,
                selected_frame_url=first,
                profile_variant_url=None,
                layers=[],
                advisory=None,
                phase1=None,
                total_frames_extracted=len(image_paths),
                analysis_json_url=None,
                cost_usd=None,
                gemini_model=None,
                summary=(
                    "Image-only request received. "
                    "Provide a video_path to enable adaptive sampling + Phase-1 moment diagnostics."
                ),
            )

    # Generate signed URL once (reuse for validation and processing)
    video_url_for_processing = None
    if video_path.startswith(("gs://", "http://", "https://")):
        from app.vision.extraction import generate_signed_url

        video_url_for_processing = generate_signed_url(video_path)

    # Validate video duration before processing
    try:
        await validate_video_duration(video_path, video_url=video_url_for_processing)
    except VideoDurationExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video duration exceeds limit",
                "detail": str(e),
                "max_duration_seconds": DEFAULT_MAX_DURATION_SECONDS,
            },
        ) from e
    except MediaValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Video validation failed",
                "detail": str(e),
            },
        ) from e

    logger.info("─" * 70)
    logger.info("STEP 1: Adaptive Sampling")
    logger.info("─" * 70)
    logger.info("Video: %s", video_path)

    # Extract frames locally for processing
    # Note: Frames are stored locally during the processing pipeline.
    # Final selected thumbnails can be uploaded separately if needed.
    adaptive_result = await orchestrate_adaptive_sampling(
        video_path=video_path,
        project_id=project_id,
        video_url=video_url_for_processing,  # Reuse signed URL
    )

    candidate_frame_urls = adaptive_result["frames"]
    logger.info("Extracted %d candidate frames", len(candidate_frame_urls))

    # Get importance segments from adaptive sampling
    segment_analysis = adaptive_result.get("segment_analysis", {})
    selected_segments = segment_analysis.get("selected_segments", [])

    # Use selected_frames from segment_analysis (best frames from important segments)
    # These frames already have full vision features and are ranked by importance
    extracted_frames = segment_analysis.get("selected_frames", [])

    if not extracted_frames:
        raise ValueError("No frames found in segment analysis")

    # Log segment selection info
    logger.info(
        "Segment Analysis: %d segments, %d frames", len(selected_segments), len(extracted_frames)
    )
    if selected_segments:
        logger.debug("Top segment time ranges:")
        for seg in selected_segments[:5]:  # Show first 5
            logger.debug(
                "  %0.1fs-%0.1fs (importance: %0.2f, score: %0.2f)",
                seg.get("start_time", 0),
                seg.get("end_time", 0),
                seg.get("avg_importance", 0),
                seg.get("segment_score", 0),
            )

    # Transform selected frames from segment_analysis to ThumbnailSelector format
    # These are the best frames (1-2 per segment) from the top importance segments (5-8 segments)
    # Each frame already has complete vision features and importance scores
    enriched_frames = []
    frames_skipped = 0
    frames_with_incomplete_analysis = 0

    for idx, frame in enumerate(extracted_frames, 1):
        timestamp = frame.get("time", 0.0)  # frames_with_features uses "time" not "timestamp"

        # Get face analysis directly from this frame (analyzed during extraction)
        face_analysis = frame.get("face_analysis", {})

        # Get both local path (for Gemini) and GCS URL (for debugging)
        local_path = frame.get("local_path", "")
        gcs_url = frame.get("gcs_url", "")  # frames_with_features uses "gcs_url" not "frame_path"

        # Skip frames without local path (needed for Gemini)
        if not local_path:
            logger.warning("Frame %d: missing local_path", idx)
            frames_skipped += 1
            continue

        # AUDIT: Verify local_path filename matches timestamp
        # Expected format: frame_XXXXms.jpg where XXXX is timestamp in milliseconds
        try:
            filename = Path(local_path).name
            if "_" in filename and "ms" in filename:
                timestamp_from_filename_ms = int(filename.split("_")[1].split("ms")[0])
                timestamp_from_filename_sec = timestamp_from_filename_ms / 1000.0
                timestamp_diff = abs(timestamp_from_filename_sec - timestamp)

                if timestamp_diff > 1.0:  # More than 1 second difference
                    logger.warning(
                        "[AUDIT] Frame %d timestamp mismatch: metadata=%0.2fs, filename=%0.2fs, diff=%0.2fs, path=%s",
                        idx,
                        timestamp,
                        timestamp_from_filename_sec,
                        timestamp_diff,
                        local_path,
                    )
        except Exception:
            # Non-critical - just for debugging
            pass

        # Use ALL vision features already computed (aesthetics, editability, composition, etc.)
        complete_visual_analysis = {
            "face_analysis": face_analysis,
            "aesthetics": frame.get("aesthetics", {}),
            "editability": frame.get("editability", {}),
            "composition": frame.get("composition", {}),
            "technical_quality": frame.get("technical_quality", {}),
            "face_quality": frame.get("face_quality", {}),
        }

        # Check if analysis is incomplete
        if not face_analysis or not face_analysis.get("has_face"):
            frames_with_incomplete_analysis += 1
            if frames_with_incomplete_analysis <= 3:  # Log first 3
                logger.debug("Frame %d: incomplete face_analysis", idx)

        # CRITICAL: Use timestamp-based frame number to maintain identity through pipeline
        # This ensures frame_number stays tied to the actual frame, not loop position
        frame_number_from_timestamp = int(timestamp * 30)  # Assume ~30fps for unique IDs

        enriched_frames.append(
            {
                "frame_number": frame_number_from_timestamp,  # Timestamp-based unique ID
                "local_path": local_path,  # Local path for Gemini (faster)
                "url": gcs_url,  # GCS URL for debugging/reference
                "timestamp": timestamp,
                "importance_score": frame.get("importance_score", 0.0),
                "importance_level": frame.get("importance_level", "unknown"),
                "visual_analysis": complete_visual_analysis,  # Complete analysis for scoring
            }
        )

    logger.info(
        "Frame enrichment: %d selected → %d enriched (skipped: %d, incomplete: %d)",
        len(extracted_frames),
        len(enriched_frames),
        frames_skipped,
        frames_with_incomplete_analysis,
    )

    extracted_frames = enriched_frames

    # ========================================================================
    # STEP 2: Map request data to thumbnail selector format
    # ========================================================================
    logger.info("─" * 70)
    logger.info("STEP 2: Preparing Thumbnail Selection Agent")
    logger.info("─" * 70)

    # Map creative direction to selector format
    selector_brief = {
        "video_title": payload.creative_direction.title_hint or "Untitled Video",
        "primary_message": payload.creative_direction.notes or "No description provided",
        "target_emotion": "curiosity",  # System determines from content
        "primary_goal": "maximize_ctr",  # System determines autonomously from content analysis
        "tone": _infer_tone_from_mood(payload.creative_direction.mood),
    }

    # Map creator context to selector format
    selector_profile = {
        "niche": payload.creator_context.niche_hint
        or "general",  # Weak signal, system infers from content
        "personality": _infer_personality(payload.creative_direction.mood),
        "visual_style": _infer_visual_style(payload.creative_direction.mood),
    }

    logger.info(
        "Creative Brief: '%s' (goal: %s, niche: %s)",
        selector_brief["video_title"],
        selector_brief["primary_goal"],
        selector_profile["niche"],
    )

    # ========================================================================
    # STEP 3: Run Thumbnail Selection Agent
    # ========================================================================
    logger.info("─" * 70)
    logger.info("STEP 3: AI Thumbnail Selection")
    logger.info("─" * 70)

    # Initialize selector (uses GEMINI_API_KEY)
    try:
        selector = ThumbnailSelector(use_pro=False)  # Use Flash by default
    except ValueError as e:
        # GEMINI_API_KEY not set - return mock response
        logger.warning("GEMINI_API_KEY not set: %s", e)
        logger.warning("Returning mock response without AI selection")

        # Cleanup local frames before returning
        cleanup_local_frames(project_id)

        return ThumbnailResponse(
            project_id=project_id,
            status="draft",
            title_hint=payload.creative_direction.title_hint or "Untitled",
            thumbnail_url=candidate_frame_urls[0]
            if candidate_frame_urls
            else "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=1200&q=80",
            selected_frame_url=candidate_frame_urls[0] if candidate_frame_urls else None,
            profile_variant_url=None,
            layers=[],
            advisory=None,
            phase1=None,
            total_frames_extracted=len(candidate_frame_urls),
            analysis_json_url=adaptive_result.get("analysis_json_url"),
            cost_usd=None,
            gemini_model=None,
            summary=(
                f"Extracted {len(candidate_frame_urls)} frames using adaptive sampling. "
                "Set GEMINI_API_KEY to enable ClickMoment Phase-1 moment diagnostics."
            ),
        )

    # Run selection agent with pre-selected frames from importance segments
    selection_result = await selector.select_best_thumbnail(
        frames=extracted_frames,
        creative_brief=selector_brief,
        channel_profile=selector_profile,
    )

    # ========================================================================
    # Save selection results to GCS
    # ========================================================================
    frames_sent_to_gemini = selection_result.get("frames_sent_to_gemini", [])

    # 1. Save the 10 frames sent to Gemini analysis
    if frames_sent_to_gemini:
        logger.info("Saving %d selected frames to GCS...", len(frames_sent_to_gemini))
        for frame in frames_sent_to_gemini:
            local_path = frame.get("local_path")
            timestamp = frame.get("timestamp", 0.0)
            if local_path and Path(local_path).exists():
                # Format: frame_00004700ms.jpg (timestamp in milliseconds)
                timestamp_ms = int(timestamp * 1000)
                filename = f"frame_{timestamp_ms:08d}ms.jpg"

                # Upload frame to selected_frames directory
                upload_project_file_to_gcs(
                    file_path=local_path,
                    project_id=project_id,
                    directory="selected_frames",
                    filename=filename,
                    bucket_name=GCS_ASSETS_BUCKET,
                    cleanup_local=False,  # Don't delete, we still need them
                )
        logger.info("Saved %d selected frames", len(frames_sent_to_gemini))

    # Extract Phase-1 diagnostic data from Gemini response
    moments = selection_result.get("moments", [])
    meta = selection_result.get("meta", {})
    debug_data = selection_result.get("debug", {})

    logger.info("ClickMoment Phase-1 Insights Generated")
    logger.info("Confidence: %s", meta.get("confidence", "unknown"))
    if moments:
        logger.info("Top moment: %s", moments[0].get("frame_id", "N/A"))

    # 2. Save Gemini's reasoning to agent_reasoning.json
    upload_json_to_gcs(
        data=selection_result,
        project_id=project_id,
        directory="analysis",
        filename="agent_reasoning.json",
        bucket_name=GCS_ASSETS_BUCKET,
    )
    logger.info("Saved agent reasoning to GCS")

    # 3. Save top frames picked by Gemini to top_frames/
    if moments:
        logger.info("Saving %d top frames picked by Gemini...", min(len(moments), 3))
        for idx, moment in enumerate(moments[:3], 1):  # Save top 3 moments
            frame_id = moment.get("frame_id", "")
            # Find the corresponding frame from frames_sent_to_gemini
            # frame_id format: "Frame 1", "Frame 2", etc.
            try:
                frame_idx = int(frame_id.split()[-1]) - 1  # Convert "Frame 1" to index 0
                if 0 <= frame_idx < len(frames_sent_to_gemini):
                    frame = frames_sent_to_gemini[frame_idx]
                    local_path = frame.get("local_path")
                    timestamp = frame.get("timestamp", 0.0)

                    if local_path and Path(local_path).exists():
                        timestamp_ms = int(timestamp * 1000)
                        filename = f"top{idx}_frame_{timestamp_ms:08d}ms.jpg"

                        # Upload frame to top_frames directory
                        upload_project_file_to_gcs(
                            file_path=local_path,
                            project_id=project_id,
                            directory="top_frames",
                            filename=filename,
                            bucket_name=GCS_ASSETS_BUCKET,
                            cleanup_local=False,
                        )
            except (ValueError, IndexError) as e:
                logger.warning("Could not save top frame %d: %s", idx, e)

        logger.info("Saved top frames picked by Gemini")

    # ========================================================================
    # STEP 4: Build Response
    # ========================================================================
    logger.info("─" * 70)
    logger.info("STEP 4: Building Response")
    logger.info("─" * 70)

    # Helper function to extract frame number, URL, and timestamp from frame_id
    def get_frame_info(frame_id: str) -> tuple[int, str, str]:
        """Extract frame number from 'Frame X' format and convert GCS URL to signed HTTP URL.

        CRITICAL: Use debug data to find the correct frame, not array indexing!
        The frame_id refers to the position in frames sent to Gemini, not the original extracted_frames array.

        Returns:
            Tuple of (frame_number, frame_url, timestamp_string)
        """
        from app.vision.extraction import generate_signed_url

        # CRITICAL: Use debug.all_frames_scored to find the correct timestamp for this frame_id
        # The frame_id number refers to the position in the TOP 10 frames sent to Gemini
        all_frames_scored = debug_data.get("all_frames_scored", [])

        # Find the frame in debug data by frame_id
        matching_debug_frame = None
        for debug_frame in all_frames_scored:
            if debug_frame.get("frame_id") == frame_id:
                matching_debug_frame = debug_frame
                break

        if matching_debug_frame:
            # Get the actual timestamp from debug data
            debug_timestamp = matching_debug_frame.get("timestamp", "0.0s")
            # Parse timestamp string (format: "XX.Xs")
            try:
                timestamp_seconds = float(debug_timestamp.replace("s", ""))
            except (ValueError, AttributeError):
                timestamp_seconds = 0.0

            # Find the frame in extracted_frames by matching timestamp (within 0.5s tolerance)
            for frame in extracted_frames:
                frame_ts = frame.get("timestamp", 0.0)
                if abs(frame_ts - timestamp_seconds) < 0.5:
                    gcs_url = frame.get("url", "")
                    frame_num = frame.get("frame_number", 1)
                    # Use the timestamp from debug data, not from extracted_frames
                    timestamp_str = f"{timestamp_seconds:.1f}s"

                    # Convert GCS URL to signed HTTP URL
                    if gcs_url.startswith("gs://"):
                        return frame_num, generate_signed_url(gcs_url), timestamp_str
                    return frame_num, gcs_url, timestamp_str

        # Fallback to first frame if no match found
        logger.warning("[FRAME LOOKUP] Could not find frame for %s, using fallback", frame_id)
        if extracted_frames:
            gcs_url = extracted_frames[0].get("url", "")
            frame_timestamp = extracted_frames[0].get("timestamp", 0.0)
            timestamp_str = f"{frame_timestamp:.1f}s"
            if gcs_url.startswith("gs://"):
                return 1, generate_signed_url(gcs_url), timestamp_str
            return 1, gcs_url, timestamp_str
        return 1, "", "0.0s"

    # Build Phase-1 response model (diagnostic, non-prescriptive)
    phase1_moments: list[Phase1MomentInsight] = []
    for m in moments[:3]:
        frame_id = m.get("frame_id", "Frame 1")
        frame_num, frame_url, frame_timestamp = get_frame_info(frame_id)
        pillars = m.get("pillars", {}) or {}

        phase1_moments.append(
            Phase1MomentInsight(
                frame_id=frame_id,
                frame_number=frame_num,
                timestamp=frame_timestamp,  # Use actual timestamp from frame data, not Gemini's response
                frame_url=frame_url,
                moment_summary=m.get("moment_summary", ""),
                viewer_feel=m.get("viewer_feel", ""),
                why_this_reads=m.get("why_this_reads", []) or [],
                optional_note=m.get("optional_note"),
                pillars=Phase1Pillars(
                    emotional_signal=pillars.get("emotional_signal", ""),
                    curiosity_gap=pillars.get("curiosity_gap", ""),
                    attention_signals=pillars.get("attention_signals", []) or [],
                    readability_speed=pillars.get("readability_speed", ""),
                ),
            )
        )

    phase1 = ClickMomentPhase1(
        moments=phase1_moments,
        meta=meta if isinstance(meta, dict) else {},
        debug=debug_data if isinstance(debug_data, dict) else {},
    )

    # Choose a default thumbnail_url from the top moment (fallback to first extracted frame)
    top_url = (
        phase1_moments[0].frame_url
        if phase1_moments
        else (candidate_frame_urls[0] if candidate_frame_urls else None)
    )

    # Build summary text (Phase 1: observational)
    summary_lines = [
        "✅ ClickMoment Phase-1 moments surfaced",
        f"🧭 Selection note: {meta.get('selection_note', 'n/a')}",
        f"🎯 Total Frames Analyzed: {len(extracted_frames)}",
    ]
    for mm in phase1_moments:
        summary_lines.append(f"- {mm.frame_id}: {mm.moment_summary}")
    summary_text = "\n".join(summary_lines)

    # ========================================================================
    # CLEANUP: Delete local frames after thumbnail selection
    # ========================================================================
    cleanup_local_frames(project_id)

    # Use safe option as default thumbnail_url
    return ThumbnailResponse(
        project_id=project_id,
        status="draft",
        title_hint=payload.creative_direction.title_hint or "Untitled",
        thumbnail_url=top_url
        or (
            candidate_frame_urls[0]
            if candidate_frame_urls
            else "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=1200&q=80"
        ),
        selected_frame_url=top_url,
        profile_variant_url=None,
        layers=[],
        advisory=None,
        phase1=phase1,
        total_frames_extracted=len(extracted_frames),
        analysis_json_url=adaptive_result.get("analysis_json_url"),
        cost_usd=selection_result.get("cost_usd"),
        gemini_model=selection_result.get("gemini_model"),
        summary=summary_text,
    )


# ============================================================================
# Helper Functions for Mapping Request Data
# ============================================================================


def _map_optimization_to_goal(optimization: str | None) -> str:
    """Map optimization target to primary_goal."""
    if not optimization:
        return "maximize_ctr"

    opt_lower = optimization.lower()
    if "ctr" in opt_lower or "click" in opt_lower:
        return "maximize_ctr"
    elif "subscriber" in opt_lower or "grow" in opt_lower:
        return "grow_subscribers"
    elif "brand" in opt_lower or "authority" in opt_lower:
        return "brand_building"
    else:
        return "maximize_ctr"


def _infer_tone_from_mood(mood: str | None) -> str:
    """Infer tone from creative brief mood."""
    if not mood:
        return "professional"

    mood_lower = mood.lower()
    if any(word in mood_lower for word in ["energetic", "exciting", "dynamic", "bold"]):
        return "energetic"
    elif any(word in mood_lower for word in ["calm", "peaceful", "serene", "soft"]):
        return "calm"
    elif any(word in mood_lower for word in ["casual", "fun", "playful", "relaxed"]):
        return "casual"
    else:
        return "professional"


def _infer_personality(mood: str | None) -> list[str]:
    """Infer personality traits from mood."""
    if not mood:
        return ["professional"]

    mood_lower = mood.lower()
    traits = []

    if "energetic" in mood_lower or "exciting" in mood_lower:
        traits.append("energetic")
    if "professional" in mood_lower or "polished" in mood_lower:
        traits.append("professional")
    if "warm" in mood_lower or "friendly" in mood_lower:
        traits.append("warm")
    if "informative" in mood_lower or "educational" in mood_lower:
        traits.append("informative")

    return traits or ["professional"]


def _infer_visual_style(mood: str | None) -> str:
    """Infer visual style from mood."""
    if not mood:
        return "modern"

    mood_lower = mood.lower()
    if "minimalist" in mood_lower or "clean" in mood_lower:
        return "minimalist"
    elif "bold" in mood_lower or "dramatic" in mood_lower:
        return "bold"
    elif "vintage" in mood_lower or "retro" in mood_lower:
        return "vintage"
    else:
        return "modern"


@app.post(
    "/get-upload-url",
    response_model=SignedUrlResponse,
    responses={
        500: {"description": "Failed to generate signed URL"},
    },
    tags=["Video Upload"],
)
async def get_upload_url(payload: SignedUrlRequest) -> SignedUrlResponse:
    """
    Generate a signed URL for direct client upload to GCS.

    The client can use this signed URL to upload files directly to Google Cloud Storage
    without going through the backend server, which is more efficient for large files.

    **Usage:**
    1. Call this endpoint with filename, user_id, and content_type
    2. Receive a signed URL that's valid for 1 hour
    3. Upload file directly to GCS using PUT request to the signed URL
    4. Use the returned gcs_path for subsequent API calls

    **Example:**
    ```bash
    # Step 1: Get signed URL
    curl -X POST "http://localhost:9000/get-upload-url" \\
      -H "Content-Type: application/json" \\
      -d '{
        "user_id": "120accfe-aa23-41a3-b04f-36f581714d52",
        "filename": "my-video.mp4",
        "content_type": "video/mp4"
      }'

    # Step 2: Upload file using the signed URL
    curl -X PUT "<signed_url_from_response>" \\
      -H "Content-Type: video/mp4" \\
      --upload-file /path/to/my-video.mp4
    ```
    """
    # Sanitize filename
    safe_filename = sanitize_filename(payload.filename)

    # Generate signed URL
    try:
        signed_url, gcs_path = generate_signed_upload_url(
            filename=safe_filename,
            user_id=payload.user_id,
            content_type=payload.content_type,
            subfolder=payload.subfolder,
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to generate signed URL",
                "detail": str(e),
            },
        ) from e

    return SignedUrlResponse(
        signed_url=signed_url,
        gcs_path=gcs_path,
        expires_in_seconds=3600,
    )


@app.post(
    "/get-video-url",
    response_model=VideoUrlResponse,
    responses={
        404: {"description": "Video not found"},
        500: {"description": "Failed to generate signed URL"},
    },
    tags=["Video Upload"],
)
async def get_video_url(payload: VideoUrlRequest) -> VideoUrlResponse:
    """
    Generate a signed URL for viewing/downloading an existing video from GCS.

    This endpoint generates a temporary signed URL that allows reading a video file
    from Google Cloud Storage without requiring authentication. The URL is valid
    for 1 hour.

    **Usage:**
    1. Call this endpoint with the GCS path of an existing video
    2. Receive a signed URL that's valid for 1 hour
    3. Use the signed URL to view/download the video (GET request)

    **Example:**
    ```bash
    curl -X POST "http://localhost:9000/get-video-url" \\
      -H "Content-Type: application/json" \\
      -d '{
        "gcs_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4"
      }'
    ```

    You can also provide just the path without the gs:// prefix:
    ```bash
    curl -X POST "http://localhost:9000/get-video-url" \\
      -H "Content-Type: application/json" \\
      -d '{
        "gcs_path": "users/120accfe-aa23-41a3-b04f-36f581714d52/videos/my-video.mp4"
      }'
    ```
    """
    # Generate signed URL for viewing/downloading
    try:
        signed_url = generate_signed_download_url(
            gcs_path=payload.gcs_path,
        )

        # Normalize GCS path to gs:// format
        if payload.gcs_path.startswith("gs://"):
            gcs_path = payload.gcs_path
        else:
            gcs_path = f"gs://clickmoment-prod-assets/{payload.gcs_path}"

    except StorageError as e:
        # Check if it's a file not found error
        if "File not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Video not found",
                    "detail": str(e),
                },
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to generate signed URL",
                    "detail": str(e),
                },
            ) from e

    return VideoUrlResponse(
        signed_url=signed_url,
        gcs_path=gcs_path,
        expires_in_seconds=3600,
    )


@app.post(
    "/videos/upload",
    response_model=VideoUploadResponse,
    responses={
        400: {"model": VideoUploadError, "description": "Missing user_id"},
        413: {"model": VideoUploadError, "description": "File too large"},
        415: {"model": VideoUploadError, "description": "Unsupported media type"},
        500: {"model": VideoUploadError, "description": "Upload failed"},
    },
    tags=["Video Upload"],
)
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload"),
    user_id: str = Form(..., description="User ID from your application"),
) -> VideoUploadResponse:
    max_duration_min = DEFAULT_MAX_DURATION_SECONDS // 60
    max_duration_sec = DEFAULT_MAX_DURATION_SECONDS
    f"""
    Upload a video file to cloud storage.

    **File Requirements:**
    - Max size: 5 GB
    - Max duration: {max_duration_min} minutes ({max_duration_sec} seconds)
    - Allowed formats: mp4, mov, avi, mkv, webm, flv

    **Note:** Videos longer than {max_duration_min} minutes will be truncated during processing.
    Only the first {max_duration_min} minutes will be analyzed for transcription and thumbnail selection.

    **Storage Path:**
    - Files saved to: `users/{{user_id}}/videos/{{filename}}`
    - Files with same name are overwritten (with warning in response)

    **Example:**
    ```bash
    curl -X POST "http://localhost:9000/videos/upload" \\
      -F "file=@/path/to/1116_1_.mp4" \\
      -F "user_id=120accfe-aa23-41a3-b04f-36f581714d52"
    ```
    """
    # Validate file extension
    filename = file.filename or "video.mp4"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "Unsupported file format",
                "detail": f"File extension '{file_ext}' not allowed",
                "allowed_formats": list(ALLOWED_VIDEO_EXTENSIONS),
                "max_size_gb": 5,
            },
        )

    # Validate MIME type (if provided by client)
    if file.content_type and file.content_type not in ALLOWED_VIDEO_MIMETYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "Unsupported media type",
                "detail": f"Content-Type '{file.content_type}' not allowed",
                "allowed_formats": list(ALLOWED_VIDEO_EXTENSIONS),
                "max_size_gb": 5,
            },
        )

    # Read file content
    file_content = await file.read()
    file_size_bytes = len(file_content)

    # Validate file size
    if file_size_bytes > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "File too large",
                "detail": f"File size {file_size_bytes / 1024 / 1024 / 1024:.2f} GB exceeds limit of 5 GB",
                "allowed_formats": list(ALLOWED_VIDEO_EXTENSIONS),
                "max_size_gb": 5,
            },
        )

    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Upload to GCS
    try:
        gcs_url, was_overwritten = await upload_file_to_gcs(
            file_content=file_content,
            filename=safe_filename,
            user_id=user_id,
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Upload failed",
                "detail": str(e),
                "allowed_formats": list(ALLOWED_VIDEO_EXTENSIONS),
                "max_size_gb": 5,
            },
        ) from e

    # Build response
    status_msg = "warning" if was_overwritten else "success"
    message = (
        "Video uploaded successfully (overwrote existing file)"
        if was_overwritten
        else "Video uploaded successfully"
    )

    return VideoUploadResponse(
        status=status_msg,
        message=message,
        gcs_path=gcs_url,
        user_id=user_id,
        project_id=None,
        filename=safe_filename,
        file_size_mb=round(file_size_bytes / 1024 / 1024, 2),
        overwritten=was_overwritten,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a concise, friendly 422 payload."""
    errors = []
    for err in exc.errors():
        location_parts = [str(part) for part in err.get("loc", []) if part != "body"]
        errors.append(
            {
                "field": "body" if not location_parts else ".".join(location_parts),
                "message": err.get("msg"),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload",
            "errors": errors,
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
