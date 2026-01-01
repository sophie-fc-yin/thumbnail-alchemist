import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analysis.adaptive_sampling import orchestrate_adaptive_sampling
from app.audio.extraction import extract_audio_from_video, transcribe_and_analyze_audio
from app.models import (
    AdaptiveSamplingResponse,
    AudioBreakdownRequest,
    AudioBreakdownResponse,
    ImportanceSegment,
    ImportanceStatistics,
    ProcessingStats,
    SignedUrlRequest,
    SignedUrlResponse,
    SourceMedia,
    ThumbnailRequest,
    ThumbnailResponse,
    VideoUploadError,
    VideoUploadResponse,
    VideoUrlRequest,
    VideoUrlResponse,
    VisionBreakdownRequest,
)
from app.utils.storage import (
    StorageError,
    generate_signed_download_url,
    generate_signed_upload_url,
    upload_file_to_gcs,
)

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

    # Create SourceMedia object for audio processing
    source_media = SourceMedia(video_path=payload.video_path)

    # Extract audio from video (speech and full audio)
    try:
        audio_result = await extract_audio_from_video(
            content_sources=source_media,
            project_id=project_id,
            max_duration_seconds=payload.max_duration_seconds,
        )
    except Exception as e:
        raise ValueError(f"Failed to extract audio from video: {str(e)}") from e

    if not audio_result:
        raise ValueError("Failed to extract audio from video: ffmpeg returned no output")

    # Use speech lane for transcription
    speech_path = audio_result["speech"]

    # Analyze audio with transcription and features
    analysis = await transcribe_and_analyze_audio(
        audio_path=speech_path,  # Use speech-only lane for better transcription
        project_id=project_id,
        language=payload.language,
    )

    # Upload audio files and timeline JSON to GCS
    from pathlib import Path

    from google.cloud import storage

    gcs_speech_url = str(speech_path)  # Default to local path
    gcs_full_audio_url = str(audio_result["full_audio"])

    try:
        client = storage.Client()
        bucket = client.bucket("clickmoment-prod-assets")

        # Upload speech audio file
        speech_blob_path = f"projects/{project_id}/signals/audio/audio_speech.wav"
        speech_blob = bucket.blob(speech_blob_path)
        speech_blob.upload_from_filename(str(speech_path))
        gcs_speech_url = f"gs://clickmoment-prod-assets/{speech_blob_path}"

        # Upload full audio file (complete audio track)
        full_audio_path = Path(audio_result["full_audio"])
        full_audio_blob_path = f"projects/{project_id}/signals/audio/audio_full.wav"
        full_audio_blob = bucket.blob(full_audio_blob_path)
        full_audio_blob.upload_from_filename(str(full_audio_path))
        gcs_full_audio_url = f"gs://clickmoment-prod-assets/{full_audio_blob_path}"

        # Delete local audio files after upload
        Path(speech_path).unlink()
        full_audio_path.unlink()
        print(f"Uploaded speech audio to {gcs_speech_url}")
        print(f"Uploaded full audio to {gcs_full_audio_url}")

        # Upload timeline JSON if it exists
        if "timeline_path" in analysis:
            print(f"Found timeline_path in analysis: {analysis['timeline_path']}")
            timeline_path = Path(analysis["timeline_path"])
            if timeline_path.exists():
                print(f"Timeline file exists at {timeline_path}, uploading...")
                json_blob_path = f"projects/{project_id}/signals/audio/{timeline_path.name}"
                json_blob = bucket.blob(json_blob_path)
                json_blob.upload_from_filename(str(timeline_path))

                gcs_timeline_url = f"gs://clickmoment-prod-assets/{json_blob_path}"
                print(f"Successfully uploaded timeline JSON to {gcs_timeline_url}")

                # Delete local JSON file after upload
                timeline_path.unlink()
            else:
                print(f"Timeline file does not exist at {timeline_path}")
        else:
            print(f"No timeline_path in analysis. Keys: {list(analysis.keys())}")

    except Exception as e:
        print(f"Failed to upload audio files to GCS: {e}")

    return AudioBreakdownResponse(
        project_id=project_id,
        audio_path=gcs_speech_url,  # Return speech-only audio path
        transcript=analysis["transcript"],
        duration_seconds=analysis["duration_seconds"],
        speakers=analysis["speakers"],
        speech_tone=analysis["speech_tone"],
        music_tone=analysis["music_tone"],
        timeline=analysis["timeline"],
    )


@app.post("/vision/breakdown", response_model=AdaptiveSamplingResponse)
async def breakdown_vision(payload: VisionBreakdownRequest) -> AdaptiveSamplingResponse:
    """
    Extract frames using adaptive moment-importance-based sampling.

    This endpoint orchestrates the full pipeline:
    1. Audio breakdown → extract audio timeline
    2. Initial sparse sampling → sample frames for importance analysis
    3. Face analysis → detect expressions and motion
    4. Moment importance calculation → combine audio + visual signals
    5. Adaptive extraction → dense sampling around important moments
    6. Storage → upload all frames to GCS

    Returns frames with importance segments and processing statistics.
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    print(f"[API] Starting adaptive sampling for project {project_id}")

    # Run orchestrated pipeline
    result = await orchestrate_adaptive_sampling(
        video_path=payload.video_path,
        project_id=project_id,
        max_frames=payload.max_frames,
    )

    # Convert to response model
    response = AdaptiveSamplingResponse(
        project_id=project_id,
        frames=result["frames"],
        total_frames=len(result["frames"]),
        importance_segments=[ImportanceSegment(**seg) for seg in result["importance_segments"]],
        importance_statistics=ImportanceStatistics(**result["importance_statistics"]),
        processing_stats=ProcessingStats(**result["processing_stats"]),
        summary=result["summary"],
    )

    print(f"[API] Adaptive sampling complete: {response.total_frames} frames extracted")

    return response


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

    print(f"\n{'='*70}")
    print(f"THUMBNAIL GENERATION - Project {project_id}")
    print(f"{'='*70}")

    # Extract video path
    video_path = payload.content_sources.video_path
    if not video_path:
        # Image-only mode: allow callers to provide pre-extracted frames/screenshots.
        # This keeps the API usable without running the heavy video pipeline.
        image_paths = payload.content_sources.image_paths
        if image_paths:
            first = image_paths[0]
            return ThumbnailResponse(
                project_id=project_id,
                status="draft",
                recommended_title=payload.creative_brief.title_hint
                or "ClickMoment Phase-1 (image-only)",
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

    # ========================================================================
    # STEP 1: Adaptive Sampling - Extract candidate frames
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 1: Adaptive Sampling")
    print(f"{'─'*70}")
    print(f"Video: {video_path}")

    # Extract frames locally for processing
    # Note: Frames are stored locally during the processing pipeline.
    # Final selected thumbnails can be uploaded separately if needed.
    adaptive_result = await orchestrate_adaptive_sampling(
        video_path=video_path,
        project_id=project_id,
        max_frames=None,  # Auto-calculate based on video duration
    )

    candidate_frame_urls = adaptive_result["frames"]
    print(f"✅ Extracted {len(candidate_frame_urls)} candidate frames")

    # Load adaptive sampling analysis to get frame metadata
    import json

    from google.cloud import storage

    # Download analysis JSON from GCS
    analysis_json_url = adaptive_result.get("analysis_json_url")
    if not analysis_json_url:
        raise ValueError("No analysis JSON URL in adaptive sampling result")

    # Parse GCS path
    gcs_path = analysis_json_url.replace("gs://clickmoment-prod-assets/", "")
    client = storage.Client()
    bucket = client.bucket("clickmoment-prod-assets")
    blob = bucket.blob(gcs_path)

    # Download and parse JSON
    analysis_data = json.loads(blob.download_as_text())
    extracted_frames = analysis_data.get("extracted_frames", [])

    if not extracted_frames:
        raise ValueError("No frames found in adaptive sampling analysis")

    # Transform extracted_frames to ThumbnailSelector format
    # Each extracted frame now has its own face_analysis (no longer sparse sampling!)
    # CRITICAL: Ensure all frames have complete visual_analysis for scoring (aesthetics, psychology, etc.)
    enriched_frames = []
    frames_skipped = 0
    frames_with_incomplete_analysis = 0

    for idx, frame in enumerate(extracted_frames, 1):
        timestamp = frame.get("timestamp", 0.0)

        # Get face analysis directly from this frame (analyzed during extraction)
        face_analysis = frame.get("face_analysis", {})

        # Get both local path (for Gemini) and GCS URL (for debugging)
        local_path = frame.get("local_path", "")
        gcs_url = frame.get("frame_path", "")

        # Skip frames without local path (needed for Gemini)
        if not local_path:
            print(f"⚠️  Frame {idx}: missing local_path")
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
                    print(f"⚠️  [AUDIT] Frame {idx}: timestamp mismatch!")
                    print(f"    Metadata timestamp: {timestamp:.2f}s")
                    print(f"    Filename timestamp: {timestamp_from_filename_sec:.2f}s")
                    print(f"    Difference: {timestamp_diff:.2f}s")
                    print(f"    Path: {local_path}")
        except Exception:
            # Non-critical - just for debugging
            pass

        # Ensure visual_analysis has all required fields for scoring
        # Required for: aesthetics, psychology, face_quality, composition, technical_quality, editability
        required_visual_fields = {
            "has_face": face_analysis.get("has_face", False),
            "dominant_emotion": face_analysis.get("dominant_emotion", "unknown"),
            "expression_intensity": face_analysis.get("expression_intensity", 0.0),
            "eye_openness": face_analysis.get("eye_openness", 0.0),
            "mouth_openness": face_analysis.get("mouth_openness", 0.0),
            "head_pose": face_analysis.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}),
        }

        # Check if analysis is incomplete (missing required fields)
        if not face_analysis or any(
            key not in face_analysis for key in required_visual_fields.keys()
        ):
            frames_with_incomplete_analysis += 1
            if frames_with_incomplete_analysis <= 3:  # Log first 3
                print(f"⚠️  Frame {idx}: incomplete visual_analysis, filling defaults")

        # Merge to ensure all required fields are present
        complete_visual_analysis = {**face_analysis, **required_visual_fields}

        # CRITICAL: Use timestamp-based frame number to maintain identity through pipeline
        # This ensures frame_number stays tied to the actual frame, not loop position
        frame_number_from_timestamp = int(timestamp * 30)  # Assume ~30fps for unique IDs

        enriched_frames.append(
            {
                "frame_number": frame_number_from_timestamp,  # Timestamp-based unique ID
                "local_path": local_path,  # Local path for Gemini (faster)
                "url": gcs_url,  # GCS URL for debugging/reference
                "timestamp": timestamp,
                "moment_score": frame.get("moment_score", 0.0),
                "importance_score": frame.get("importance_score", 0.0),
                "importance_level": frame.get("importance_level", "unknown"),
                "visual_analysis": complete_visual_analysis,  # Complete analysis for scoring
            }
        )

    print("[API] Frame enrichment complete:")
    print(f"  Total frames: {len(extracted_frames)}")
    print(f"  Enriched frames: {len(enriched_frames)}")
    print(f"  Skipped (no local_path): {frames_skipped}")
    print(f"  Frames with incomplete analysis (filled): {frames_with_incomplete_analysis}")

    extracted_frames = enriched_frames

    # ========================================================================
    # STEP 2: Map request data to thumbnail selector format
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 2: Preparing Thumbnail Selection Agent")
    print(f"{'─'*70}")

    # Map creative brief
    selector_brief = {
        "video_title": payload.creative_brief.title_hint or "Untitled Video",
        "primary_message": payload.creative_brief.notes or "No description provided",
        "target_emotion": "curiosity",  # Default
        "primary_goal": _map_optimization_to_goal(payload.target.optimization),
        "tone": _infer_tone_from_mood(payload.creative_brief.mood),
    }

    # Map channel profile
    selector_profile = {
        "niche": payload.channel_profile.content_niche or "general",
        "personality": _infer_personality(payload.creative_brief.mood),
        "visual_style": _infer_visual_style(payload.creative_brief.mood),
    }

    print(f"Creative Brief: {selector_brief['video_title']}")
    print(f"Goal: {selector_brief['primary_goal']}")
    print(f"Niche: {selector_profile['niche']}")

    # ========================================================================
    # STEP 3: Run Thumbnail Selection Agent
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 3: AI Thumbnail Selection")
    print(f"{'─'*70}")

    from app.thumbnail_agent import ThumbnailSelector

    # Initialize selector (uses GEMINI_API_KEY)
    try:
        selector = ThumbnailSelector(use_pro=False)  # Use Flash by default
    except ValueError as e:
        # GEMINI_API_KEY not set - return mock response
        print(f"⚠️  {e}")
        print("Returning mock response without AI selection")

        return ThumbnailResponse(
            project_id=project_id,
            status="draft",
            recommended_title=payload.creative_brief.title_hint or "AI-Generated Thumbnail",
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

    # Run selection agent
    selection_result = await selector.select_best_thumbnail(
        frames=extracted_frames,
        creative_brief=selector_brief,
        channel_profile=selector_profile,
    )

    # Extract Phase-1 diagnostic data from Gemini response
    moments = selection_result.get("moments", [])
    meta = selection_result.get("meta", {})
    debug_data = selection_result.get("debug", {})

    print("✅ ClickMoment Phase-1 Insights Generated")
    print(f"📊 Confidence: {meta.get('confidence', 'unknown')}")
    if moments:
        print(f"✨ Top moment: {moments[0].get('frame_id', 'N/A')}")

    # ========================================================================
    # STEP 4: Build Response
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 4: Building Response")
    print(f"{'─'*70}")

    # Helper function to extract frame number, URL, and timestamp from frame_id
    def get_frame_info(frame_id: str) -> tuple[int, str, str]:
        """Extract frame number from 'Frame X' format and convert GCS URL to signed HTTP URL.

        Returns:
            Tuple of (frame_number, frame_url, timestamp_string)
        """
        from app.vision.extraction import generate_signed_url

        try:
            num = int(frame_id.replace("Frame", "").strip())
            # Find matching frame in extracted_frames
            if 0 < num <= len(extracted_frames):
                frame = extracted_frames[num - 1]
                gcs_url = frame.get("url", "")
                # Get actual timestamp from frame data (not from Gemini response)
                frame_timestamp = frame.get("timestamp", 0.0)
                timestamp_str = f"{frame_timestamp:.1f}s"
                # Convert GCS URL to signed HTTP URL
                if gcs_url.startswith("gs://"):
                    return num, generate_signed_url(gcs_url), timestamp_str
                return num, gcs_url, timestamp_str
        except (ValueError, IndexError):
            pass

        # Fallback to first frame
        if extracted_frames:
            gcs_url = extracted_frames[0].get("url", "")
            frame_timestamp = extracted_frames[0].get("timestamp", 0.0)
            timestamp_str = f"{frame_timestamp:.1f}s"
            if gcs_url.startswith("gs://"):
                return 1, generate_signed_url(gcs_url), timestamp_str
            return 1, gcs_url, timestamp_str
        return 1, "", "0.0s"

    # Build Phase-1 response model (diagnostic, non-prescriptive)
    from app.models.thumbnail import ClickMomentPhase1, Phase1MomentInsight, Phase1Pillars

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
    from app.analysis.adaptive_sampling import cleanup_local_frames

    cleanup_local_frames(project_id)

    # Use safe option as default thumbnail_url
    return ThumbnailResponse(
        project_id=project_id,
        status="draft",
        recommended_title=payload.creative_brief.title_hint or "AI-Analyzed Thumbnail Options",
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
    """
    Upload a video file to cloud storage.

    **File Requirements:**
    - Max size: 5 GB
    - Allowed formats: mp4, mov, avi, mkv, webm, flv

    **Storage Path:**
    - Files saved to: `users/{user_id}/videos/{filename}`
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
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
