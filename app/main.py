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
    PaceSegment,
    PaceStatistics,
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
    Extract frames using adaptive pace-based sampling.

    This endpoint orchestrates the full pipeline:
    1. Audio breakdown → extract audio timeline
    2. Initial sparse sampling → sample frames for pace analysis
    3. Face analysis → detect expressions and motion
    4. Pace calculation → combine audio + visual signals
    5. Adaptive extraction → dense sampling where pace is high
    6. Storage → upload all frames to GCS

    Returns frames with pace segments and processing statistics.
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    print(f"[API] Starting adaptive sampling for project {project_id}")

    # Run orchestrated pipeline
    result = await orchestrate_adaptive_sampling(
        video_path=payload.video_path,
        project_id=project_id,
        max_frames=payload.max_frames,
        upload_to_gcs=True,
    )

    # Convert to response model
    response = AdaptiveSamplingResponse(
        project_id=project_id,
        frames=result["frames"],
        total_frames=len(result["frames"]),
        pace_segments=[PaceSegment(**seg) for seg in result["pace_segments"]],
        pace_statistics=PaceStatistics(**result["pace_statistics"]),
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
    1. Adaptive sampling → Extract candidate frames based on pace
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="video_path is required in content_sources",
        )

    # ========================================================================
    # STEP 1: Adaptive Sampling - Extract candidate frames
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 1: Adaptive Sampling")
    print(f"{'─'*70}")
    print(f"Video: {video_path}")

    adaptive_result = await orchestrate_adaptive_sampling(
        video_path=video_path,
        project_id=project_id,
        max_frames=None,  # Auto-calculate based on video duration
        upload_to_gcs=True,
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
    visual_analysis_frames = analysis_data.get("visual_analysis", {}).get("sample_frames", [])

    # Enrich extracted_frames with required fields
    enriched_frames = []
    for idx, frame in enumerate(extracted_frames, 1):
        timestamp = frame.get("timestamp", 0.0)

        # Get matching visual analysis (find closest)
        closest_visual = (
            min(
                visual_analysis_frames,
                key=lambda v: abs(v.get("timestamp", 0) - timestamp),
                default={},
            )
            if visual_analysis_frames
            else {}
        )

        enriched_frames.append(
            {
                "frame_number": idx,
                "path": frame.get("frame_path", ""),
                "url": frame.get("frame_path", ""),  # Same as path for GCS
                "timestamp": timestamp,
                "moment_score": frame.get("moment_score", 0.0),
                "pace_score": frame.get("pace_score", 0.0),
                "visual_analysis": closest_visual.get("face_analysis", {}),
            }
        )

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
            total_frames_extracted=len(candidate_frame_urls),
            analysis_json_url=adaptive_result.get("analysis_json_url"),
            cost_usd=None,
            gemini_model=None,
            summary=f"Extracted {len(candidate_frame_urls)} frames using adaptive sampling. Set GEMINI_API_KEY to enable AI-powered thumbnail advisory with strategic options (safe/bold/avoid).",
        )

    # Run selection agent
    selection_result = await selector.select_best_thumbnail(
        frames=extracted_frames,
        creative_brief=selector_brief,
        channel_profile=selector_profile,
    )

    # Extract advisory data from Gemini response
    safe_option = selection_result.get("safe", {})
    high_variance_option = selection_result.get("high_variance", {})
    avoid_option = selection_result.get("avoid", {})
    meta = selection_result.get("meta", {})
    debug_data = selection_result.get("debug", {})

    print("✅ AI Advisory Generated")
    print(f"📊 Confidence: {meta.get('confidence', 'unknown')}")
    print(f"💡 Safe option: {safe_option.get('frame_id', 'N/A')}")
    print(f"🎯 Bold option: {high_variance_option.get('frame_id', 'N/A')}")
    print(f"⚠️  Avoid: {avoid_option.get('frame_id', 'N/A')}")

    # ========================================================================
    # STEP 4: Build Response
    # ========================================================================
    print(f"\n{'─'*70}")
    print("STEP 4: Building Response")
    print(f"{'─'*70}")

    # Helper function to extract frame number and URL from frame_id
    def get_frame_info(frame_id: str) -> tuple[int, str]:
        """Extract frame number from 'Frame X' format."""
        try:
            num = int(frame_id.replace("Frame", "").strip())
            # Find matching frame in extracted_frames
            if 0 < num <= len(extracted_frames):
                frame = extracted_frames[num - 1]
                return num, frame.get("url", "")
        except (ValueError, IndexError):
            pass
        return 1, extracted_frames[0].get("url", "") if extracted_frames else ""

    # Build advisory with full data
    from app.models.thumbnail import AdvisoryMeta, FrameOption, ThumbnailAdvisory

    safe_num, safe_url = get_frame_info(safe_option.get("frame_id", "Frame 1"))
    advisory = ThumbnailAdvisory(
        safe=FrameOption(
            frame_id=safe_option.get("frame_id", "Frame 1"),
            frame_number=safe_num,
            timestamp=safe_option.get("timestamp", "0.0s"),
            frame_url=safe_url,
            one_liner=safe_option.get("one_liner", ""),
            reasons=safe_option.get("reasons", []),
            risk_notes=safe_option.get("risk_notes", []),
        ),
        high_variance=FrameOption(
            frame_id=high_variance_option.get("frame_id", "Frame 1"),
            frame_number=get_frame_info(high_variance_option.get("frame_id", "Frame 1"))[0],
            timestamp=high_variance_option.get("timestamp", "0.0s"),
            frame_url=get_frame_info(high_variance_option.get("frame_id", "Frame 1"))[1],
            one_liner=high_variance_option.get("one_liner", ""),
            reasons=high_variance_option.get("reasons", []),
            risk_notes=high_variance_option.get("risk_notes", []),
        ),
        avoid=FrameOption(
            frame_id=avoid_option.get("frame_id", "Frame 1"),
            frame_number=get_frame_info(avoid_option.get("frame_id", "Frame 1"))[0],
            timestamp=avoid_option.get("timestamp", "0.0s"),
            frame_url=get_frame_info(avoid_option.get("frame_id", "Frame 1"))[1],
            one_liner=avoid_option.get("one_liner", ""),
            reasons=avoid_option.get("reasons", []),
            risk_notes=avoid_option.get("risk_notes", []),
        ),
        meta=AdvisoryMeta(
            confidence=meta.get("confidence", "medium"),
            what_changed=meta.get("what_changed", ""),
            user_control_note=meta.get("user_control_note", ""),
        ),
        debug=debug_data,
    )

    # Build summary text
    summary_parts = [
        "✅ AI Thumbnail Advisory Generated",
        f"\n\n📊 Confidence: {meta.get('confidence', 'medium').upper()}",
        f"\n🎯 Total Frames Analyzed: {len(extracted_frames)}",
        f"\n\n🛡️  SAFE OPTION (Frame {safe_num}):",
        f"\n{safe_option.get('one_liner', '')}",
        "\n\n🚀 BOLD OPTION:",
        f"\n{high_variance_option.get('one_liner', '')}",
        "\n\n⚠️  AVOID:",
        f"\n{avoid_option.get('one_liner', '')}",
        f"\n\n💡 {meta.get('user_control_note', 'You decide which option fits your vision best.')}",
    ]
    summary_text = "".join(summary_parts)

    # Use safe option as default thumbnail_url
    return ThumbnailResponse(
        project_id=project_id,
        status="draft",
        recommended_title=payload.creative_brief.title_hint or "AI-Analyzed Thumbnail Options",
        thumbnail_url=safe_url,
        selected_frame_url=safe_url,
        profile_variant_url=None,
        layers=[],
        advisory=advisory,
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
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
