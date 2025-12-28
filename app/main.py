import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.adaptive_sampling import orchestrate_adaptive_sampling
from app.audio_media import extract_audio_from_video, transcribe_and_analyze_audio
from app.models import (
    AdaptiveSamplingResponse,
    AudioBreakdownRequest,
    AudioBreakdownResponse,
    CompositionLayer,
    FrameScore,
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
    VisionBreakdownResponse,
)
from app.storage import (
    StorageError,
    generate_signed_download_url,
    generate_signed_upload_url,
    upload_file_to_gcs,
)
from app.vision_media import extract_candidate_frames, validate_and_load_content
from app.vision_stack import analyze_frame_quality, rank_frames

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

    # Extract audio from video (speech and music lanes)
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
    gcs_music_url = str(audio_result["music"])

    try:
        client = storage.Client()
        bucket = client.bucket("clickmoment-prod-assets")

        # Upload speech audio file
        speech_blob_path = f"projects/{project_id}/signals/audio/audio_speech.wav"
        speech_blob = bucket.blob(speech_blob_path)
        speech_blob.upload_from_filename(str(speech_path))
        gcs_speech_url = f"gs://clickmoment-prod-assets/{speech_blob_path}"

        # Upload music audio file (full audio)
        music_path = Path(audio_result["music"])
        music_blob_path = f"projects/{project_id}/signals/audio/audio_music.wav"
        music_blob = bucket.blob(music_blob_path)
        music_blob.upload_from_filename(str(music_path))
        gcs_music_url = f"gs://clickmoment-prod-assets/{music_blob_path}"

        # Delete local audio files after upload
        Path(speech_path).unlink()
        music_path.unlink()
        print(f"Uploaded speech audio to {gcs_speech_url}")
        print(f"Uploaded music audio to {gcs_music_url}")

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


@app.post("/vision/breakdown", response_model=VisionBreakdownResponse)
async def breakdown_vision(payload: VisionBreakdownRequest) -> VisionBreakdownResponse:
    """
    Break down video/images into scored frames.

    Returns scored frames with quality metrics and rankings.
    """
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    # Create SourceMedia object
    source_media = SourceMedia(video_path=payload.video_path)

    # Get video duration first
    import asyncio
    import shutil

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        from app.vision_media import generate_signed_url

        video_url = generate_signed_url(payload.video_path)

        probe_process = await asyncio.create_subprocess_exec(
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await probe_process.communicate()
        try:
            video_duration = float(stdout.decode().strip())
        except (ValueError, AttributeError):
            video_duration = None
    else:
        video_duration = None

    # Extract frames from video (keep local for analysis)
    candidate_frames = await extract_candidate_frames(
        content_sources=source_media,
        project_id=project_id,
        max_frames=payload.max_frames,
        upload_to_gcs=False,  # Don't upload yet - need local files for analysis
    )

    if not candidate_frames:
        raise ValueError("No frames extracted or provided")

    # Calculate timestamp for each frame (evenly distributed across video duration)
    frame_timestamps = {}
    if video_duration:
        for idx, frame_path in enumerate(candidate_frames):
            timestamp = (
                (idx / max(1, len(candidate_frames) - 1)) * video_duration
                if len(candidate_frames) > 1
                else 0
            )
            frame_timestamps[str(frame_path)] = timestamp

    # Analyze frame quality (requires local files)
    frame_signals = analyze_frame_quality(candidate_frames)

    # Rank frames by quality
    ranked_frames = rank_frames(frame_signals)

    # Upload frames to GCS after analysis
    from pathlib import Path

    from google.cloud import storage

    gcs_frame_urls = {}
    try:
        client = storage.Client()
        bucket = client.bucket("clickmoment-prod-assets")

        for signal in ranked_frames:
            local_path = Path(signal.path)
            blob_path = f"projects/{project_id}/signals/frames/{local_path.name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))

            gcs_url = f"gs://clickmoment-prod-assets/{blob_path}"
            gcs_frame_urls[str(local_path)] = gcs_url

            # Delete local file after upload
            local_path.unlink()
    except Exception as e:
        print(f"Failed to upload frames to GCS: {e}")

    # Convert to response format
    scored_frames = [
        FrameScore(
            frame_path=gcs_frame_urls.get(str(signal.path), str(signal.path)),
            timestamp=frame_timestamps.get(str(signal.path)),
            brightness=signal.brightness,
            sharpness=signal.sharpness,
            motion=signal.motion,
            face_score=signal.face,
            expression_score=signal.expression,
            highlight_score=signal.highlight_score,
            rank=idx + 1,
        )
        for idx, signal in enumerate(ranked_frames)
    ]

    best_frame = scored_frames[0] if scored_frames else None

    # Build summary with safe formatting
    if best_frame and best_frame.highlight_score is not None:
        face_score_str = f"{best_frame.face_score:.3f}" if best_frame.face_score else "0.000"
        summary = f"Analyzed {len(scored_frames)} frames. Best frame has highlight score of {best_frame.highlight_score:.3f} with face detection score of {face_score_str}."
    else:
        summary = f"Analyzed {len(scored_frames)} frames."

    return VisionBreakdownResponse(
        project_id=project_id,
        total_frames=len(scored_frames),
        scored_frames=scored_frames,
        best_frame=best_frame,
        summary=summary,
    )


@app.post("/vision/adaptive-sampling", response_model=AdaptiveSamplingResponse)
async def adaptive_sampling(payload: VisionBreakdownRequest) -> AdaptiveSamplingResponse:
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
    # Use provided project_id or generate new one
    project_id = payload.project_id or str(uuid4())

    # Extract and process incoming request data
    content_sources = payload.content_sources
    _profile_photos = payload.profile_photos
    _target = payload.target
    creative_brief = payload.creative_brief

    # Validate all content source paths exist and are accessible, then extract metadata
    _content_metadata = await validate_and_load_content(content_sources)

    # Extract candidate frames from video
    candidate_frames = await extract_candidate_frames(content_sources, project_id=project_id)

    # TODO: analyze_frame_quality - Score frames based on composition, lighting, emotion, and platform requirements
    # scored_frames = await analyze_frame_quality(candidate_frames, target.platform, creative_brief.mood)

    # TODO: select_best_frame - Choose optimal frame based on scoring algorithm and optimization target
    # selected_frame = select_best_frame(scored_frames, target.optimization)

    # TODO: create_user_avatar - Load and analyze profile photo files to create/select best avatar for thumbnail composition
    # This processes multiple profile photo files from provided paths to find the best pose, angle, and expression
    # user_avatar = await create_user_avatar(profile_photos, target.platform, creative_brief.mood) if profile_photos else None

    # TODO: generate_background_layer - Create or select background using mood, brand colors, and platform
    # background_layer = await generate_background_layer(selected_frame, creative_brief.mood, creative_brief.brand_colors, target.platform)

    # TODO: extract_subject - Remove background from subject and prepare for composition
    # subject_layer = await extract_subject(selected_frame, user_avatar)

    # TODO: generate_title_text - Use AI to generate compelling title based on hint, notes, and content analysis
    # generated_title = await generate_title_text(creative_brief.title_hint, creative_brief.notes, content_metadata)

    # TODO: design_title_layer - Create title typography with styling, effects, and brand colors
    # title_layer = await design_title_layer(generated_title, creative_brief.brand_colors, creative_brief.mood)

    # TODO: identify_callout_elements - Determine what visual callouts would enhance thumbnail based on platform and optimization target
    # callout_elements = await identify_callout_elements(selected_frame, target.platform, target.optimization)

    # TODO: compose_final_thumbnail - Combine all layers into final thumbnail composition optimized for platform
    # final_thumbnail = await compose_final_thumbnail(
    #     background_layer, subject_layer, title_layer, callout_elements, target.platform
    # )

    # TODO: upload_to_storage - Upload generated assets to cloud storage and return URLs
    # uploaded_assets = await upload_to_storage(final_thumbnail, selected_frame, user_avatar, project_id)

    # Return mock response for now
    return ThumbnailResponse(
        project_id=project_id,
        status="draft",
        recommended_title=creative_brief.title_hint or "The AI That Designs Thumbnails For You",
        thumbnail_url="https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f"
        "?auto=format&fit=crop&w=1200&q=80",
        selected_frame_url="https://images.unsplash.com/photo-1521737604893-d14cc237f11d"
        "?auto=format&fit=crop&w=900&q=80",
        profile_variant_url="https://images.unsplash.com/photo-1494790108377-be9c29b29330"
        "?auto=format&fit=crop&w=800&q=80",
        layers=[
            CompositionLayer(
                kind="background",
                description="High-energy gradient with subtle glow to lift the subject.",
                asset_url="https://images.unsplash.com/photo-1502082553048-f009c37129b9"
                "?auto=format&fit=crop&w=1200&q=80",
            ),
            CompositionLayer(
                kind="subject",
                description="Cutout of creator with key light on the right shoulder.",
                asset_url="https://images.unsplash.com/photo-1469474968028-56623f02e42e"
                "?auto=format&fit=crop&w=800&q=80",
            ),
            CompositionLayer(
                kind="title",
                description="Bold, uppercase typography with yellow stroke and drop shadow.",
                asset_url=None,
            ),
            CompositionLayer(
                kind="callout",
                description="Arrow pointing at the laptop screen to anchor viewer focus.",
                asset_url=None,
            ),
        ],
        summary=(
            f"Draft composition combining your sources with a bold, high-contrast layout. "
            f"Collected {len(candidate_frames)} candidate frame(s) from your uploads. "
            "Static assets are returned for now; a later iteration will run the full "
            "agent pipeline to select frames, pose the subject, and render the final thumbnail."
        ),
    )


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
