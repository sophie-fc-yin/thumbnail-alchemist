# thumbnail-alchemist
Thumbnail Alchemist is a multimodal AI agent that turns raw video, images, and creator profiles into high-impact, scroll-stopping YouTube thumbnails. Like an alchemist transforming base materials into gold, this system fuses visual understanding, stylistic intent, and agent-driven reasoning to craft thumbnails engineered for attention and CTR.

Powered by advanced vision-language models and a multi-agent architecture, Thumbnail Alchemist analyzes your content, understands your narrative, generates clickable titles, selects the perfect snapshot, positions your character, and blends everything into a polished, creator-ready thumbnail.

✨ What Thumbnail Alchemist Does
- Understands your content
  - Extracts context, mood, themes, and key frames from video or image inputs.
- Transforms your profile photo
  - Generates poses, expressions, and stylistic variations that match your brand and thumbnail style.
- Generates powerful, emotional titles
  - Tailors text to your content goals (funny, dramatic, educational, cinematic, etc.).
- Selects the strongest video frame
  - Analyzes clarity, energy, facial visibility, and narrative relevance.
- Composes a professional thumbnail
  - Blends profile photo, video frame, product images, lighting, color accents, typography, and layout.

⚙️ Agent Architecture

Thumbnail Alchemist uses a structured agent flow:
1. Analyzer Agent – extracts context from images, video, text, and product context
2. Title Alchemist Agent – creates optimized, attention-grabbing titles
3. Snapshot Selector Agent – picks or generates the best video moment
4. Composition Agent – combines all assets into a cohesive final thumbnail

Each agent collaborates and provides feedback, forming an iterative loop that refines the final output.

## Design (Agent Spec)

This section describes how to design the multi-modal agent that consumes video frames, creative brief, and audio context, then outputs thumbnail components with reasoning.

### Inputs
- Video frames: keyframes + scene cuts + top-N high-motion frames
- Creative brief: goal, audience, brand constraints, style keywords
- Audio context: transcript, sentiment, emphasis peaks, hook segments
- Platform: YouTube, TikTok, Shorts, etc.

### Core pipeline
1. Frame selection
   - Sample keyframes and high-motion segments.
   - Filter for clarity, faces, and action.
2. Audio + brief alignment
   - Summarize the audio to 1-2 sentences.
   - Detect emotional peaks and "hook" moments.
3. Cross-modal synthesis
   - Derive a "core promise" and "primary hook."
   - Map visual motifs to the hook.
4. Component planning
   - Decide components and hierarchy (hero subject, background, text, iconography, brand marks).
5. Component specification
   - Produce structured specs for each component (text, image, vector).
6. Optimization and scoring
   - Score against platform heuristics (readability, contrast, safe areas, clutter).
7. Explainability
   - Provide rationale for each component and layout decision.

### Component spec schema (example)
Use a structured output to keep the system deterministic and easy to consume:

```json
{
  "platform": "youtube",
  "thumbnail_goal": "curiosity + clarity",
  "components": [
    {
      "component_type": "image",
      "purpose": "hero subject",
      "content": "host close-up, surprised expression, gaze toward product",
      "style": "high contrast, warm highlights, shallow depth of field",
      "layout": "left third, 10% margin from edges",
      "reasoning": "faces increase CTR; gaze directs attention"
    },
    {
      "component_type": "text",
      "purpose": "hook",
      "content": "I Tried the Impossible",
      "style": "bold condensed, white with black stroke",
      "layout": "right third, top half, safe area",
      "reasoning": "short copy improves mobile readability"
    },
    {
      "component_type": "vector",
      "purpose": "emphasis",
      "content": "arrow pointing to product",
      "style": "thick stroke, brand accent color",
      "layout": "near product, avoids text",
      "reasoning": "directs attention to key object"
    }
  ],
  "variants": ["emotion-first", "curiosity-first"]
}
```

### Platform rules (starter heuristics)
- YouTube: 16:9, 2-5 words, strong face lighting, clear subject separation.
- TikTok/Shorts: 9:16, larger text, center-safe layout, minimal text.

### Minimal MVP
- Rule-based component planner + simple scoring.
- Add asset generation later (text styling, vector overlays, image compositing).

## Setup & Installation

### Prerequisites

1. **Install uv** (if not already installed):
```bash
brew install uv
```

### Environment Setup

2. **Create virtual environment and install dependencies**:
```bash
# Create virtual environment (creates .venv directory)
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies from pyproject.toml
uv sync
```

Alternatively, you can use `uv` without activating the venv:
```bash
# Create and use venv automatically
uv venv
uv run uvicorn app.main:app --reload
```

### Pre-commit Hooks

3. **Set up pre-commit hooks** (optional but recommended):
```bash
# Install dev dependencies (includes pre-commit)
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files (optional)
pre-commit run --all-files
```

The pre-commit configuration includes:
- Code formatting (ruff)
- Linting (ruff)
- Import sorting (isort)
- File checks (trailing whitespace, end of file, YAML/JSON/TOML validation)
- Security checks (private key detection)

## API (WIP)

This repo now includes a starter FastAPI service with a static thumbnail generation demo.

- Endpoint: `POST /thumbnails/generate`
- Request model: describes input media and creative intent (`sources`, optional `profile_photo_url`, `mood`, `title_hint`, `goal`, `brand_colors`, `notes`).
- Response model: returns a placeholder composition with demo assets and layer descriptions.

### Run locally

```bash
# If venv is activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

# Or using uv directly (no activation needed)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

# Then visit http://localhost:9000/docs

# Or use make (uvicorn must be installed)
make serve
```

## 📊 API Limits & Configuration

### Video Upload Limits

The API enforces the following limits for video uploads and processing:

| Setting | Limit | Description |
|---------|-------|-------------|
| **Maximum File Size** | 5 GB | Maximum video file size for upload via `/videos/upload` |
| **Maximum Audio Duration** | 30 minutes (1800s) | Maximum duration of audio/video to process for transcription and analysis |
| **Request Timeout** | 15 minutes (900s) | Cloud Run request timeout - allows time for large file uploads and processing |
| **Allowed Formats** | mp4, mov, avi, mkv, webm, flv | Supported video file formats |

### Application Configuration

These limits are configured in the application code:

- **File size limit**: `app/main.py` - `MAX_VIDEO_SIZE_BYTES = 5 * 1024 * 1024 * 1024`
- **Audio duration**: `app/models/audio.py` - `max_duration_seconds: int = Field(1800, ...)`
- **Audio extraction**: `app/audio_media.py` - `max_duration_seconds: int = 1800`

### Infrastructure Configuration

Infrastructure settings (Cloud Run timeout, IAM permissions, etc.) are managed via **Terraform**. See the `terraform/` directory for:

- Cloud Run service configuration
- IAM roles and permissions
- GCS bucket setup
- Network and security settings

To apply infrastructure changes:

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

See `terraform/README.md` for detailed instructions.

## 🎯 Adaptive Pace-Based Frame Sampling

Thumbnail Alchemist uses **intelligent frame sampling** that adapts to video pace, ensuring we capture key moments without wasting processing power on static scenes.

### How It Works

Instead of uniformly sampling frames, we analyze video pace using **multi-signal fusion**:

```
Input Signals (all normalized to [0, 1]):
├─ Facial Expression Delta (25%) - emotion changes via MediaPipe + FER+
├─ Landmark Motion (15%)         - head movement, gestures
├─ Audio Energy Delta (20%)      - music, excitement, emphasis
├─ Speech Emotion Delta (15%)    - vocal intensity changes
└─ Audio Score (25%)             - comprehensive audio analysis [NEW]
          ↓
    Pace Score [0, 1]
          ↓
┌──────────────────────────────────────┐
│ Low (0.0-0.3)  → 1.5-2.0s intervals  │  Calm talking, slow scenes
│ Med (0.3-0.7)  → 0.5-1.0s intervals  │  Normal engagement
│ High (0.7-1.0) → 0.1-0.25s intervals │  Emotional peaks, action
└──────────────────────────────────────┘
```

### Key Features

**🎭 Face Expression Analysis** (`app/face_analysis.py`)
- MediaPipe for face structure (landmarks, eye/mouth openness, head pose)
- FER+ ONNX model for emotion detection
- **Expression intensity** = `1 - P(neutral)` (single scalar measuring "how much is happening")
- Tracks facial motion between frames

**🎤 Audio Scoring** (`app/audio_media.py`)
- Comprehensive audio analysis with 4 key components:
  - **Speech Gate**: Binary gate (1.0 if speech present, 0.0 otherwise)
  - **Text Importance**: Categorical scoring (1.0=claim/reveal, 0.7=emphasis, 0.4=neutral, 0.1=filler)
  - **Emphasis Score**: Local energy deviation from rolling baseline [0, 1]
  - **BGM Penalty**: Background music penalty based on energy variance [0.2, 1.0]
- Final formula: `audio_score = speech_gate × text_importance × emphasis_score × bgm_penalty`
- Detects important moments based on content, delivery, and audio clarity
- Automatically scores each timeline segment

**📊 Pace Calculation** (`app/pace_analysis.py`)
- Multi-signal fusion with configurable weights
- Segments video into pace regions (low/medium/high)
- Converts pace score to sampling intervals
- Calculates audio energy and speech emotion deltas
- Integrates comprehensive audio score for better segment detection

**⚡ Why This Approach?**

1. **Efficient**: Only densely sample where it matters (emotional moments, action)
2. **Human-Like**: Mimics how editors scrub through footage
3. **Signal Fusion**: Combines audio + visual cues for robust detection
4. **Not ML**: Pure signal processing - fast, deterministic, explainable

### Example

```python
from app.face_analysis import FaceExpressionAnalyzer
from app.pace_analysis import calculate_pace_score, pace_to_sampling_interval

# Analyze a frame
analyzer = FaceExpressionAnalyzer()
analysis = analyzer.analyze_frame("frame.jpg")

# Calculate pace from multiple signals
pace = calculate_pace_score(
    expression_delta=0.8,      # High expression change
    landmark_motion=0.6,       # Moderate head movement
    audio_energy_delta=0.9,    # High audio energy spike
    speech_emotion_delta=0.7,  # Strong vocal emphasis
    audio_score=0.85,          # High audio score (speech + importance + emphasis)
)
# pace ≈ 0.82 (high pace!)

# Convert to sampling interval
interval = pace_to_sampling_interval(pace)
# interval ≈ 0.12s (dense sampling for this exciting moment)
```

### Installation

The adaptive sampling requires additional dependencies:

```bash
# Already included in pyproject.toml
uv sync  # Installs mediapipe, onnxruntime, huggingface-hub
```

On first run, the FER+ emotion model will auto-download from HuggingFace (~50MB).

### Technical Details

**Face Analysis Pipeline:**
1. MediaPipe detects face and extracts 468 landmarks
2. Calculate geometric features (eye openness, mouth openness, head pose)
3. Crop and normalize face to 64x64 grayscale
4. FER+ ONNX model outputs emotion probabilities
5. Expression intensity = distance from neutral state

**Pace Segmentation:**
1. Calculate pace score for each time point
2. Group consecutive similar-pace sections
3. Sample densely in high-pace segments, sparsely in low-pace
4. Result: 70% fewer frames than uniform sampling, better moment capture

**Performance:**
- Face analysis: ~10-20ms per frame (CPU)
- Pace calculation: <1ms per time point
- FER+ inference: ~5ms per face (ONNX on CPU)

### API Endpoint

**POST `/vision/adaptive-sampling`**

The adaptive sampling pipeline is available as a single API endpoint that orchestrates everything:

```bash
curl -X POST "http://localhost:9000/vision/adaptive-sampling" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "gs://clickmoment-prod-assets/users/120accfe-aa23-41a3-b04f-36f581714d52/videos/1116_1_.mp4",
    "project_id": "my-project-123",
    "max_frames": 100
  }'
```

**Response:**
```json
{
  "project_id": "my-project-123",
  "frames": [
    "gs://clickmoment-prod-assets/projects/my-project-123/signals/frames/frame_0ms.jpg",
    "gs://clickmoment-prod-assets/projects/my-project-123/signals/frames/frame_1500ms.jpg",
    "gs://clickmoment-prod-assets/projects/my-project-123/signals/frames/frame_2100ms.jpg"
  ],
  "total_frames": 87,
  "pace_segments": [
    {
      "start_time": 0.0,
      "end_time": 15.3,
      "avg_pace": 0.25,
      "pace_category": "low"
    },
    {
      "start_time": 15.3,
      "end_time": 32.7,
      "avg_pace": 0.82,
      "pace_category": "high"
    }
  ],
  "pace_statistics": {
    "avg_pace": 0.54,
    "segment_counts": {
      "low": 2,
      "medium": 3,
      "high": 2
    },
    "total_segments": 7
  },
  "processing_stats": {
    "audio_time": 12.3,
    "initial_sampling_time": 3.2,
    "face_analysis_time": 5.1,
    "pace_calculation_time": 0.4,
    "adaptive_extraction_time": 8.7,
    "total_time": 29.7
  },
  "summary": "Adaptive sampling complete: 87 frames extracted. Pace segments: 2 low, 3 medium, 2 high. Average pace: 0.54. Processing time: 29.7s"
}
```

### Orchestrated Pipeline

The endpoint runs a **5-step coordinated workflow** (`app/adaptive_sampling.py`):

1. **Audio Breakdown** - Extract audio, transcribe, analyze timeline
2. **Initial Sampling** - Extract ~20 sample frames evenly distributed
3. **Face Analysis** - MediaPipe + FER+ on samples for expression/motion
4. **Pace Calculation** - Fuse audio + visual signals into pace scores
5. **Adaptive Extraction** - Dense sampling where pace is high, sparse where low
6. **Storage** - Upload all frames to GCS with millisecond timestamps

**All data is recorded and returned:**
- Frame paths (GCS URLs)
- Pace segments with categories
- Processing time breakdown
- Pace statistics (avg, counts)

### Configuration

Customize pace weights in your code:

```python
# Default weights
weights = {
    "expression": 0.25,    # Visual emotion (key for thumbnails)
    "landmark": 0.15,      # Head movement, gestures
    "audio": 0.2,          # Music, excitement
    "speech": 0.15,        # Vocal emphasis
    "audio_score": 0.25,   # Comprehensive audio analysis (NEW)
}

pace = calculate_pace_score(
    expression_delta=expr_delta,
    landmark_motion=motion,
    audio_energy_delta=audio_delta,
    speech_emotion_delta=speech_delta,
    audio_score=audio_score_value,  # From timeline segments
    weights=weights,  # Custom weights
)
```

**Audio Scoring Components:**

```python
from app.audio_media import calculate_audio_score

# Calculate audio score for a timeline segment
audio_scores = calculate_audio_score(
    segment=segment_event,  # Timeline segment with text, start_ms, end_ms
    audio_features=audio_features,  # From analyze_audio_features()
)

# Returns:
# {
#   "speech_gate": 1.0,         # Binary: speech detected
#   "text_importance": 0.7,     # Categorical: emphasis/contrast
#   "emphasis_score": 0.82,     # Energy deviation from baseline
#   "bgm_penalty": 0.95,        # Low music interference
#   "audio_score": 0.55         # Final score (product of all)
# }
```
