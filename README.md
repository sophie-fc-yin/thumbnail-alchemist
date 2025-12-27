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
