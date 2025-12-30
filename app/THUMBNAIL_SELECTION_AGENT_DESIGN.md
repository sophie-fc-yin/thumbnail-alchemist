# Thumbnail Selection Agent Design

**Purpose**: Intelligent agent that analyzes adaptive sampling results and creator preferences to select the optimal thumbnail frame that maximizes engagement and achieves creator goals.

**Version**: 1.1 (Added Editability Scoring)
**Last Updated**: 2025-12-29

---

## Table of Contents
- [Overview](#overview)
- [Agent Architecture](#agent-architecture)
- [Input Schema](#input-schema)
- [Selection Process](#selection-process)
- [Psychology Principles](#psychology-principles)
- [Scoring System](#scoring-system)
- [Implementation](#implementation)

---

## Overview

### Goal
Select the single best frame from adaptive sampling results that:
1. ✅ **Aligns with creator's brand** (channel profile, style, audience)
2. ✅ **Meets creative brief** (specific requirements, constraints, preferences)
3. ✅ **Achieves target metrics** (CTR, views, retention)
4. ✅ **High visual quality** (composition, clarity, lighting, expression)
5. ✅ **Leverages psychology** (curiosity, emotion, pattern interrupts)
6. ✅ **Maintains originality** (avoids clichés, stands out in niche)

### Process Flow
```
┌─────────────────────────────────────────────────────────────┐
│              INPUT: Creator Data + Timeline                  │
├─────────────────────────────────────────────────────────────┤
│  • Creative Brief (Supabase)                                 │
│  • Channel Profile (Supabase)                                │
│  • Target Metrics (Supabase)                                 │
│  • Adaptive Sampling JSON (GCS)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         PHASE 1: Moment Candidate Selection (10 moments)    │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Temporal Clustering (1.0s window)                   │
│  → Group frames within 1 second into clusters                │
│  → Select BEST frame from each cluster                       │
│     (highest expression + quality)                           │
│                                                              │
│  Step 2: Score each unique moment:                           │
│  → Narrative importance (Stream A)                           │
│  → Perceptual impact (Stream B)                              │
│  → Visual quality (face expression)                          │
│  → Alignment with creative brief                             │
│                                                              │
│  Step 3: Visual Diversity Enforcement                        │
│  → Compute CLIP embeddings for all candidates                │
│  → Remove frames > 85% similar                               │
│  → Enforce minimum 2.0s gap between selections               │
│                                                              │
│  Step 4: Emotional/Vibe Diversity                            │
│  → Detect emotion (surprise, joy, serious, etc.)             │
│  → Detect narrative type (hook, peak, explanation)           │
│  → Max 2 frames per emotion/narrative combo                  │
│  → Ensure diverse psychological angles                       │
│                                                              │
│  → Select top 10 DIVERSE moments (visual + emotional)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         PHASE 2: Contextual Scoring (per frame)             │
├─────────────────────────────────────────────────────────────┤
│  For each candidate frame:                                   │
│  → Compute niche-specific scores:                            │
│     • Aesthetic quality (niche-adjusted criteria)            │
│     • Psychology triggers (goal-aligned priorities)          │
│     • Editability (workability for creators)                 │
│     • Face quality (expression, clarity)                     │
│     • Creator alignment (brand match)                        │
│     • Technical quality (sharpness, exposure)                │
│  → Weighted total score (niche-specific weights)             │
│  → Sort frames by total score                                │
│                                                              │
│  NOTE: Scores are for INTERNAL analysis only, not shown     │
│  to creators unless in debug mode.                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         PHASE 3: Risk Advisory (VLM decision support)       │
├─────────────────────────────────────────────────────────────┤
│  → VLM analyzes ALL frames visually                          │
│  → Provides THREE strategic recommendations:                 │
│     1. SAFE / DEFENSIBLE                                     │
│        Low-regret, clear emotion, good baseline              │
│     2. HIGH-VARIANCE / BOLD                                  │
│        Standout potential with creative risk                 │
│     3. AVOID / COMMON PITFALL                                │
│        Tempting but often underperforms                      │
│  → Plain-English explanations (no scores in creator view)    │
│  → Debug data includes all scores for analysis               │
│                                                              │
│  PHILOSOPHY: Decision relief, not decision-making            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         OUTPUT: 3 Strategic Options + Debug Data            │
├─────────────────────────────────────────────────────────────┤
│  CREATOR-FACING OUTPUT:                                      │
│  • safe: { frame, why, trade-offs }                          │
│  • high_variance: { frame, why, risks }                      │
│  • avoid: { frame, pitfall_explanation }                     │
│  • meta: { confidence, user_control_note }                   │
│                                                              │
│  DEBUG OUTPUT (for developers):                              │
│  • all_frames_scored: [ {scores, features, notes}, ... ]     │
│  • scoring_notes: "How scores influenced choices"            │
│                                                              │
│  Creator makes final decision with confidence.               │
└─────────────────────────────────────────────────────────────┘
```

---

## Input Schema

### 1. Creative Brief (from Supabase)
```typescript
interface CreativeBrief {
  project_id: string;

  // Core Requirements
  video_title: string;
  video_description: string;
  primary_message: string;  // What's the key takeaway?
  target_emotion: string;   // "curiosity", "excitement", "shock", etc.

  // Visual Preferences
  preferred_style: "expressive" | "calm" | "dynamic" | "professional";
  face_preference: "extreme_emotion" | "moderate" | "neutral" | "no_preference";
  color_preference?: string[];  // ["vibrant", "warm", "cool", "high_contrast"]

  // Content Constraints
  avoid_moments?: string[];  // Timestamps or types to avoid
  must_include?: string[];   // Required elements (e.g., "product", "text_overlay")
  text_overlay_planned: boolean;

  // Brand Guidelines
  brand_colors?: string[];   // Hex colors
  brand_style?: string;      // "minimalist", "bold", "playful", etc.
}
```

### 2. Channel Profile (from Supabase)
```typescript
interface ChannelProfile {
  channel_id: string;

  // Channel Identity
  niche: string;  // "tech reviews", "cooking", "gaming", etc.
  content_type: "educational" | "entertainment" | "lifestyle" | "gaming" | "commentary";
  audience_age: "13-17" | "18-24" | "25-34" | "35-44" | "45+";
  audience_geography: string[];  // ["US", "UK", "Global"]

  // Performance Data
  avg_ctr: number;           // Average click-through rate
  top_performing_thumbnails: string[];  // URLs to reference
  audience_retention_pattern: "hook_driven" | "story_driven" | "personality_driven";

  // Channel Brand
  personality: string[];     // ["energetic", "informative", "calm", "humorous"]
  visual_consistency: "high" | "medium" | "low";
  thumbnail_patterns?: {     // Patterns in successful thumbnails
    common_emotions?: string[];
    common_compositions?: string[];
    text_usage?: "heavy" | "moderate" | "minimal";
  };
}
```

### 3. Target Metrics (from Supabase)
```typescript
interface TargetMetrics {
  primary_goal: "ctr" | "views" | "retention" | "engagement";
  target_ctr?: number;       // e.g., 0.12 (12%)
  target_views?: number;

  // Competition
  competitor_thumbnails?: string[];  // URLs for differentiation analysis
  niche_saturation: "low" | "medium" | "high";

  // A/B Testing
  ab_test_enabled: boolean;
  num_variants?: number;     // How many thumbnails to generate
}
```

### 4. Adaptive Sampling JSON
```typescript
// Already have this from previous implementation
interface AdaptiveSamplingAnalysis {
  merged_timeline: MomentCandidate[];
  extracted_frames: ExtractedFrame[];
  stream_a: StreamAResults;
  stream_b: StreamBResults;
  visual_analysis: VisualAnalysis;
  // ... (full schema from previous design)
}
```

---

## Selection Process

### Phase 1: Moment Candidate Selection

**Goal**: Narrow down from all moments to ~10 best candidates with temporal + visual diversity

**Step 1A: Temporal Clustering**

Group frames into temporal clusters to avoid picking multiple frames from the same moment:

```python
def cluster_frames_temporally(extracted_frames, time_window=1.0):
    """
    Group frames within time_window seconds into clusters.

    Args:
        time_window: Seconds to consider same moment (default: 1.0s)

    Returns:
        List of frame clusters: [[frame1, frame2], [frame3], ...]
    """
    clusters = []
    sorted_frames = sorted(extracted_frames, key=lambda x: x["timestamp"])

    current_cluster = [sorted_frames[0]]

    for frame in sorted_frames[1:]:
        # If frame is within time_window of cluster start, add to cluster
        if frame["timestamp"] - current_cluster[0]["timestamp"] <= time_window:
            current_cluster.append(frame)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [frame]

    clusters.append(current_cluster)
    return clusters
```

**Example**:
```
Input frames:
- frame_5000ms.jpg (5.0s)
- frame_5200ms.jpg (5.2s)  ← Same exciting moment
- frame_5500ms.jpg (5.5s)  ← Same exciting moment
- frame_8700ms.jpg (8.7s)  ← Different moment

Output clusters:
- Cluster 1: [5.0s, 5.2s, 5.5s]  ← Pick BEST from this cluster
- Cluster 2: [8.7s]
```

**Step 1B: Best-of-Cluster Selection**

For each temporal cluster, select the single best frame:

```python
def select_best_from_cluster(cluster_frames):
    """
    Pick the highest-quality frame from a temporal cluster.

    Scoring (within cluster):
    - expression_intensity: 40%
    - moment_score: 30%
    - face_quality: 20%
    - technical_quality: 10%
    """
    best_frame = max(cluster_frames, key=lambda f: (
        0.40 * f.get("expression_intensity", 0) +
        0.30 * f.get("moment_score", 0) +
        0.20 * f.get("face_quality", 0) +
        0.10 * f.get("technical_quality", 0)
    ))
    return best_frame
```

**Step 1C: Moment Scoring**

Score each cluster's best frame:

```python
moment_score = (
    0.30 * narrative_importance +    # Stream A score
    0.25 * perceptual_impact +       # Stream B score
    0.20 * visual_quality +          # Face expression intensity
    0.15 * creative_brief_match +    # Alignment with brief
    0.10 * uniqueness                # Avoids clichés
)
```

**Step 1D: Visual Diversity Enforcement**

After scoring, ensure top 10 candidates are visually distinct:

```python
def enforce_visual_diversity(candidate_frames, similarity_threshold=0.85, top_n=10):
    """
    Remove visually similar frames to ensure diverse candidates.

    Uses perceptual hashing (pHash) or CLIP embeddings for similarity.

    Args:
        similarity_threshold: Max cosine similarity allowed (0.85 = 85% similar)
        top_n: Number of diverse candidates to return

    Returns:
        List of visually diverse frames
    """
    selected = []
    candidates_sorted = sorted(candidate_frames, key=lambda x: x["score"], reverse=True)

    for candidate in candidates_sorted:
        # Download frame and compute embedding
        frame_embedding = compute_clip_embedding(candidate["frame_path"])

        # Check similarity to already selected frames
        is_duplicate = False
        for selected_frame in selected:
            selected_embedding = selected_frame["embedding"]
            similarity = cosine_similarity(frame_embedding, selected_embedding)

            if similarity > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            candidate["embedding"] = frame_embedding
            selected.append(candidate)

        if len(selected) >= top_n:
            break

    return selected
```

**Visual Similarity Methods**:

1. **Perceptual Hashing (Fast)**:
   ```python
   import imagehash
   from PIL import Image

   hash1 = imagehash.phash(Image.open(frame1))
   hash2 = imagehash.phash(Image.open(frame2))
   similarity = 1 - (hash1 - hash2) / 64.0  # Hamming distance
   ```

2. **CLIP Embeddings (More Accurate)**:
   ```python
   from transformers import CLIPProcessor, CLIPModel

   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

   inputs = processor(images=[frame1, frame2], return_tensors="pt")
   embeddings = model.get_image_features(**inputs)
   similarity = cosine_similarity(embeddings[0], embeddings[1])
   ```

**Step 1E: Emotional/Vibe Diversity Enforcement**

Ensure top 10 represent different emotional angles and narrative strategies:

```python
def enforce_emotional_diversity(candidate_frames, max_per_emotion=2):
    """
    Ensure diverse emotional/narrative vibes in final candidates.

    Different emotions create different thumbnail strategies:
    - Surprise: Curiosity trigger (high CTR)
    - Joy/Excitement: Positive emotion (broad appeal)
    - Serious/Focused: Authority/expertise
    - Confusion: Relatability ("I wondered this too!")
    - Triumph: Aspiration ("I want that result")
    - Contemplative: Depth/thoughtfulness

    Args:
        max_per_emotion: Max frames per emotion type (default: 2)
    """
    emotion_counts = {}
    selected = []

    for candidate in sorted(candidate_frames, key=lambda x: x["score"], reverse=True):
        emotion = candidate["dominant_emotion"]
        narrative_type = candidate.get("narrative_type", "unknown")

        # Create vibe signature (emotion + narrative type)
        vibe = f"{emotion}_{narrative_type}"

        # Count this vibe
        count = emotion_counts.get(vibe, 0)

        # Allow if under limit
        if count < max_per_emotion:
            selected.append(candidate)
            emotion_counts[vibe] = count + 1

        if len(selected) >= 10:
            break

    return selected
```

**Emotional Diversity Categories**:

1. **Surprise/Shock** (curiosity trigger)
   - Stream A: unexpected reveals, plot twists
   - Stream B: sudden sound effects, silence→impact
   - Psychology: Curiosity gap, FOMO

2. **Joy/Excitement** (positive emotion)
   - Stream A: celebrations, "it worked!", discoveries
   - Stream B: energy peaks, music drops
   - Psychology: Emotional contagion, aspirational

3. **Serious/Focused** (authority/expertise)
   - Stream A: explanations, deep analysis moments
   - Stream B: calm, concentrated audio
   - Psychology: Trust, credibility

4. **Confusion/Questioning** (relatability)
   - Stream A: "wait, what?", problems identified
   - Stream B: uncertain moments
   - Psychology: "Me too!", shared struggle

5. **Triumph/Satisfaction** (aspiration)
   - Stream A: solutions found, victories
   - Stream B: climactic moments
   - Psychology: Desired outcome, transformation

6. **Contemplative/Thoughtful** (depth)
   - Stream A: reflection, insights
   - Stream B: quiet, deliberate moments
   - Psychology: Intelligence, nuance

**Example Diverse Selection**:

```json
{
  "selected_candidates": [
    {
      "rank": 1,
      "emotion": "surprise",
      "narrative_type": "hook_statement",
      "vibe": "curiosity trigger",
      "score": 0.94,
      "why": "Highest CTR potential, classic shock/surprise"
    },
    {
      "rank": 2,
      "emotion": "joy",
      "narrative_type": "emotional_peak",
      "vibe": "positive emotion",
      "score": 0.91,
      "why": "Broad appeal, positive contagion"
    },
    {
      "rank": 3,
      "emotion": "serious",
      "narrative_type": "explanation",
      "vibe": "authority",
      "score": 0.88,
      "why": "Expertise signal, professional angle"
    },
    {
      "rank": 4,
      "emotion": "confusion",
      "narrative_type": "problem",
      "vibe": "relatability",
      "score": 0.86,
      "why": "Viewer sees themselves, 'I wondered that too!'"
    },
    {
      "rank": 5,
      "emotion": "triumph",
      "narrative_type": "story_beat",
      "vibe": "aspiration",
      "score": 0.84,
      "why": "Transformation promise, desired outcome"
    },
    // ... 5 more diverse vibes
  ],

  "diversity_analysis": {
    "emotion_distribution": {
      "surprise": 2,
      "joy": 2,
      "serious": 2,
      "confusion": 1,
      "triumph": 2,
      "contemplative": 1
    },
    "narrative_distribution": {
      "hook_statement": 3,
      "emotional_peak": 2,
      "story_beat": 2,
      "explanation": 2,
      "reveal": 1
    },
    "vibe_coverage": "High - 10 distinct psychological angles"
  }
}
```

**Strategic Vibe Selection**:

Different vibes work for different goals:

| Vibe | Best For | CTR Potential | Niche |
|------|----------|---------------|-------|
| Surprise | Curiosity-driven clicks | ⭐⭐⭐⭐⭐ | Entertainment, reveals |
| Joy | Broad appeal, positive emotion | ⭐⭐⭐⭐ | Lifestyle, success stories |
| Serious | Authority, educational | ⭐⭐⭐ | Tech, tutorials, science |
| Confusion | Relatability, "me too" | ⭐⭐⭐⭐ | Problem-solving, guides |
| Triumph | Aspiration, results | ⭐⭐⭐⭐ | Fitness, productivity |
| Contemplative | Depth, intelligence | ⭐⭐⭐ | Commentary, analysis |

**Contrast Strategy**:

The agent can recommend based on **title contrast**:

- **Title is question/problem** → Use "triumph" or "surprise" thumbnail (shows solution/answer)
- **Title is surprising claim** → Use "serious" thumbnail (credibility contrast)
- **Title is positive** → Use "contemplative" or "serious" thumbnail (depth contrast)
- **Title is clickbait-y** → Use "serious" thumbnail (trustworthy contrast)

**Diversity Rules** (updated):
- ✅ Visual diversity: Frames > 85% similar → skip
- ✅ Temporal diversity: Minimum 2.0s gap between selections
- ✅ Emotional diversity: Max 2 frames per emotion/narrative combo
- ✅ Compositional diversity: Max 2 frames from same scene
- ✅ Strategic variety: Different psychological angles for different approaches

**Output**: Top 10 candidates with diverse vibes, each offering a different psychological strategy

**Selection Criteria**:

1. **Narrative Importance** (Stream A):
   - Hook statements: +0.3
   - Emotional peaks: +0.25
   - Story beats: +0.2
   - Matches `primary_message`: +0.15

2. **Perceptual Impact** (Stream B):
   - Energy peaks: +0.2
   - Silence-to-impact: +0.25 (dramatic!)
   - Spectral changes: +0.15
   - Matches `target_emotion`: +0.2

3. **Visual Quality**:
   - `expression_intensity > 0.7`: +0.3
   - `has_face == true`: +0.2
   - `eye_openness > 0.5`: +0.15 (engaged, alert)
   - Matches `face_preference`: +0.2

4. **Creative Brief Match**:
   - Moment type aligns with `preferred_style`
   - Not in `avoid_moments` list
   - Contains `must_include` elements (if specified)

5. **Uniqueness**:
   - Compare to `top_performing_thumbnails` embeddings
   - Penalize if too similar to existing successful thumbnails
   - Reward novelty in niche

**Output**: Top 10 moment candidates ranked by score

---

### Phase 2: Frame-Level Visual Analysis

**Goal**: Analyze each frame in the 10 moments for visual quality and psychology

For each frame, run comprehensive visual analysis:

#### A. Aesthetic Quality Analysis

**Goal**: Measure overall visual appeal and professional quality

```python
aesthetic_score = compute_aesthetic_quality(frame)

# Using specialized aesthetic models:
# 1. LAION Aesthetics Predictor v2 (most popular)
# 2. ImageReward (CLIP-based quality)
# 3. NIMA (Neural Image Assessment)

# Returns score 0.0-1.0 where:
# - 0.0-0.4: Poor quality (blurry, bad lighting, amateur)
# - 0.4-0.6: Average quality (acceptable but unremarkable)
# - 0.6-0.8: Good quality (professional, appealing)
# - 0.8-1.0: Excellent quality (stunning, gallery-worthy)
```

**What Aesthetic Models Measure**:
- Color harmony and grading
- Lighting quality (exposure, shadows, highlights)
- Sharpness and clarity
- Composition balance
- Visual appeal and "pro" look
- Absence of artifacts/noise
- Overall "beauty" score

**Example Results**:
```python
{
    "aesthetic_score": 0.75,  # Good quality
    "components": {
        "color_harmony": 0.82,
        "lighting": 0.71,
        "sharpness": 0.79,
        "composition": 0.68,
        "professional_look": 0.76
    },
    "interpretation": "Professional quality with good color grading and lighting"
}
```

#### B. Face Quality Analysis
```python
face_quality_score = (
    0.30 * expression_strength +     # How expressive?
    0.25 * eye_contact_quality +     # Looking at camera?
    0.20 * facial_clarity +          # Sharp, well-lit face?
    0.15 * emotion_authenticity +    # Genuine emotion?
    0.10 * composition_quality       # Face placement, framing
)
```

**Metrics**:
- **Expression Strength**: Use FER+ emotion probabilities
  - Strong emotions (surprise, happiness, shock): high score
  - Neutral/subtle: low score
  - Match to `target_emotion` from brief

- **Eye Contact**: Analyze gaze direction
  - Direct camera gaze: +0.3 (creates connection)
  - Slightly off-camera: +0.15 (natural)
  - Looking away: -0.2 (disengagement)

- **Facial Clarity**:
  - Face size (% of frame): 15-40% optimal
  - Sharpness: Laplacian variance > threshold
  - Lighting: Histogram analysis (avoid over/underexposed)

- **Emotion Authenticity**:
  - Micro-expression analysis (slight asymmetry = genuine)
  - Matches audio/context (congruent emotion)

- **Composition**:
  - Rule of thirds: face on intersection points
  - Headroom: 10-20% optimal
  - Background: not cluttered (saliency map)

#### B. Composition Analysis
```python
composition_score = (
    0.25 * rule_of_thirds_alignment +
    0.25 * contrast_strength +
    0.20 * color_harmony +
    0.15 * visual_balance +
    0.15 * background_cleanliness
)
```

**Metrics**:
- **Rule of Thirds**: Place key elements on grid intersections
- **Contrast**: High contrast = attention-grabbing (measure LAB color space)
- **Color Harmony**: Complementary colors, avoid clashing
  - Bonus if matches `brand_colors`
- **Visual Balance**: Symmetry vs. asymmetry (context-dependent)
- **Background**: Blur/bokeh = face pops, clean = professional

#### C. Technical Quality
```python
technical_quality = (
    0.40 * sharpness +               # Laplacian variance
    0.30 * noise_level +             # Low noise = high quality
    0.30 * exposure_quality          # Histogram analysis
)
```

#### D. Psychology Triggers
```python
psychology_score = sum([
    trigger_scores[trigger]
    for trigger in detected_triggers
])
```

**Detected Triggers**:

1. **Curiosity Gap** (+0.25):
   - Incomplete action (pointing, looking at something off-screen)
   - Question-like expression (raised eyebrows)
   - Unexpected element in frame

2. **Emotional Contagion** (+0.25):
   - Strong positive emotion (smile, excitement)
   - Strong negative emotion (shock, fear) - if appropriate for niche
   - Authentic expression

3. **Pattern Interrupt** (+0.2):
   - Unusual composition
   - Unexpected color palette
   - Breaks niche conventions (if `niche_saturation == "high"`)

4. **Social Proof** (+0.15):
   - Face looking directly at viewer (parasocial connection)
   - Open body language
   - Confident expression

5. **Scarcity/Urgency** (+0.15):
   - Mid-action shot (dynamic, moment in time)
   - Intense expression (something important happening)

6. **Before/After Implication** (+0.1):
   - Reaction shot (implies something happened)
   - Transformation visual cue

---

### Phase 3: Multi-Criteria Ranking

**Final Score** (with editability):
```python
final_score = (
    0.22 * creator_alignment_score +   # Matches brief + brand + goals
    0.18 * aesthetic_score +            # Overall visual appeal
    0.18 * psychology_score +           # Engagement triggers
    0.15 * editability_score +          # ⭐ NEW: Workability for creators
    0.13 * face_quality_score +         # Expression + emotion
    0.08 * originality_score +          # Stands out in niche
    0.04 * composition_score +          # Rule of thirds, balance
    0.02 * technical_quality_score      # Sharpness, noise, exposure
)
```

**Weight Justification**:
- **22% Creator Alignment**: Most important - must match what creator wants
- **18% Aesthetic Quality**: High visual appeal = professional look, stands out
- **18% Psychology**: Engagement triggers drive clicks
- **15% Editability**: Can creators crop, zoom, add text without losing impact?
- **13% Face Quality**: Face is primary element in most thumbnails
- **8% Originality**: Important in saturated niches
- **4% Composition**: Supporting factor (included in aesthetics)
- **2% Technical**: Minimum quality bar (rarely differentiator)

#### Editability Score (NEW - v1.1)

**Critical Question**: "Can this frame survive being cropped, zoomed, simplified, and still communicate emotion instantly?"

Creators need frames they can actually work with. A beautiful frame is useless if:
- Cropping removes the key emotion
- No space for title text without covering the subject
- Zooming loses context or emotional clarity
- Compression/filters destroy the emotional impact

```python
editability_score = (
    0.15 * crop_resilience +           # Can crop 30-40% and still work?
    0.15 * zoom_potential +             # Can zoom 1.5-2x without losing emotion?
    0.30 * text_overlay_space +         # Clear space for title text? ⭐ Critical
    0.30 * emotion_resilience +         # Emotion survives simplification? ⭐ Critical
    0.10 * composition_flexibility      # Multiple cropping options?
)
```

**Components**:

1. **Crop Resilience** (15%): Can crop 30-40% and still have good composition?
2. **Zoom Potential** (15%): Can zoom 1.5-2x without losing emotional context?
3. **Text Overlay Space** (30%): Clear areas for title without obscuring subject?
4. **Emotion Resilience** (30%): Will emotion survive compression/filters?
5. **Composition Flexibility** (10%): Multiple valid crop options available?

**Why Editability Matters**:
- 90% of creators add text overlays to thumbnails
- Mobile viewers see tiny thumbnails (cropping common)
- Platform compression destroys subtle emotions
- Creators need flexibility for A/B testing variants

**Adjustable by Niche**:
```python
# For beauty/lifestyle channels (aesthetics matter more, less text)
weights = {
    "creator_alignment": 0.18,
    "aesthetic": 0.28,  # ⬆️ Higher
    "psychology": 0.18,
    "editability": 0.13,  # ⬇️ Lower (less text overlays)
    "face_quality": 0.13,
    "originality": 0.06,
    "composition": 0.03,
    "technical": 0.01
}

# For tech/educational channels (text overlays critical)
weights = {
    "creator_alignment": 0.22,
    "aesthetic": 0.13,  # ⬇️ Lower
    "psychology": 0.22,  # ⬆️ Higher
    "editability": 0.18,  # ⬆️ Higher (text overlays critical)
    "face_quality": 0.13,
    "originality": 0.09,
    "composition": 0.02,
    "technical": 0.01
}

# For gaming channels (highest editability need)
weights = {
    "creator_alignment": 0.19,
    "aesthetic": 0.15,
    "psychology": 0.19,
    "editability": 0.20,  # ⬆️ Highest (heavy text overlays)
    "face_quality": 0.15,
    "originality": 0.09,
    "composition": 0.02,
    "technical": 0.01
}
```

---

### Contextual Scoring: Context-Aware Aesthetic & Psychology Evaluation

**Problem**: Generic aesthetic and psychology scoring doesn't account for niche-specific criteria and creator goals.

**Solution**: Make scoring components **contextually aware** - adapt evaluation criteria based on channel niche and creative brief.

#### What Makes "Good Aesthetics" Different by Niche

Same frame can be:
- **Excellent** for beauty channel (soft lighting, warm tones, polished)
- **Poor** for tech channel (needs bright, sharp, clear product visibility)
- **Average** for gaming channel (needs high saturation, dramatic lighting, energy)

#### Niche-Specific Aesthetic Criteria

Each niche has different aesthetic priorities:

**Beauty/Lifestyle Channels**:
```python
aesthetic_criteria = {
    "lighting": {
        "preferred": ["soft", "warm", "golden_hour", "ring_light"],
        "avoid": ["harsh", "overhead", "fluorescent"],
        "weight": 0.25  # Critical for beauty
    },
    "color_palette": {
        "preferred": ["pastel", "warm_tones", "complementary"],
        "avoid": ["oversaturated", "clashing", "muddy"],
        "weight": 0.25
    },
    "polish_level": {
        "preferred": ["professional", "edited", "magazine_quality"],
        "avoid": ["raw", "unedited", "amateur"],
        "weight": 0.20
    }
}
```

**Tech/Educational Channels**:
```python
aesthetic_criteria = {
    "clarity": {
        "preferred": ["sharp", "readable_text", "clear_details"],
        "avoid": ["blurry", "unreadable", "soft_focus"],
        "weight": 0.30  # ⬆️ Highest priority for tech
    },
    "lighting": {
        "preferred": ["bright", "even", "studio_lighting"],
        "avoid": ["dim", "uneven", "overly_artistic"],
        "weight": 0.15
    },
    "composition": {
        "preferred": ["tech_visible", "clear_subject", "uncluttered"],
        "avoid": ["obscured_product", "unclear_focus"],
        "weight": 0.25
    }
}
```

**Gaming Channels**:
```python
aesthetic_criteria = {
    "color_palette": {
        "preferred": ["saturated", "bold", "neon", "RGB"],
        "avoid": ["desaturated", "muted", "monotone"],
        "weight": 0.25  # ⬆️ Bold colors critical
    },
    "energy_level": {
        "preferred": ["high_energy", "intense", "exciting"],
        "avoid": ["calm", "subdued", "low_energy"],
        "weight": 0.25  # ⬆️ Energy is key
    },
    "composition": {
        "preferred": ["dynamic", "action_focused", "cinematic"],
        "avoid": ["static", "boring", "passive"],
        "weight": 0.25
    }
}
```

#### Niche-Specific Psychology Triggers

Different niches respond to different psychological triggers:

**Beauty/Lifestyle**: Aspiration (30%), Emotional Contagion (25%), Curiosity (20%)
**Tech/Educational**: Curiosity Gap (35%), Authority (25%), Clarity (20%)
**Gaming**: Excitement (30%), Surprise (25%), Triumph (20%)
**Commentary**: Authenticity (30%), Relatability (25%), Controversy (20%)

#### Goal-Driven Psychology Prioritization

Same triggers, different priorities based on creator goal:

**Goal: Maximize CTR**
- Priority triggers: curiosity_gap, pattern_interrupt, surprise, fomo
- Boost weights: curiosity +30%, surprise +25%

**Goal: Grow Subscribers**
- Priority triggers: authority, authenticity, relatability, social_proof
- Boost weights: authority +30%, authenticity +25%

**Goal: Brand Building**
- Priority triggers: authenticity, aspiration, authority, emotional_contagion
- Boost weights: authenticity +30%, aspiration +25%

#### Tone Adjustments

Creator's tone preference further refines criteria:

**Professional Tone**:
- Aesthetic boost: polish_level, clarity, clean_background
- Psychology boost: authority, clarity, utility
- Aesthetic penalty: raw, unpolished, casual

**Casual Tone**:
- Aesthetic boost: natural, authentic, relatable
- Psychology boost: authenticity, relatability, emotional_contagion
- Aesthetic penalty: overly_polished, corporate

**Energetic Tone**:
- Aesthetic boost: vibrant, high_contrast, dynamic
- Psychology boost: excitement, surprise, fomo
- Aesthetic penalty: subdued, calm, muted

#### Contextual Scoring Integration

```python
# 1. Get niche + brief specific criteria
aesthetic_criteria = get_aesthetic_criteria(channel_profile, creative_brief)
psychology_priorities = get_psychology_priorities(channel_profile, creative_brief)

# 2. Evaluate frame against contextual criteria
aesthetic_score = evaluate_aesthetic_alignment(
    frame_features,  # Detected: lighting, colors, composition
    aesthetic_criteria  # Niche + tone adjusted criteria
)

psychology_score = evaluate_psychology_alignment(
    detected_triggers,  # Detected: curiosity_gap, authority, etc.
    psychology_priorities  # Goal-prioritized triggers
)

# 3. Use contextual scores in final weighted calculation
final_score = (
    0.25 * creator_alignment +
    0.20 * aesthetic_score +      # ← Contextually evaluated
    0.20 * psychology_score +     # ← Contextually evaluated
    0.15 * face_quality +
    0.10 * originality +
    0.05 * composition +
    0.05 * technical_quality
)
```

**Example: Same Frame, Different Scores**

Frame: Soft lighting, warm tones, gentle smile

**Beauty Channel** (professional tone, brand building goal):
- Aesthetic: 0.91 ✅ (matches soft + warm + polished criteria)
- Psychology: 0.88 ✅ (aspiration + emotional_contagion align with brand goal)
- **Final Score: 0.89**

**Tech Channel** (professional tone, maximize CTR goal):
- Aesthetic: 0.62 ⚠️ (soft focus doesn't meet clarity requirement 30%)
- Psychology: 0.65 ⚠️ (missing curiosity_gap, top priority for CTR)
- **Final Score: 0.68**

**Result**: Same frame scores 30% higher for beauty channel due to contextual evaluation.

**Implementation**: See `app/thumbnail_agent/contextual_scoring.py` and `contextual_scoring_examples.py`

---

#### Creator Alignment Score
```python
creator_alignment = (
    0.40 * creative_brief_match +
    0.35 * channel_brand_match +
    0.25 * target_metrics_fit
)
```

**Creative Brief Match**:
- Does `target_emotion` match detected emotion?
- Does `preferred_style` match frame style?
- Does `face_preference` match expression level?
- Are `must_include` elements present?
- Are `avoid_moments` respected?

**Channel Brand Match**:
- Emotion aligns with `channel.personality`?
- Composition matches `thumbnail_patterns`?
- Fits `niche` conventions (or intentionally breaks them)?
- Consistent with `top_performing_thumbnails` style?

**Target Metrics Fit**:
- If `primary_goal == "ctr"`: prioritize curiosity triggers
- If `primary_goal == "retention"`: prioritize story/narrative moments
- If `primary_goal == "views"`: prioritize broad appeal
- If `primary_goal == "engagement"`: prioritize emotional intensity

#### Originality Score
```python
originality = 1.0 - similarity_to_existing_thumbnails

# Where similarity is calculated using:
# - CLIP embeddings of frame vs. top_performing_thumbnails
# - CLIP embeddings vs. competitor_thumbnails
# - High similarity = low originality
```

**Originality Boost**:
- If `niche_saturation == "high"`: +0.2 for high originality
- If `visual_consistency == "low"`: allow more experimental frames
- Avoid exact duplicates of successful thumbnails (diminishing returns)

#### Constraints & Filters

Before final ranking, apply hard constraints:

1. **Quality Minimum**:
   - `technical_quality_score >= 0.6`
   - `has_face == true` (unless `face_preference == "no_preference"`)
   - `sharpness > threshold`

2. **Brand Guidelines**:
   - If `brand_colors` specified, frame must contain at least one
   - If `brand_style` is "professional", avoid extreme expressions

3. **Content Safety**:
   - No inappropriate content
   - No misleading frames (clickbait detection)

---

## Psychology Principles

### Core Principles Applied:

1. **Curiosity Gap** (Loewenstein, 1994):
   - Show incomplete information
   - Raise questions visually
   - Create information gap that title/description will fill

2. **Emotional Contagion** (Hatfield et al., 1993):
   - Authentic facial expressions trigger mirror neurons
   - Positive emotions = higher engagement (for most niches)
   - Strong emotions > neutral

3. **Pattern Interrupts**:
   - Break expected visual patterns in saturated niches
   - Novel compositions stand out in feed
   - Balance: novel but not alienating

4. **Social Proof & Parasocial Relationships**:
   - Direct eye contact = connection
   - Familiar face = trust (for established creators)
   - Expression matches audience expectations

5. **Scarcity & FOMO**:
   - Mid-action = "moment in time"
   - Dynamic poses > static
   - Implication of unique event

### Niche-Specific Psychology:

- **Educational**: Curiosity + authority (confident expression)
- **Entertainment**: Extreme emotion + novelty
- **Gaming**: Action + excitement
- **Lifestyle**: Aspiration + relatability
- **Commentary**: Strong opinion visual (raised eyebrow, direct gaze)

---

## Scoring System

### Example Score Breakdown:

```json
{
  "frame_path": "gs://.../frame_5200ms.jpg",
  "timestamp": 5.2,
  "final_score": 0.92,

  "score_breakdown": {
    "creator_alignment": {
      "score": 0.92,
      "weight": 0.25,
      "contribution": 0.23,
      "components": {
        "creative_brief_match": 0.95,  // Perfect emotion match
        "channel_brand_match": 0.88,   // Consistent with channel
        "target_metrics_fit": 0.93     // High CTR potential
      }
    },
    "aesthetic_quality": {
      "score": 0.84,
      "weight": 0.20,
      "contribution": 0.168,
      "model": "LAION Aesthetics v2",
      "components": {
        "color_harmony": 0.88,
        "lighting": 0.82,
        "sharpness": 0.87,
        "composition_balance": 0.79,
        "professional_look": 0.86
      },
      "interpretation": "High-quality professional look with excellent color grading"
    },
    "psychology": {
      "score": 0.96,
      "weight": 0.20,
      "contribution": 0.192,
      "triggers": ["curiosity_gap", "emotional_contagion", "direct_gaze"],
      "trigger_scores": {
        "curiosity_gap": 0.25,
        "emotional_contagion": 0.25,
        "direct_gaze": 0.20,
        "pattern_interrupt": 0.15,
        "before_after": 0.11
      }
    },
    "face_quality": {
      "score": 0.92,
      "weight": 0.15,
      "contribution": 0.138,
      "components": {
        "expression_strength": 0.94,
        "eye_contact": 0.91,
        "facial_clarity": 0.89,
        "emotion_authenticity": 0.93
      }
    },
    "originality": {
      "score": 0.78,
      "weight": 0.10,
      "contribution": 0.078,
      "similar_to_top_performers": 0.22,  // Low similarity = good
      "novel_elements": ["unusual_angle", "color_grading"]
    },
    "composition": {
      "score": 0.85,
      "weight": 0.05,
      "contribution": 0.0425,
      "components": {
        "rule_of_thirds": 0.88,
        "visual_balance": 0.82,
        "background_cleanliness": 0.86
      }
    },
    "technical_quality": {
      "score": 0.91,
      "weight": 0.05,
      "contribution": 0.0455,
      "sharpness": 0.94,
      "noise_level": 0.88,
      "exposure": 0.91
    }
  },

  "total_score_calculation": {
    "creator_alignment": "0.92 × 0.25 = 0.230",
    "aesthetic_quality": "0.84 × 0.20 = 0.168",
    "psychology": "0.96 × 0.20 = 0.192",
    "face_quality": "0.92 × 0.15 = 0.138",
    "originality": "0.78 × 0.10 = 0.078",
    "composition": "0.85 × 0.05 = 0.043",
    "technical": "0.91 × 0.05 = 0.046",
    "final_score": "0.895 ≈ 0.90"
  },

  "reasoning": "Selected for strong curiosity trigger (raised eyebrows, pointing gesture) combined with authentic surprise emotion that matches creator's target_emotion. High contrast and direct eye contact create immediate connection. Novel composition (off-center framing) provides pattern interrupt in saturated tech review niche while maintaining brand consistency.",

  "psychology_analysis": {
    "primary_trigger": "curiosity_gap",
    "secondary_triggers": ["emotional_contagion", "direct_gaze"],
    "emotion_detected": "surprise",
    "emotion_strength": 0.92,
    "viewer_prediction": "High CTR (est. 14-16%) due to strong curiosity trigger + authentic emotion"
  },

  "alternatives": [
    {
      "rank": 2,
      "frame_path": "gs://.../frame_8700ms.jpg",
      "final_score": 0.89,
      "reason": "High visual quality but slightly lower psychology score"
    },
    {
      "rank": 3,
      "frame_path": "gs://.../frame_12300ms.jpg",
      "final_score": 0.86,
      "reason": "Good originality but lower emotion intensity"
    }
  ]
}
```

---

## Implementation

### Module Structure

```
app/
├── thumbnail_agent/
│   ├── __init__.py
│   ├── agent.py                    ← Main agent orchestrator
│   ├── moment_selector.py          ← Phase 1: Moment selection
│   ├── frame_analyzer.py           ← Phase 2: Frame analysis
│   ├── visual_quality.py           ← Visual quality metrics
│   ├── psychology_detector.py      ← Psychology trigger detection
│   ├── creator_alignment.py        ← Match to creator preferences
│   ├── originality_scorer.py       ← CLIP-based similarity
│   └── ranking.py                  ← Phase 3: Multi-criteria ranking
```

### API Endpoint

```python
@router.post("/thumbnail-agent/select")
async def select_best_thumbnail(
    project_id: str,
    creative_brief: CreativeBrief,
    channel_profile: ChannelProfile,
    target_metrics: TargetMetrics,
) -> ThumbnailSelectionResult:
    """
    Run thumbnail selection agent.

    Returns:
        Best frame + reasoning + alternatives
    """
```

### Key Dependencies

```toml
# Computer Vision
"opencv-python>=4.8.0"
"pillow>=10.0.0"

# Deep Learning
"torch>=2.0.0"
"torchvision>=0.15.0"
"transformers>=4.30.0"  # For CLIP embeddings

# Image Analysis
"scikit-image>=0.21.0"  # Sharpness, contrast, etc.

# LLM (for reasoning generation)
"openai>=1.0.0"  # GPT-4o for final reasoning
```

---

## Workflow Example

```python
# User creates project in Supabase
creative_brief = {
    "video_title": "I Tested the NEW iPhone 16 Camera",
    "primary_message": "iPhone 16 camera improvements are impressive",
    "target_emotion": "surprise",
    "preferred_style": "expressive",
    "face_preference": "extreme_emotion",
    "text_overlay_planned": True
}

channel_profile = {
    "niche": "tech reviews",
    "content_type": "educational",
    "avg_ctr": 0.11,
    "personality": ["energetic", "informative"],
    "thumbnail_patterns": {
        "common_emotions": ["surprise", "curiosity"],
        "text_usage": "heavy"
    }
}

target_metrics = {
    "primary_goal": "ctr",
    "target_ctr": 0.14,
    "niche_saturation": "high"
}

# Run adaptive sampling (already implemented)
sampling_result = await orchestrate_adaptive_sampling(...)

# Run thumbnail agent
result = await ThumbnailAgent.select_best(
    analysis_json_url=sampling_result["analysis_json_url"],
    creative_brief=creative_brief,
    channel_profile=channel_profile,
    target_metrics=target_metrics
)

# Output
{
    "selected_frame": "gs://.../frame_5200ms.jpg",
    "final_score": 0.94,
    "reasoning": "Strong surprise emotion matches creator's target...",
    "estimated_ctr": 0.15,
    "alternatives": [...]
}
```

---

## Next Steps

1. ✅ **Phase 1 Implementation**: Moment selection logic
2. ✅ **Phase 2 Implementation**: Frame-level visual analysis
3. ✅ **CLIP Integration**: Originality scoring via embeddings
4. ✅ **Psychology Detector**: Trigger detection algorithms
5. ✅ **LLM Integration**: GPT-4o for reasoning generation
6. ✅ **Supabase Schema**: Create tables for briefs/profiles/metrics
7. ✅ **API Endpoint**: Expose agent via FastAPI
8. ✅ **Testing**: Validate on real YouTube thumbnails

---

**Maintainer**: Claude Code
**Status**: Design Complete - Ready for Implementation
