# Thumbnail Agent Model Architecture

**Decision**: Which models/pipeline for thumbnail selection?

---

## Option 1: Single Unified VLM ❌ (Not Recommended)

### Approach:
Send all 10 frames + creator brief to one powerful Vision-Language Model

**Models**: GPT-4V, Claude 3.5 Sonnet, Gemini Pro Vision, GPT-4o

### Example:
```python
prompt = f"""
You are a thumbnail expert. Analyze these 10 frames and select the best one.

Creator Brief:
- Title: {title}
- Target emotion: {target_emotion}
- Niche: {niche}
- Channel style: {style}

Which frame is best and why?
"""

response = await openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "url": frame1_url},
            {"type": "image_url", "url": frame2_url},
            # ... 8 more frames
        ]}
    ]
)
```

### Pros:
- ✅ Simple pipeline (one API call)
- ✅ Holistic reasoning about image + context
- ✅ Natural language explanation
- ✅ Can handle nuance and edge cases

### Cons:
- ❌ **Expensive**: $0.01-0.10 per image × 10 frames = $0.10-1.00 per selection
- ❌ **Slow**: 5-10 seconds for 10 images
- ❌ **Inconsistent**: May give different answers for same input
- ❌ **Less control**: Can't tune individual components
- ❌ **Token limits**: May struggle with 10 high-res images + long context
- ❌ **Hallucination risk**: May invent features that aren't there

### Verdict: ❌ Too expensive and slow for production use

---

## Option 2: Specialized Pipeline ✅ (Recommended for Analysis)

### Approach:
Use specialized models for each analysis task, then combine scores

**Pipeline Architecture**:
```
┌─────────────┐
│ 10 Frames   │
└──────┬──────┘
       │
       ├─────────────────────────────────────────────────┐
       │                                                 │
       ▼                                                 ▼
┌──────────────────┐                          ┌──────────────────┐
│ Visual Analysis  │                          │ Semantic         │
│ (Parallel)       │                          │ Understanding    │
├──────────────────┤                          ├──────────────────┤
│                  │                          │                  │
│ • CLIP           │                          │ • Stream A data  │
│ • Aesthetic      │                          │ • Stream B data  │
│ • Face (FER+)    │                          │ • Brief matching │
│ • Composition    │                          │                  │
│ • Technical      │                          │                  │
└────────┬─────────┘                          └────────┬─────────┘
         │                                             │
         └──────────────────┬──────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Score Fusion    │
                  │  (Weighted Sum)  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Top 1 Frame      │
                  │ + Alternatives   │
                  └──────────────────┘
```

### Models by Task:

#### 1. **Visual Quality & Aesthetics**
```python
# CLIP for visual understanding
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# Cost: FREE (local inference)
# Speed: ~50ms per image (GPU), ~200ms (CPU)

# ImageReward for aesthetic quality
from ImageReward import load
model = load("ImageReward-v1.0")
score = model.score(image)  # 0-1 quality score
# Cost: FREE (local)
# Speed: ~100ms per image (GPU)

# LAION Aesthetics Predictor
from aesthetics_predictor import AestheticsPredictor
predictor = AestheticsPredictor()
aesthetic_score = predictor(image)  # 0-10 score
# Cost: FREE (local)
# Speed: ~80ms per image (GPU)
```

#### 2. **Face Quality & Emotion**
```python
# Already have: MediaPipe + FER+
from app.face_analysis import FaceExpressionAnalyzer
analyzer = FaceExpressionAnalyzer()
result = analyzer.analyze_frame(frame_path)
# Cost: FREE (local)
# Speed: ~15ms per frame (CPU)
```

#### 3. **Composition Analysis**
```python
# Computer vision algorithms (OpenCV + scikit-image)
import cv2
from skimage import filters, measure

def analyze_composition(image):
    """
    Rule-based composition analysis.
    - Rule of thirds alignment
    - Contrast (LAB color space)
    - Color harmony
    - Visual balance
    - Sharpness (Laplacian variance)
    """
    # All FREE, local processing
    # Speed: ~20ms per image
```

#### 4. **Visual Similarity (Deduplication)**
```python
# Option A: Perceptual Hashing (Fast)
import imagehash
hash1 = imagehash.phash(image)
# Speed: ~5ms per image

# Option B: CLIP Embeddings (More accurate)
embeddings = clip_model.get_image_features(image)
similarity = cosine_similarity(emb1, emb2)
# Speed: ~50ms per image
```

#### 5. **Psychology Trigger Detection**
```python
# Custom CV + CLIP-based detectors
def detect_curiosity_gap(image, clip_model):
    """
    Detect curiosity triggers:
    - Pointing gesture
    - Looking off-screen
    - Incomplete action
    - Question-like expression
    """
    # Combine CLIP embeddings + pose detection
    # Speed: ~100ms per image

def detect_emotional_contagion(face_analysis):
    """
    Strong authentic emotions.
    """
    # Use existing FER+ results
    # Speed: 0ms (already computed)
```

### Specialized Pipeline Costs:

**Per frame analysis**:
- CLIP embeddings: FREE (local)
- Aesthetic scoring: FREE (local)
- Face analysis: FREE (already have)
- Composition: FREE (CV algorithms)
- Psychology detection: FREE (CLIP + rules)

**Total**: ~$0.00 per frame, ~500ms per frame (with GPU)

### Pros:
- ✅ **Free/cheap**: Mostly local models
- ✅ **Fast**: Can parallelize (10 frames in ~500ms with GPU)
- ✅ **Consistent**: Deterministic scoring
- ✅ **Debuggable**: Know exactly why each score was computed
- ✅ **Tunable**: Adjust weights and thresholds
- ✅ **Scalable**: Run on GPU for speed

### Cons:
- ⚠️ Less nuanced than human/VLM judgment
- ⚠️ May miss subtle creative decisions
- ⚠️ Requires careful weight tuning

---

## Option 3: Hybrid Approach ⭐ (RECOMMENDED)

### Approach:
**Phase 1**: Specialized models for objective analysis
**Phase 2**: VLM for final ranking + reasoning (top 3 only)

### Pipeline:

```
Phase 1: Objective Analysis (Fast, Local)
┌─────────────┐
│ 10 Frames   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Specialized Models (Parallel)       │
│  • CLIP                              │
│  • Aesthetics                        │
│  • Face/Emotion (FER+)               │
│  • Composition (CV)                  │
│  • Technical Quality                 │
│  • Psychology Triggers               │
│  • Brief Matching                    │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Score Fusion │
        │ (Weighted)   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Top 3 Frames │ ← Narrow to 3 best candidates
        └──────┬───────┘
               │
               ▼
Phase 2: VLM Final Decision (Slow, Expensive)
┌──────────────────────────────────────┐
│  GPT-4o / Claude 3.5 Sonnet          │
│  • Analyzes top 3 frames             │
│  • Considers creator brief nuance    │
│  • Makes final call                  │
│  • Generates reasoning               │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Final Frame  │
        │ + Reasoning  │
        └──────────────┘
```

### Example Implementation:

```python
async def select_best_thumbnail(frames, creative_brief, channel_profile):
    # PHASE 1: Fast local analysis (500ms)
    scores = []
    for frame in frames:
        score = await analyze_frame_specialized(
            frame=frame,
            brief=creative_brief,
            profile=channel_profile
        )
        scores.append(score)

    # Get top 3 candidates
    top_3 = sorted(scores, key=lambda x: x["total_score"], reverse=True)[:3]

    # PHASE 2: VLM final decision (3-5 seconds)
    final_result = await vlm_final_decision(
        top_3_frames=top_3,
        brief=creative_brief,
        profile=channel_profile
    )

    return final_result

async def vlm_final_decision(top_3_frames, brief, profile):
    """
    Use VLM to make final call on top 3 frames.
    """
    prompt = f"""
You are a thumbnail optimization expert. I've narrowed down to these 3 frames
using quantitative analysis. Now I need your creative judgment.

Creator Brief:
- Video Title: "{brief.video_title}"
- Primary Message: "{brief.primary_message}"
- Target Emotion: {brief.target_emotion}
- Channel Niche: {profile.niche}
- Channel Personality: {profile.personality}

Here are the top 3 frames with their scores:

Frame 1: Score {top_3_frames[0]["total_score"]}
- Visual Quality: {top_3_frames[0]["visual_quality"]}
- Psychology: {top_3_frames[0]["psychology_triggers"]}
- Emotion: {top_3_frames[0]["emotion"]}

Frame 2: Score {top_3_frames[1]["total_score"]}
[similar details]

Frame 3: Score {top_3_frames[2]["total_score"]}
[similar details]

Which frame would YOU choose and why? Consider:
1. Title/thumbnail synergy
2. Niche fit
3. Psychological impact
4. Uniqueness in this space

Return JSON:
{{
  "selected_frame": 1 or 2 or 3,
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "key_factors": ["factor1", "factor2"],
  "alternative_use_case": "When frame X might be better..."
}}
"""

    response = await openai.chat.completions.create(
        model="gpt-4o",  # or claude-3-5-sonnet-20241022
        messages=[
            {"role": "system", "content": "You are a thumbnail expert with deep knowledge of YouTube psychology and creator branding."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "url": top_3_frames[0]["url"]},
                {"type": "image_url", "url": top_3_frames[1]["url"]},
                {"type": "image_url", "url": top_3_frames[2]["url"]},
            ]}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### Hybrid Costs:

**Phase 1 (10 frames)**:
- Specialized models: $0.00 (local)
- Time: ~500ms (GPU) to 2s (CPU)

**Phase 2 (3 frames)**:
- GPT-4o: ~$0.03 (3 images × $0.01)
- Claude 3.5 Sonnet: ~$0.045 (3 images × $0.015)
- Time: 3-5 seconds

**Total per selection**: ~$0.03-0.05, ~4-7 seconds

### Pros:
- ✅ **Best of both worlds**: Fast objective analysis + smart creative decision
- ✅ **Cost effective**: Only pay VLM for top 3 (not all 10)
- ✅ **Explainable**: Quantitative scores + qualitative reasoning
- ✅ **Tunable**: Adjust Phase 1 weights without affecting Phase 2
- ✅ **Consistent**: Phase 1 deterministic, Phase 2 adds nuance

---

## Recommended Model Choices

### For Specialized Pipeline (Phase 1):

| Task | Model | Cost | Speed (GPU) | Accuracy |
|------|-------|------|-------------|----------|
| **Visual Embeddings** | CLIP ViT-L/14 | FREE | 50ms | ⭐⭐⭐⭐⭐ |
| **Aesthetics** | LAION Aesthetics v2 | FREE | 80ms | ⭐⭐⭐⭐ |
| **Face/Emotion** | FER+ (ONNX) | FREE | 15ms | ⭐⭐⭐⭐ |
| **Composition** | OpenCV + scikit-image | FREE | 20ms | ⭐⭐⭐ |
| **Similarity** | CLIP embeddings | FREE | 5ms | ⭐⭐⭐⭐⭐ |

**Total Phase 1**: ~$0.00, ~170ms per frame (parallel: ~500ms for 10 frames)

### For VLM Decision (Phase 2):

| Model | Cost (3 images) | Speed | Reasoning Quality | JSON Support |
|-------|-----------------|-------|-------------------|--------------|
| **GPT-4o** | $0.03 | 3-4s | ⭐⭐⭐⭐⭐ | ✅ Native |
| **Claude 3.5 Sonnet** | $0.045 | 4-6s | ⭐⭐⭐⭐⭐ | ✅ Native |
| **Gemini Pro Vision** | $0.006 | 2-3s | ⭐⭐⭐⭐ | ⚠️ Manual |
| **GPT-4o-mini** | $0.003 | 2-3s | ⭐⭐⭐ | ✅ Native |

**Recommended**: **GPT-4o** or **Claude 3.5 Sonnet**
- Best reasoning quality
- Reliable JSON output
- Worth the cost for final decision

---

## Final Recommendation: Hybrid Architecture

```python
# Phase 1: Specialized Models (Fast, Free, Objective)
class SpecializedAnalyzer:
    def __init__(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.aesthetics = load_aesthetics_model()
        self.face_analyzer = FaceExpressionAnalyzer()

    async def analyze_frame(self, frame, brief, profile):
        # Parallel analysis
        visual_score = self.compute_visual_quality(frame)
        aesthetic_score = self.compute_aesthetics(frame)
        face_score = self.analyze_face(frame)
        composition_score = self.analyze_composition(frame)
        psychology_score = self.detect_psychology_triggers(frame)
        brief_match = self.match_creative_brief(frame, brief)

        total_score = (
            0.25 * visual_score +
            0.20 * aesthetic_score +
            0.20 * face_score +
            0.15 * composition_score +
            0.10 * psychology_score +
            0.10 * brief_match
        )

        return {
            "total_score": total_score,
            "components": {...},
            "features": {...}
        }

# Phase 2: VLM Final Decision (Smart, Creative)
async def vlm_final_ranking(top_3, brief, profile):
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "url": f["url"]} for f in top_3]
            ]
        }],
        response_format={"type": "json_object"}
    )
    return response
```

### Why This Works:

1. **Phase 1 filters objectively**: Removes obviously bad frames fast
2. **Phase 2 adds creativity**: VLM considers nuance and brand fit
3. **Cost efficient**: Only 3 VLM API calls instead of 10
4. **Fast enough**: ~5s total (good UX for high-value decision)
5. **Explainable**: Quantitative + qualitative reasoning
6. **Tunable**: Adjust Phase 1 weights without re-prompting VLM

---

## Alternative: All Local (No VLM)

If you want to avoid VLM costs entirely:

```python
# Use specialized pipeline + weighted scoring
final_score = (
    0.30 * creator_alignment_score +
    0.25 * visual_quality_score +
    0.25 * psychology_score +
    0.15 * originality_score +
    0.05 * technical_quality
)

# Generate reasoning with template
reasoning = f"""
Selected frame at {timestamp}s (score: {final_score:.2f})

Key strengths:
- {top_strength_1}
- {top_strength_2}
- {top_strength_3}

Psychology triggers: {triggers}
Best for: {use_case}
"""
```

**Pros**: Free, fast, deterministic
**Cons**: Less nuanced, template-based reasoning

---

## Summary Table

| Approach | Cost | Speed | Quality | Recommended? |
|----------|------|-------|---------|--------------|
| Single VLM (all 10 frames) | $0.10-1.00 | 10s | ⭐⭐⭐⭐ | ❌ Too expensive |
| Specialized only | $0.00 | 0.5s | ⭐⭐⭐ | ⚠️ Good for budget |
| **Hybrid (Phase 1 + 2)** | **$0.03** | **5s** | **⭐⭐⭐⭐⭐** | **✅ BEST** |

---

**Recommendation**: **Hybrid approach with Gemini 2.5 Flash** ⭐

This gives you:
- ✅ Speed and cost efficiency from specialized models
- ✅ Intelligence and nuance from VLM
- ✅ Explainability from both quantitative and qualitative analysis
- ✅ **~$0.0023 per selection** (13x cheaper than GPT-4o!)
- ✅ ~2 seconds total (excellent UX)

## UPDATE: Gemini 2.5 Flash - Production Recommendation

### Why Gemini 2.5 Flash?

**Cost Advantage**:
- Gemini 2.5 Flash: **$0.0023** per selection
- GPT-4o: $0.03 per selection
- **13x cheaper** with comparable quality

**Pricing** (as of January 2025):
- Input: $0.30 per 1M tokens
- Output: $2.50 per 1M tokens
- Images: ~258 tokens each

**Cost Calculation** (10 frames + prompt):
```
Input tokens: (10 images × 258) + 1,000 prompt = 3,580 tokens
Output tokens: ~500 tokens (JSON response)

Input cost: (3,580 / 1,000,000) × $0.30 = $0.00107
Output cost: (500 / 1,000,000) × $2.50 = $0.00125
Total: ~$0.0023 per selection
```

**At Scale**:
| Volume | Gemini 2.5 Flash | GPT-4o | Savings |
|--------|------------------|--------|---------|
| 100 selections | $0.23 | $3.00 | 92% |
| 1,000 selections | $2.30 | $30.00 | 92% |
| 10,000 selections | $23.00 | $300.00 | 92% |
| 100,000 selections | $230.00 | $3,000.00 | 92% |

### Implementation

```python
import google.generativeai as genai
from app.thumbnail_agent import ThumbnailSelector

# Initialize with Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create selector (uses Gemini 2.5 Flash by default)
selector = ThumbnailSelector()

# Or use Gemini 2.5 Pro for higher quality ($0.007 per selection)
selector_pro = ThumbnailSelector(use_pro=True)

# Select best thumbnail
result = await selector.select_best_thumbnail(
    frames=extracted_frames,
    creative_brief=brief,
    channel_profile=profile
)

print(f"Selected: {result['selected_frame_path']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Cost: ${result['cost_usd']:.4f}")
```

### Features:
- ✅ Native JSON mode (structured output)
- ✅ 1M token context window (handles long briefs)
- ✅ Fast (1-2s response time)
- ✅ Multimodal (text + images)
- ✅ High quality reasoning
