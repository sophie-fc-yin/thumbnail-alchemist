# Performance Optimization Brainstorm
## Analysis Time: 5 minutes for 2-minute video → Target: <1 minute

---

## 🔍 Current Performance Bottlenecks

Based on code analysis, the main bottlenecks are:

1. **Vision Feature Computation** (LARGEST BOTTLENECK)
   - Sequential processing of ALL extracted frames
   - Each frame gets: face analysis (MediaPipe + FER+), aesthetics, composition, editability, technical quality
   - For a 2-min video: ~20-40 frames extracted, all analyzed fully
   - Face analysis alone can take 100-500ms per frame

2. **Frame Extraction**
   - Extracts frames from ALL importance segments
   - Only top 5-8 segments selected AFTER full analysis
   - Many frames analyzed but never used

3. **Audio Analysis**
   - Transcription + Stream A/B analysis (moderate impact)

4. **No Parallel Processing**
   - Vision features computed sequentially (even with batching, analyzer is reused but frames processed one-by-one)

---

## 💡 Optimization Strategies

### 🎯 HIGH IMPACT (Recommended to implement)

#### 1. **Two-Stage Filtering** ⭐⭐⭐
**Problem**: Full vision analysis on frames that get discarded
**Solution**: Quick pre-filter → Full analysis only for top candidates

```python
# Stage 1: Quick filter (fast, lightweight)
- Importance score only (already computed)
- Simple image quality check (brightness, contrast - very fast)
- Skip face analysis, composition, editability
- Filter to top 20-30 frames

# Stage 2: Full analysis (only for top candidates)
- Full vision features for top 20-30 frames
- Then select top 10-15 for Gemini
```

**Expected Speedup**: 2-3x (analyze 30 frames instead of 40+)

---

#### 2. **Parallel Vision Feature Computation** ⭐⭐⭐
**Problem**: Sequential processing in `compute_vision_features_batch()`
**Solution**: Process frames in parallel batches

```python
async def compute_vision_features_parallel(
    frame_paths: list[Path],
    niche: str,
    batch_size: int = 5,
    max_workers: int = 4
) -> list[dict]:
    """Process frames in parallel batches."""
    # Use asyncio or multiprocessing
    # Create multiple analyzers (one per worker)
    # Process batches concurrently
```

**Expected Speedup**: 3-4x (if 4 workers process 4 frames simultaneously)

**Trade-off**: Higher memory usage (multiple analyzers/models)

---

#### 3. **Reduce Frame Extraction Density** ⭐⭐
**Problem**: Extracting too many frames from each segment
**Solution**: Be more aggressive with intervals

```python
# In constants.py or get_adaptive_intervals():
BASE_DENSE_INTERVAL_CRITICAL = 0.4  # was 0.3 → extract fewer frames
BASE_DENSE_INTERVAL_HIGH = 0.6      # was 0.5
# OR: Cap max frames per segment
MAX_FRAMES_PER_SEGMENT = 3  # New constant
```

**Expected Speedup**: 1.5x (fewer frames = less analysis)

**Trade-off**: Might miss some good frames (but probably negligible)

---

#### 4. **Skip Low-Importance Segments** ⭐⭐
**Problem**: Analyzing frames from "low" and "minimal" segments
**Solution**: Only extract/analyze critical/high/medium segments

```python
# In identify_important_moments():
MIN_IMPORTANCE_LEVEL = "medium"  # Skip "low" and "minimal"
importance_segments = [s for s in segments
                      if s["importance_level"] in ["critical", "high", "medium"]]
```

**Expected Speedup**: 1.3-1.5x (skip ~20-30% of frames)

---

### 🎯 MEDIUM IMPACT

#### 5. **Lazy/Lightweight Initial Analysis** ⭐⭐
**Problem**: Full analysis for all frames
**Solution**: Compute only essential features initially

```python
def compute_quick_vision_features(frame_path: Path) -> dict:
    """Fast path: Only essential features."""
    - Image quality (brightness, contrast) - fast
    - Basic face detection (MediaPipe only, skip FER+ emotion model)
    - Skip: composition, editability, technical quality (detailed)
    - Return quick score for filtering
```

**Expected Speedup**: 2x for initial pass (skip expensive features)

**Use case**: Stage 1 filtering, then full analysis for top candidates

---

#### 6. **Reduce Face Analysis Complexity** ⭐
**Problem**: FER+ emotion model is expensive
**Solution**: Options:
- Skip FER+ for non-top candidates
- Use lighter emotion model
- Quantize models (if using TensorFlow/PyTorch)

```python
# Option 1: Skip FER+ for initial filtering
def analyze_frame_quick(self, frame_path):
    # MediaPipe only (fast)
    # Skip FER+ emotion model

def analyze_frame_full(self, frame_path):
    # MediaPipe + FER+ (slow but accurate)
```

**Expected Speedup**: 1.5-2x per frame (if skipping FER+)

---

#### 7. **Cache/Reuse Computations** ⭐
**Problem**: Some computations might be redundant
**Solution**:
- Cache image quality metrics (if frames are similar)
- Reuse face analyzer (already done)
- Cache segment importance calculations

**Expected Speedup**: 1.1-1.2x (marginal but free)

---

#### 8. **Optimize Frame Selection Logic** ⭐
**Problem**: Selecting top segments after full analysis
**Solution**: Select segments BEFORE frame extraction

```python
# In orchestrate_adaptive_sampling():
# 1. Calculate importance segments
# 2. Select top 5-8 segments FIRST
# 3. Extract frames ONLY from selected segments
# 4. Analyze only those frames
```

**Expected Speedup**: 1.3-1.5x (skip extraction + analysis for low segments)

---

### 🎯 LOW IMPACT (Long-term)

#### 9. **GPU Acceleration**
- Use GPU for face detection/analysis (if available)
- MediaPipe can use GPU
- FER+ model on GPU

**Expected Speedup**: 2-3x (if GPU available)

**Complexity**: High (requires GPU setup, dependency management)

---

#### 10. **Model Optimization**
- Quantize FER+ model (INT8 quantization)
- Use lighter face detection model
- Replace FER+ with lighter emotion model

**Expected Speedup**: 1.5-2x

**Trade-off**: Potential accuracy loss

---

#### 11. **Reduce Audio Analysis Time**
- Parallel Stream A + B (already done ✓)
- Skip transcription for very short videos?
- Cache audio features if video unchanged

**Expected Speedup**: 0.5-1x (audio is not the main bottleneck)

---

#### 12. **Async Frame Upload**
**Problem**: Blocking GCS uploads
**Solution**: Upload frames asynchronously while processing next batch

**Expected Speedup**: 0.2-0.3x (overlaps I/O with computation)

---

## 🚀 Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. **Skip Low-Importance Segments** (#4) - Easy, immediate 1.3x speedup
2. **Reduce Frame Density** (#3) - Easy config change, 1.5x speedup
3. **Select Segments Before Extraction** (#8) - Moderate refactor, 1.3x speedup

**Combined Expected**: ~2.5x speedup → **2 minutes instead of 5**

---

### Phase 2: High Impact (3-5 days)
4. **Two-Stage Filtering** (#1) - Moderate complexity, 2-3x speedup
5. **Parallel Vision Features** (#2) - Higher complexity, 3-4x speedup

**Combined with Phase 1**: ~5-6x speedup → **30-50 seconds instead of 5 minutes**

---

### Phase 3: Polish (optional)
6. **Lightweight Initial Analysis** (#5) - Enhance two-stage filtering
7. **Reduce Face Analysis Complexity** (#6) - For non-top candidates
8. **GPU Acceleration** (#9) - If infrastructure supports

---

## 📊 Expected Performance After Optimizations

| Optimization | Time (2-min video) | Speedup |
|-------------|-------------------|---------|
| Baseline | 5 minutes | 1x |
| Phase 1 (Quick Wins) | 2 minutes | 2.5x |
| Phase 2 (High Impact) | 30-50 seconds | 6-10x |
| Phase 3 (Polish) | 20-30 seconds | 10-15x |

---

## 🎯 Specific Code Changes Needed

### 1. Skip Low-Importance Segments
**File**: `app/analysis/processing.py`
**Function**: `identify_important_moments()`
```python
# Filter segments before extraction
MIN_IMPORTANCE_LEVELS = ["critical", "high", "medium"]
importance_segments = [
    s for s in importance_segments
    if s["importance_level"] in MIN_IMPORTANCE_LEVELS
]
```

---

### 2. Two-Stage Filtering
**File**: `app/analysis/processing.py`
**Function**: `identify_important_moments()`
```python
# After frame extraction:
# Stage 1: Quick filter
quick_scores = [compute_quick_score(f) for f in extracted_frames]
top_indices = sorted(range(len(quick_scores)),
                     key=lambda i: quick_scores[i],
                     reverse=True)[:30]

# Stage 2: Full analysis only for top 30
top_frames = [extracted_frames[i] for i in top_indices]
frames_with_features = compute_vision_features_batch(top_frames)
```

---

### 3. Parallel Vision Features
**File**: `app/vision/feature_analysis.py`
**Function**: `compute_vision_features_batch()`
```python
async def compute_vision_features_batch_parallel(
    frame_paths: list[Path],
    niche: str = "general",
    max_workers: int = 4,
) -> list[dict]:
    """Process frames in parallel."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def process_frame(frame_path):
        analyzer = get_face_expression_analyzer()
        return compute_vision_features(frame_path, niche, analyzer)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_frame, frame_paths))

    return results
```

---

### 4. Select Segments Before Extraction
**File**: `app/analysis/adaptive_sampling.py`
**Function**: `orchestrate_adaptive_sampling()`
```python
# After calculating importance_segments:
# Select top segments FIRST
selected_segments = select_top_segments(
    importance_segments,
    min_segments=5,
    max_segments=8
)

# Extract frames ONLY from selected segments
frames_with_features = await identify_important_moments(
    ...,
    importance_segments=selected_segments,  # Only top segments
)
```

---

## 🔧 Configuration Tuning

### Reduce Frame Density
**File**: `app/constants.py`
```python
# More aggressive intervals (fewer frames)
BASE_DENSE_INTERVAL_CRITICAL = 0.4  # was 0.3
BASE_DENSE_INTERVAL_HIGH = 0.6      # was 0.5
BASE_DENSE_INTERVAL_MEDIUM = 1.0    # was 0.8

# OR cap frames per segment
MAX_FRAMES_PER_SEGMENT = 3  # New constant
```

---

## ⚠️ Trade-offs & Considerations

1. **Accuracy**: Two-stage filtering might miss some good frames (mitigated by generous top-N)
2. **Memory**: Parallel processing uses more RAM (multiple analyzers)
3. **Complexity**: Parallel code is harder to debug
4. **Infrastructure**: GPU requires setup/configuration

---

## 📝 Next Steps

1. **Measure baseline**: Add timing logs to identify exact bottlenecks
2. **Implement Phase 1**: Quick wins first (low risk, high reward)
3. **Test & validate**: Ensure quality doesn't degrade
4. **Implement Phase 2**: High-impact optimizations
5. **Monitor**: Track performance metrics in production
