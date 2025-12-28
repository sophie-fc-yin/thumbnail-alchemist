# Adaptive Frame Sampling Architecture

**Purpose**: Extract the most important video frames for thumbnail generation by analyzing audio semantics, visual signals, and perceptual impact.

**Status**: 🟡 In Progress
**Last Updated**: 2025-01-28

---

## Table of Contents
- [Overview](#overview)
- [Current Architecture](#current-architecture)
- [Implementation Status](#implementation-status)
- [Dual-Stream Architecture (Target)](#dual-stream-architecture-target)
- [Technical Details](#technical-details)
- [Roadmap](#roadmap)
- [API Reference](#api-reference)

---

## Overview

### Goal
Identify and extract video frames that are most likely to make compelling thumbnails by detecting:
- **Narrative importance** (what's being said, story beats)
- **Perceptual impact** (epic moments, surprises, reveals)
- **Visual quality** (facial expressions, motion, composition)

### Core Principle
Instead of uniformly sampling frames, we sample **adaptively** based on multi-modal signals:
- Audio semantics (Stream A)
- Audio saliency (Stream B)
- Visual features (face expression, motion)

---

## Current Architecture

### Pipeline Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐               ┌──────────┐
│  AUDIO  │               │  VIDEO   │
│ STREAM  │               │  STREAM  │
└────┬────┘               └─────┬────┘
     │                          │
     │ ┌─────────────────┐      │
     ├─►Speech Detection │      │
     │ │ (Silero VAD)    │      │
     │ └─────────────────┘      │
     │                          │
     │ ┌─────────────────┐      │
     ├─►Transcription    │      │
     │ │ (OpenAI ASR)    │      │
     │ └─────────────────┘      │
     │                          │
     │ ┌─────────────────┐      │
     └─►Audio Features   │      │
       │ (Librosa)       │      │
       └─────────────────┘      │
                                │
                          ┌─────▼──────┐
                          │Initial     │
                          │Sampling    │
                          │(~20 frames)│
                          └─────┬──────┘
                                │
                          ┌─────▼──────┐
                          │Face        │
                          │Analysis    │
                          │(MediaPipe) │
                          └─────┬──────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
        ┌───────────────┐             ┌─────────────────┐
        │ Expression    │             │ Landmark        │
        │ Delta         │             │ Motion          │
        └───────┬───────┘             └────────┬────────┘
                │                              │
                └──────────┬───────────────────┘
                           │
                     ┌─────▼─────────┐
                     │ Pace Score    │
                     │ Calculation   │
                     └─────┬─────────┘
                           │
                     ┌─────▼─────────┐
                     │ Adaptive      │
                     │ Extraction    │
                     └───────────────┘
```

### Current Data Flow

**Step 1: Audio Extraction & Separation**
```python
audio_result = extract_audio_from_video(video_path, project_id)
# Returns:
{
    "speech": Path("audio_speech.wav"),      # Speech-only (Silero VAD filtered)
    "full_audio": Path("audio_full.wav"),    # Complete audio track
    "speech_ratio": 0.42,                     # % of video that's speech
    "segments": [(0.5, 3.2), (5.1, 8.7)],    # Speech timestamps
}
```

**Step 2: Speech Transcription**
```python
analysis = transcribe_and_analyze_audio(speech_path, project_id)
# Uses: OpenAI GPT-4o Transcribe + Diarize
# Returns: transcript, speakers, timeline with emotions
```

**Step 3: Audio Feature Extraction**
```python
features = analyze_audio_features(full_audio_path)
# Returns: pitch, energy, spectral features, tempo, beats
```

**Step 4: Initial Visual Sampling**
```python
# Extract ~20 sample frames evenly distributed
sample_frames = extract_sample_frames_to_temp(video, project_id)
```

**Step 5: Face Analysis**
```python
# MediaPipe + FER+ ONNX emotion model
for frame in sample_frames:
    analysis = analyzer.analyze_frame(frame)
    # Returns: expression_intensity, eye_openness, head_pose, emotion_probs
```

**Step 6: Pace Calculation**
```python
pace = calculate_pace_score(
    expression_delta=0.8,    # Face expression change (30%)
    landmark_motion=0.6,     # Head/facial movement (20%)
    audio_energy_delta=0.9,  # Audio loudness change (30%)
    speech_emotion_delta=0.7 # Vocal emotion change (20%)
)
# pace ∈ [0, 1] → sampling interval ∈ [0.1s, 2.0s]
```

**Step 7: Adaptive Frame Extraction**
```python
# High pace (0.7-1.0) → sample every 0.1-0.25s (dense)
# Med pace (0.3-0.7)  → sample every 0.5-1.0s (normal)
# Low pace (0.0-0.3)  → sample every 1.5-2.0s (sparse)

for segment in pace_segments:
    interval = pace_to_sampling_interval(segment["avg_pace"])
    extract_frames_for_segment(segment, interval)
```

---

## Implementation Status

### ✅ Completed

#### Audio Processing
- [x] **Speech/Full Audio Separation** (`app/audio_media.py`)
  - Silero VAD for speech detection
  - Pitch analysis to filter singing from speaking
  - Two-lane extraction (speech + full audio)

- [x] **Speech Detection** (`app/speech_detection.py`)
  - `SpeechDetector` class with Silero VAD
  - `detect_speech_segments()` - VAD-based speech timestamps
  - `filter_singing_segments()` - Pitch variance filtering (speaking vs singing)
  - `create_speech_only_audio()` - Extract and concatenate speech

- [x] **Audio Feature Extraction** (`app/audio_media.py`)
  - `analyze_audio_features()` using librosa
  - Pitch, energy (RMS), zero-crossing rate
  - Spectral features (centroid, rolloff)
  - Tempo and beat tracking

- [x] **Transcription & Timeline** (`app/audio_media.py`)
  - OpenAI GPT-4o Transcribe + Diarize
  - Speaker diarization
  - Millisecond-precision timeline
  - Audio normalization (EBU R128)

#### Visual Processing
- [x] **Face Analysis** (`app/face_analysis.py`)
  - MediaPipe face detection (468 landmarks)
  - FER+ ONNX emotion model
  - Expression intensity: `1 - P(neutral)`
  - Eye/mouth openness calculation
  - Head pose estimation (pitch, yaw, roll)
  - Landmark motion tracking

#### Pace Analysis
- [x] **Pace Calculation** (`app/pace_analysis.py`)
  - Multi-signal fusion (expression, motion, audio, speech)
  - Configurable weights (default: 30/20/30/20)
  - Pace-to-interval conversion
  - Video segmentation by pace
  - Audio energy delta calculation
  - Speech emotion delta calculation

#### Orchestration
- [x] **Adaptive Sampling Pipeline** (`app/adaptive_sampling.py`)
  - 5-step orchestrated workflow
  - Temp bucket integration (clickmoment-prod-temp)
  - Progress logging and stats tracking
  - Local cleanup after processing
  - Sample frame download for analysis

- [x] **API Endpoint** (`app/main.py`)
  - `POST /vision/adaptive-sampling`
  - Returns: frames, pace segments, statistics, processing time

#### Infrastructure
- [x] Dependencies: torch, torchaudio, soundfile, silero-vad
- [x] GCS integration (temp + production buckets)
- [x] Retry logic for OpenAI API (3 attempts, exponential backoff)
- [x] Audio normalization (loudnorm filter)

### 🟡 In Progress

- [ ] **Stream A: Speech Semantics** (partially done)
  - ✅ Speech detection
  - ✅ Transcription
  - ✅ Basic prosody (pitch, energy)
  - ⏳ Tone detection (emotion classification on speech)
  - ⏳ Narrative context (important sentences via LLM)
  - ⏳ Speaking style change detection

- [ ] **Stream B: Full Audio Saliency** (needs implementation)
  - ✅ Energy extraction
  - ✅ Spectral features
  - ⏳ Energy peak detection
  - ⏳ Spectral change detection
  - ⏳ Silence-to-impact detection
  - ⏳ Non-speech sound detection

### 📋 Planned (Not Started)

- [ ] **Timeline Merger** (`app/moment_fusion.py`)
  - Combine Stream A + Stream B moments
  - Score and rank moment candidates
  - Weighted fusion (configurable)

- [ ] **Moment-Based Sampling**
  - Replace pace-based with moment-based sampling
  - Extract frames at high-importance moments
  - Adaptive density around key moments

- [ ] **Visual Saliency**
  - Object detection (products, faces, text)
  - Scene change detection
  - Composition quality scoring

- [ ] **LLM Integration for Narrative**
  - Identify hook statements
  - Detect emotional peaks in transcript
  - Extract story beats

---

## Dual-Stream Architecture (Target)

### Design Philosophy

**Current Problem**: Pace-based sampling misses important moments that aren't "high pace"
- Dramatic silence before a reveal
- Epic soundtrack moment without speaking
- Sudden visual without audio change

**Solution**: Dual-stream analysis that captures both semantic AND perceptual importance

### Architecture Diagram

```
RAW AUDIO
   │
   ├─────────────────────────────────────┬─────────────────────────────────────┐
   │                                     │                                     │
   ▼                                     ▼                                     ▼
┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│   Stream A: SEMANTICS    │  │  Stream B: SALIENCY      │  │   Stream C: VISUAL       │
│   (What's being said)    │  │  (What hits emotionally) │  │   (What looks good)      │
└────────────┬─────────────┘  └────────────┬─────────────┘  └────────────┬─────────────┘
             │                              │                              │
             ▼                              ▼                              ▼
    ┌────────────────┐           ┌────────────────┐           ┌────────────────┐
    │ Speech         │           │ Energy peaks   │           │ Face quality   │
    │ Transcription  │           │ Spectral       │           │ Scene changes  │
    │ Tone/prosody   │           │   change       │           │ Composition    │
    │ Narrative      │           │ Silence→impact │           │ Motion         │
    │   context      │           │ Sound effects  │           │ Saliency       │
    └────────┬───────┘           └────────┬───────┘           └────────┬───────┘
             │                            │                            │
             └────────────────┬───────────┴────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MOMENT CANDIDATES  │
                    │  Scored & Ranked    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ADAPTIVE SAMPLING   │
                    │ Extract frames at   │
                    │ high-value moments  │
                    └─────────────────────┘
```

### Stream A: Speech Semantics

**Goal**: Understand WHAT is being said and WHY it matters narratively

**Inputs**:
- Speech-only audio (Silero VAD filtered)
- Transcript with speaker diarization
- Prosody features (pitch, energy, speaking rate)

**Analysis**:
1. **Narrative Context** (via LLM)
   ```python
   # Analyze transcript to find:
   - Hook statements ("I can't believe...", "This changed everything...")
   - Emotional peaks (excitement, surprise, tension)
   - Story beats (setup, conflict, resolution)
   - Key reveals or information
   ```

2. **Tone Detection**
   ```python
   # Classify speaking tone:
   - Calm/neutral
   - Excited/enthusiastic
   - Serious/dramatic
   - Humorous/playful
   ```

3. **Speaking Style Changes**
   ```python
   # Detect transitions:
   - Calm → Excited (important moment coming)
   - Normal → Whisper (tension)
   - Slow → Fast (urgency)
   ```

**Output**:
```python
[
    {
        "time": 5.2,
        "type": "hook_statement",
        "importance": 0.9,
        "text": "I can't believe this actually worked",
        "tone": "excited"
    },
    {
        "time": 15.8,
        "type": "emotional_peak",
        "importance": 0.85,
        "text": "This is the moment I realized...",
        "tone": "dramatic"
    }
]
```

### Stream B: Full Audio Saliency

**Goal**: Detect moments that HIT hard perceptually, regardless of speech

**Inputs**:
- Full audio track (speech + music + effects)
- Energy, spectral, and temporal features

**Analysis**:
1. **Energy Peak Detection**
   ```python
   # Find sudden loudness spikes:
   - Music drops
   - Sound effects
   - Sudden sounds

   # Method: derivative of RMS energy
   energy_delta = diff(rms)
   peaks = energy_delta > threshold
   ```

2. **Spectral Change Detection**
   ```python
   # Find timbre/character changes:
   - Music genre shifts
   - New instruments entering
   - Sound effect triggers

   # Method: spectral flux (change in frequency distribution)
   spectral_diff = diff(spectral_centroid)
   changes = spectral_diff > threshold
   ```

3. **Silence-to-Impact Detection**
   ```python
   # Find dramatic moments:
   - Quiet build-up → loud payoff
   - Tension → release

   # Method: find low-energy periods followed by spikes
   quiet_moments = rms < quiet_threshold
   following_spikes = rms[t+1] > loud_threshold
   silence_to_impact = quiet_moments & following_spikes
   ```

4. **Non-Speech Sound Detection**
   ```python
   # Identify impactful non-voice sounds:
   - Explosions (high energy + spectral spread)
   - Music crescendos (gradual energy increase)
   - Percussive hits (high ZCR + sharp attack)

   # Method: combine energy, ZCR, spectral features
   is_speech_segment = False  # Outside VAD segments
   high_impact = energy > threshold
   non_speech_sounds = is_speech_segment & high_impact
   ```

**Output**:
```python
[
    {
        "time": 5.1,
        "type": "energy_peak",
        "saliency": 0.95,
        "reason": "Sudden loudness spike (+20dB)"
    },
    {
        "time": 12.8,
        "type": "silence_to_impact",
        "saliency": 0.88,
        "reason": "3s silence → loud music drop"
    },
    {
        "time": 23.4,
        "type": "spectral_change",
        "saliency": 0.82,
        "reason": "Timbre shift (new instrument)"
    }
]
```

### Stream C: Visual Saliency (Future)

**Goal**: Identify visually compelling frames

**Analysis** (planned):
- Face expression peaks
- Scene changes
- Object detection (products, text)
- Composition quality
- Motion/action peaks

### Timeline Merger

**Goal**: Combine all streams into ranked moment candidates

```python
def merge_moment_timelines(
    stream_a: list[dict],  # Speech semantics
    stream_b: list[dict],  # Audio saliency
    stream_c: list[dict],  # Visual saliency
    weights: dict = {
        "speech": 0.4,
        "audio": 0.35,
        "visual": 0.25
    }
) -> list[dict]:
    """
    Merge and score all moment candidates.

    Returns:
        Sorted list of moments with unified scores:
        [
            {
                "time": 5.2,
                "score": 0.92,
                "sources": ["speech_hook", "energy_peak"],
                "metadata": {...}
            },
            ...
        ]
    """
```

**Merging Logic**:
1. **Temporal Alignment**: Group moments within ±0.5s window
2. **Score Fusion**: Weighted combination of stream scores
3. **Source Tracking**: Record which streams contributed
4. **Ranking**: Sort by final score
5. **Clustering**: Merge nearby moments to avoid over-sampling

**Example Output**:
```python
[
    {
        "time": 5.2,
        "score": 0.93,
        "sources": ["speech_hook", "energy_peak", "face_expression"],
        "metadata": {
            "speech": {"text": "I can't believe...", "tone": "excited"},
            "audio": {"type": "energy_peak", "delta": 0.8},
            "visual": {"expression_intensity": 0.9}
        }
    },
    {
        "time": 15.8,
        "score": 0.87,
        "sources": ["emotional_peak", "silence_to_impact"],
        "metadata": {...}
    }
]
```

---

## Technical Details

### Speech Detection (Silero VAD)

**Model**: Silero Voice Activity Detector (PyTorch)
- State-of-the-art VAD with 16kHz audio
- Returns speech timestamps with confidence
- Filters out music, noise, background sounds

**Parameters**:
```python
min_speech_duration_ms=250   # Minimum speech segment length
min_silence_duration_ms=100  # Minimum gap between segments
speech_pad_ms=30             # Padding around detected speech
```

**Singing Filter** (Pitch Variance):
```python
# Speaking: high pitch variance (natural speech melody)
# Singing: low pitch variance (sustained notes)
pitch_variance_threshold=30.0  # Hz std dev

speaking = pitch_variance > 30.0  # Variable pitch
singing = pitch_variance < 30.0   # Sustained pitch
```

### Face Analysis (MediaPipe + FER+)

**MediaPipe Face Mesh**:
- 468 facial landmarks in 3D
- Eye openness: vertical distance between upper/lower lids
- Mouth openness: vertical distance between lips
- Head pose: pitch, yaw, roll from landmark geometry

**FER+ Emotion Model** (ONNX):
- Trained on FER+ dataset
- 8 emotions: neutral, happy, sad, surprise, fear, disgust, anger, contempt
- Expression intensity: `1 - P(neutral)` → measures "how much is happening"

**Landmark Motion**:
```python
# Euclidean distance of landmark movement between frames
motion = sqrt(sum((curr_landmarks - prev_landmarks)^2))
normalized_motion = motion / (image_width * image_height)
```

### Pace Calculation

**Multi-Signal Fusion**:
```python
pace = (
    0.30 * expression_delta +    # Face emotion change
    0.20 * landmark_motion +      # Head/face movement
    0.30 * audio_energy_delta +   # Audio loudness change
    0.20 * speech_emotion_delta   # Vocal tone change
)
```

**Pace-to-Interval Mapping**:
```python
def pace_to_sampling_interval(pace: float) -> float:
    if pace < 0.3:      # Low pace
        return 1.5 + (0.3 - pace) / 0.3 * 0.5  # 1.5-2.0s
    elif pace < 0.7:    # Medium pace
        return 0.5 + (0.7 - pace) / 0.4 * 1.0  # 0.5-1.5s
    else:               # High pace
        return 0.1 + (1.0 - pace) / 0.3 * 0.4  # 0.1-0.5s
```

### Audio Features (Librosa)

**Frame Parameters**:
- Frame length: 100ms (0.1s)
- Hop length: 50ms (50% overlap)
- Sample rate: 16kHz

**Features Extracted**:
```python
{
    "pitch": librosa.piptrack(),              # Fundamental frequency
    "energy": librosa.feature.rms(),          # Root Mean Square energy
    "zcr": librosa.zero_crossing_rate(),      # Zero crossing rate
    "spectral_centroid": librosa.spectral_centroid(),  # Brightness
    "spectral_rolloff": librosa.spectral_rolloff(),    # High-freq content
    "tempo": librosa.beat.beat_track(),       # BPM
    "beat_times": librosa.frames_to_time()    # Beat locations
}
```

### Storage Architecture

**Temp Bucket** (`clickmoment-prod-temp`):
- Sample frames for analysis (auto-delete after 1 day)
- Intermediate processing files
- Metadata: `{"temp": "true", "project_id": "..."}`

**Production Bucket** (`clickmoment-prod-assets`):
- Final extracted frames: `projects/{project_id}/signals/frames/`
- Audio files: `projects/{project_id}/signals/audio/`
- Naming: `frame_{timestamp_ms}ms.jpg`, `audio_speech.wav`, `audio_full.wav`

---

## Roadmap

### Phase 1: Audio Saliency (Stream B) ⏳ Next
**Goal**: Detect perceptual impact moments in audio

**Tasks**:
1. Create `app/audio_saliency.py`
2. Implement energy peak detection
3. Implement spectral change detection
4. Implement silence-to-impact detection
5. Implement non-speech sound classification
6. Return scored moment candidates

**Deliverable**:
```python
saliency_moments = detect_salient_moments(audio_features)
# Returns: [{"time": 5.2, "saliency": 0.9, "type": "energy_peak"}, ...]
```

**Estimated Effort**: 2-3 days

---

### Phase 2: Speech Semantics Enhancement (Stream A) ⏳ After Phase 1
**Goal**: Understand narrative importance of speech

**Tasks**:
1. Implement tone detection on speech segments
2. Add LLM-based narrative analysis
   - Use Claude/GPT to analyze transcript
   - Identify hooks, reveals, emotional peaks
3. Detect speaking style changes
4. Return importance-scored moments

**Deliverable**:
```python
semantic_moments = analyze_speech_semantics(transcript, prosody)
# Returns: [{"time": 5.2, "importance": 0.9, "type": "hook_statement"}, ...]
```

**Estimated Effort**: 3-4 days

---

### Phase 3: Timeline Merger 📋 After Phase 1 & 2
**Goal**: Combine all moment streams into ranked candidates

**Tasks**:
1. Create `app/moment_fusion.py`
2. Implement temporal alignment (group moments within window)
3. Implement weighted score fusion
4. Add moment clustering (avoid over-sampling)
5. Return sorted, merged timeline

**Deliverable**:
```python
moments = merge_moment_timelines(stream_a, stream_b, stream_c)
# Returns: [{"time": 5.2, "score": 0.93, "sources": [...], "metadata": {...}}, ...]
```

**Estimated Effort**: 2 days

---

### Phase 4: Moment-Based Sampling 📋 After Phase 3
**Goal**: Replace pace-based with moment-based frame extraction

**Tasks**:
1. Update `orchestrate_adaptive_sampling()` to use merged moments
2. Extract frames at high-scoring moments
3. Implement adaptive density (more frames around important moments)
4. Update API response to include moment metadata

**Deliverable**:
- Frames extracted at narrative + perceptual key moments
- Response includes moment scores and types

**Estimated Effort**: 2-3 days

---

### Phase 5: Visual Saliency (Stream C) 📋 Future
**Goal**: Add visual analysis to moment detection

**Tasks**:
1. Scene change detection
2. Object detection (YOLO/Faster R-CNN)
3. Composition quality scoring
4. Face quality metrics (beyond expression)
5. Motion/action peak detection

**Estimated Effort**: 1-2 weeks

---

### Phase 6: Optimization & Polish 📋 Future
**Tasks**:
- Performance optimization (parallel processing)
- Model caching (Silero, FER+)
- Better error handling
- Comprehensive logging
- Testing suite

**Estimated Effort**: 1 week

---

## API Reference

### Current Endpoint

**`POST /vision/adaptive-sampling`**

**Request**:
```json
{
    "video_path": "gs://clickmoment-prod-assets/users/.../video.mp4",
    "project_id": "550e8400-e29b-41d4-a716-446655440000",
    "max_frames": 100
}
```

**Response**:
```json
{
    "project_id": "550e8400-e29b-41d4-a716-446655440000",
    "frames": [
        "gs://.../frame_0ms.jpg",
        "gs://.../frame_1500ms.jpg",
        ...
    ],
    "total_frames": 87,
    "pace_segments": [
        {
            "start_time": 0.0,
            "end_time": 15.3,
            "avg_pace": 0.25,
            "pace_category": "low"
        },
        ...
    ],
    "pace_statistics": {
        "avg_pace": 0.54,
        "segment_counts": {"low": 2, "medium": 3, "high": 2},
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
    "summary": "Adaptive sampling complete: 87 frames extracted..."
}
```

### Future Endpoint (Post Phase 4)

**Response will include**:
```json
{
    ...
    "moment_candidates": [
        {
            "time": 5.2,
            "score": 0.93,
            "sources": ["speech_hook", "energy_peak", "face_expression"],
            "frame_path": "gs://.../frame_5200ms.jpg",
            "metadata": {
                "speech": {"text": "I can't believe...", "tone": "excited"},
                "audio": {"type": "energy_peak", "saliency": 0.95},
                "visual": {"expression_intensity": 0.9}
            }
        },
        ...
    ]
}
```

---

## File Structure

```
app/
├── ADAPTIVE_SAMPLING_DESIGN.md  ← This document
├── adaptive_sampling.py          ← Main orchestrator (✅ done)
├── audio_media.py                ← Audio extraction, transcription (✅ done)
├── speech_detection.py           ← Silero VAD, singing filter (✅ done)
├── face_analysis.py              ← MediaPipe + FER+ (✅ done)
├── pace_analysis.py              ← Multi-signal pace fusion (✅ done)
├── audio_saliency.py             ← Audio impact detection (⏳ TODO)
├── moment_fusion.py              ← Timeline merger (📋 TODO)
├── vision_media.py               ← Frame extraction utilities (✅ done)
└── main.py                       ← FastAPI endpoints (✅ done)
```

---

## Dependencies

### Core ML/Audio
- `torch>=2.0.0` - PyTorch for Silero VAD
- `torchaudio>=2.0.0` - Audio processing
- `librosa>=0.10.2` - Audio feature extraction
- `soundfile>=0.12.0` - Audio I/O
- `mediapipe>=0.10.0` - Face mesh detection
- `onnxruntime>=1.16.0` - FER+ emotion model
- `opencv-python>=4.8.0` - Image processing
- `numpy` - Numerical operations

### API & Cloud
- `openai>=1.0.0` - Transcription API
- `google-cloud-storage>=3.7.0` - GCS integration
- `fastapi` - API framework
- `uvicorn[standard]` - ASGI server

---

## Notes & Considerations

### Why Dual-Stream?
Pace-based sampling (current) works well for high-energy moments but misses:
- **Narrative hooks** without visual/audio "pace" (calm but important speech)
- **Epic moments** without speech (music drops, sound effects)
- **Dramatic silence** (tension before reveal)

Dual-stream captures both:
- **Stream A**: "This is important because of what's being said"
- **Stream B**: "This hits hard perceptually"

### Performance Targets
- Audio processing: <20s for 5min video
- Face analysis: ~15ms per frame (CPU)
- Total pipeline: <60s for 5min video @ 100 frames

### Quality Metrics
How to evaluate if this works:
1. **Manual Review**: Do extracted frames make good thumbnails?
2. **Coverage**: Do we capture all important moments?
3. **Precision**: Are we avoiding boring/irrelevant frames?
4. **Efficiency**: Frames extracted vs. frames needed

### Future Enhancements
- Real-time processing (streaming)
- GPU acceleration for face analysis
- Custom emotion models (fine-tuned for YouTube)
- Multi-language support
- Video quality filtering (blur, darkness)
- Brand safety detection

---

**Last Updated**: 2025-01-28
**Maintainer**: Claude Code
**Status**: Living document - update as implementation progresses
