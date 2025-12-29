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
│                       VIDEO INPUT                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
┌─────────────────────┐         ┌──────────────────────┐
│  Extract Audio      │         │  Initial Sampling    │
│  (Speech + Full)    │         │  (~20 frames)        │
│  + Normalization    │         │                      │
└──────────┬──────────┘         └──────────┬───────────┘
           │                               │
           │                               │
┌──────────▼──────────┐         ┌──────────▼───────────┐
│  AUDIO PROCESSING   │         │  VISUAL PROCESSING   │
│                     │         │                      │
│ ┌─────────────────┐ │         │ ┌────────────────┐   │
│ │Speech Detection │ │         │ │Face Detection  │   │
│ │(Silero VAD)     │ │         │ │(MediaPipe)     │   │
│ └────────┬────────┘ │         │ └───────┬────────┘   │
│          │          │         │         │            │
│ ┌────────▼────────┐ │         │ ┌───────▼────────┐   │
│ │Transcription    │ │         │ │Emotion Model   │   │
│ │(OpenAI ASR)     │ │         │ │(FER+ ONNX)     │   │
│ └────────┬────────┘ │         │ └───────┬────────┘   │
│          │          │         │         │            │
│ ┌────────▼────────┐ │         │ ┌───────▼────────┐   │
│ │Audio Features   │ │         │ │Expression      │   │
│ │(Librosa)        │ │         │ │Delta + Motion  │   │
│ └────────┬────────┘ │         │ └───────┬────────┘   │
│          │          │         │         │            │
│    ┌─────┴─────┐   │         │         │            │
│    │           │   │         │         │            │
│    ▼           ▼   │         │         │            │
│ ┌────────┐ ┌─────┐│         │         │            │
│ │Stream A│ │Str B││         │         │            │
│ │(Speech)│ │(Sal)││         │         │            │
│ └───┬────┘ └──┬──┘│         │         │            │
│     │         │   │         │         │            │
└─────┼─────────┼───┘         └─────────┼────────────┘
      │         │                       │
      │         │                       │
      └────┬────┘                       │
           │                            │
           ▼                            │
    ┌──────────────┐                   │
    │  Stream A+B  │                   │
    │  Timeline    │                   │
    └──────┬───────┘                   │
           │                            │
           └────────────┬───────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Pace/Moment     │
              │  Score Fusion    │
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Adaptive Frame  │
              │  Extraction      │
              └──────────────────┘
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

- [x] **Stream A: Speech Semantics** (`app/speech_semantics.py`) ✅ COMPLETE
  - ✅ Speech detection (Silero VAD)
  - ✅ Transcription (OpenAI GPT-4o)
  - ✅ Basic prosody (pitch, energy, speaking rate)
  - ✅ Tone detection (rule-based classification: excited, calm, dramatic, neutral)
  - ✅ Narrative context (GPT-4o-mini LLM analysis)
  - ✅ Speaking style change detection (prosodic transitions)
  - ✅ Unified timeline format (time, type, score, source, metadata)

- [x] **Stream B: Full Audio Saliency** (`app/audio_saliency.py`) ✅ COMPLETE
  - ✅ Energy extraction (librosa RMS)
  - ✅ Spectral features (centroid, rolloff)
  - ✅ Energy peak detection (percentile-based, top 10%)
  - ✅ Spectral change detection (spectral flux analysis)
  - ✅ Silence-to-impact detection (quiet → loud patterns)
  - ✅ Non-speech sound detection (music, effects outside speech)
  - ✅ Unified timeline format (compatible with Stream A)

### 🟡 In Progress

- [ ] **Timeline Merger** (`app/moment_fusion.py`)
  - Combine Stream A + Stream B timelines
  - Temporal alignment (group moments within window)
  - Weighted score fusion
  - Moment clustering and ranking

- [ ] **Orchestrator Integration**
  - Call Stream A analysis in pipeline
  - Call Stream B analysis in pipeline
  - Store moment timelines in response

### 📋 Planned (Not Started)

- [ ] **Moment-Based Sampling**
  - Replace pace-based with moment-based sampling
  - Extract frames at high-importance moments
  - Adaptive density around key moments

- [ ] **Visual Saliency (Stream C)**
  - Object detection (products, faces, text)
  - Scene change detection
  - Composition quality scoring
  - Face expression peaks
  - Motion/action detection

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
- **Single face detection**: `max_num_faces=1` (optimized for solo creator videos)

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

**Design Note**: Currently configured for single-face videos (solo creators). Multi-face support can be added later if needed for interviews/collaborations.

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

### Stream A: Speech Semantics (Technical Details)

**Module**: `app/speech_semantics.py`

#### 1. Tone Detection

**Algorithm**: Rule-based prosodic feature classification

**Input Features**:
```python
# Extracted from speech segments using librosa
pitch_mean: float        # Average fundamental frequency (Hz)
pitch_variance: float    # Variance of pitch (Hz²)
pitch_range: float       # Max - Min pitch (Hz)
energy_mean: float       # Average RMS energy
energy_variance: float   # Variance of RMS energy
speaking_rate: float     # Syllables per second (estimated)
```

**Classification Logic**:
```python
# Normalize features
pitch_norm = min(pitch_mean / 200.0, 2.0)      # 100Hz=low, 300Hz=high
variance_norm = min(pitch_variance / 50.0, 2.0)
energy_norm = min(energy_mean / 0.2, 2.0)
rate_norm = min(speaking_rate / 2.0, 2.0)

# Calculate tone scores
excited_score = (
    0.30 * pitch_norm +       # High pitch
    0.25 * variance_norm +    # High variation
    0.25 * energy_norm +      # Loud
    0.20 * rate_norm          # Fast speaking
)

calm_score = (
    0.30 * (2.0 - pitch_norm) +      # Low pitch
    0.25 * (2.0 - variance_norm) +   # Low variation
    0.25 * (2.0 - energy_norm) +     # Quiet
    0.20 * (2.0 - rate_norm)         # Slow speaking
)

dramatic_score = (
    0.50 * min(pitch_range / 100.0, 2.0) +      # Wide pitch range
    0.50 * min(energy_variance / 0.05, 2.0)     # High energy variation
)

neutral_score = 1.0  # Baseline

# Select highest score
tone = max([excited, calm, dramatic, neutral], key=score)
confidence = min(score / 2.0, 1.0)
```

**Tone Categories**:
- `excited`: High pitch, high variance, high energy, fast rate
- `calm`: Low pitch, low variance, low energy, slow rate
- `dramatic`: High pitch range, high energy variance
- `neutral`: Middle values, baseline

**Segment Requirements**:
- Minimum segment duration: 300ms
- Minimum pitch samples: 5 valid frames

#### 2. Speaking Style Change Detection

**Algorithm**: Prosodic delta analysis between consecutive segments

**Delta Calculation**:
```python
# Compare consecutive toned segments
pitch_delta = abs(curr_pitch_mean - prev_pitch_mean)
energy_delta = abs(curr_energy_mean - prev_energy_mean)
rate_delta = abs(curr_speaking_rate - prev_speaking_rate)

# Normalize deltas
pitch_delta_norm = min(pitch_delta / 50.0, 1.0)     # 50Hz = max expected change
energy_delta_norm = min(energy_delta / 0.1, 1.0)    # 0.1 = max expected change
rate_delta_norm = min(rate_delta / 1.0, 1.0)        # 1.0 syllable/s = max change

# Calculate importance score
importance = (
    0.40 * pitch_delta_norm +     # Pitch change weight
    0.30 * energy_delta_norm +    # Energy change weight
    0.30 * rate_delta_norm        # Rate change weight
)
```

**Detection Threshold**:
```python
style_change_threshold = 0.3  # Minimum importance to flag transition
```

**Output Format**:
```python
{
    "time": float,              # Timestamp of new segment start
    "from_tone": str,           # Previous tone
    "to_tone": str,             # New tone
    "importance": float,        # 0.0-1.0
    "reason": "tone_shift",
    "metadata": {
        "pitch_delta": float,   # Raw Hz change
        "energy_delta": float,  # Raw RMS change
        "rate_delta": float     # Raw rate change
    }
}
```

#### 3. Narrative Context Analysis

**Model**: OpenAI GPT-4o-mini
- Model ID: `gpt-4o-mini`
- Temperature: `0.3` (lower for consistency)
- Max tokens: Default
- Timeout: `60.0` seconds

**Input Format**:
```python
# Timestamped transcript
"[0.5s] Welcome back to the channel\n"
"[3.2s] Today I'm going to show you something incredible\n"
"[5.8s] I can't believe this actually worked\n"
...
```

**Prompt Structure**:
```
System: "You are an expert at analyzing video transcripts to identify
         the most compelling moments for thumbnails. Return only valid JSON."

User: "Analyze this video transcript and identify the most important
       narrative moments for creating a compelling thumbnail.

Identify moments that are:
1. Hook statements (surprising, intriguing, attention-grabbing)
2. Emotional peaks (excitement, tension, revelation)
3. Story beats (key turning points)
4. Important reveals or information

Return JSON format:
[
  {"time": 5.2, "type": "hook_statement", "importance": 0.9,
   "text": "quote", "reason": "why it matters"},
  ...
]"
```

**Response Processing**:
- Parse JSON response (with markdown code block cleanup)
- Validate timestamps against video duration
- Filter out low-importance moments (< 0.5)

**Expected Output**:
```python
[
    {
        "time": 5.2,
        "type": "hook_statement",
        "importance": 0.9,
        "text": "I can't believe this actually worked",
        "reason": "introduces conflict/surprise"
    },
    {
        "time": 15.8,
        "type": "emotional_peak",
        "importance": 0.85,
        "text": "This is the moment I realized...",
        "reason": "story climax"
    }
]
```

**Performance**:
- Average latency: 2-5 seconds for 5min transcript
- Cost: ~$0.001-0.003 per analysis (GPT-4o-mini pricing)

#### 4. Unified Timeline Format

**Stream A Output**:
```python
{
    "time": float,           # Timestamp in seconds
    "type": str,             # "style_change", "hook_statement", "emotional_peak", etc.
    "score": float,          # Importance/confidence [0.0, 1.0]
    "source": "speech",      # Always "speech" for Stream A
    "metadata": dict         # Type-specific details
}
```

**Integration**:
- All Stream A detections (tone changes, narrative moments) converted to unified format
- Timeline sorted by timestamp
- Compatible with Stream B format for merging

---

### Stream B: Audio Saliency (Technical Details)

**Module**: `app/audio_saliency.py`

#### 1. Energy Peak Detection

**Algorithm**: Percentile-based loudness spike detection

**Input**: RMS energy array from `analyze_audio_features()`
- Frame length: 100ms
- Hop length: 50ms
- Sample rate: 16kHz

**Detection Logic**:
```python
# Calculate energy delta (rate of change)
energy_delta = abs(diff(energy, prepend=energy[0]))

# Percentile-based threshold
threshold = percentile(energy_delta, 90.0)  # Top 10%
threshold = max(threshold, energy_peak_threshold)  # Minimum 0.15

# Find peaks
peak_indices = where(energy_delta > threshold)

# Filter peaks too close together
min_peak_spacing = 0.5  # seconds

# Calculate peak ratio (current / previous)
if energy[idx-1] > 0:
    peak_ratio = energy[idx] / energy[idx-1]
else:
    peak_ratio = 2.0

# Normalize score
score = min(energy_delta[idx] / 0.3, 1.0)
```

**Parameters**:
```python
energy_peak_threshold = 0.15      # Minimum RMS delta
energy_peak_percentile = 90.0     # Top 10% of deltas
min_peak_spacing = 0.5            # seconds between peaks
score_normalization = 0.3         # Max expected delta
```

**Output**:
```python
{
    "time": 5.2,
    "type": "energy_peak",
    "score": 0.95,
    "source": "audio",
    "metadata": {
        "energy": 0.25,           # Current RMS
        "energy_delta": 0.18,     # Change in RMS
        "peak_ratio": 3.2         # Spike multiplier
    }
}
```

#### 2. Spectral Change Detection

**Algorithm**: Spectral flux analysis (timbre shift detection)

**Input**: Spectral centroid (brightness) from `analyze_audio_features()`

**Detection Logic**:
```python
# Calculate spectral flux (change in brightness)
spectral_delta = abs(diff(spectral_brightness, prepend=spectral_brightness[0]))

# Percentile-based threshold
threshold = percentile(spectral_delta, 85.0)  # Top 15%
threshold = max(threshold, spectral_change_threshold)  # Minimum 200.0 Hz

# Find significant changes
change_indices = where(spectral_delta > threshold)

# Filter changes too close together
min_change_spacing = 0.5  # seconds

# Determine brightness direction
if spectral_brightness[idx] > spectral_brightness[idx-1]:
    brightness_change = "brighter"  # Higher frequencies
else:
    brightness_change = "darker"    # Lower frequencies

# Normalize score
score = min(spectral_delta[idx] / 500.0, 1.0)
```

**Parameters**:
```python
spectral_change_threshold = 200.0   # Minimum centroid delta (Hz)
spectral_change_percentile = 85.0   # Top 15% of deltas
min_change_spacing = 0.5            # seconds between changes
score_normalization = 500.0         # Max expected delta (Hz)
```

**Output**:
```python
{
    "time": 12.8,
    "type": "spectral_change",
    "score": 0.82,
    "source": "audio",
    "metadata": {
        "centroid_delta": 250.5,       # Hz change
        "brightness_change": "darker",  # Direction
        "centroid_before": 2100.5,     # Hz before
        "centroid_after": 1850.0       # Hz after
    }
}
```

#### 3. Silence-to-Impact Detection

**Algorithm**: Pattern matching for quiet → loud dramatic moments

**Detection Logic**:
```python
# Scan for impact moments
for idx in range(silence_window, len(energy)):
    # Check if current moment is loud
    if energy[idx] < impact_threshold:
        continue  # Skip quiet moments

    # Check if previous window was quiet
    silence_window_energy = energy[idx - silence_window : idx]
    if mean(silence_window_energy) > silence_threshold:
        continue  # Not quiet enough

    # Find actual silence start (look back up to 2.5s)
    silence_start_idx = idx - silence_window
    for i in range(idx-1, max(0, idx-50), -1):
        if energy[i] > silence_threshold:
            silence_start_idx = i + 1
            break

    # Calculate silence duration
    silence_duration = times[idx] - times[silence_start_idx]

    # Skip very short silences
    if silence_duration < 0.3:
        continue

    # Calculate contrast ratio
    silence_avg = mean(energy[silence_start_idx:idx])
    if silence_avg > 0:
        contrast_ratio = energy[idx] / silence_avg
    else:
        contrast_ratio = 10.0  # Max

    # Score based on silence length + contrast
    silence_score = min(silence_duration / 2.0, 1.0)
    contrast_score = min(contrast_ratio / 10.0, 1.0)
    score = 0.6 * contrast_score + 0.4 * silence_score
```

**Parameters**:
```python
silence_threshold = 0.05        # Maximum RMS for silence
impact_threshold = 0.20         # Minimum RMS for impact
silence_window = 10             # Frames (~0.5s) to check
min_silence_duration = 0.3      # seconds
max_lookback = 50               # Frames (~2.5s)
min_impact_spacing = 1.0        # seconds between impacts

# Score weights
contrast_weight = 0.6
silence_duration_weight = 0.4
```

**Output**:
```python
{
    "time": 23.4,
    "type": "silence_to_impact",
    "score": 0.88,
    "source": "audio",
    "metadata": {
        "silence_duration": 1.2,      # seconds of quiet
        "impact_energy": 0.28,        # RMS at impact
        "contrast_ratio": 5.6,        # Impact / silence ratio
        "silence_start": 22.2         # When silence began
    }
}
```

#### 4. Non-Speech Sound Detection

**Algorithm**: High-energy sound classification outside speech segments

**Detection Logic**:
```python
# Create speech mask from VAD segments
speech_mask = zeros(len(times), dtype=bool)
for segment in speech_segments:
    start_idx = searchsorted(times, segment["start"])
    end_idx = searchsorted(times, segment["end"])
    speech_mask[start_idx:end_idx] = True

# Find high-energy threshold
high_energy_threshold = percentile(energy, 75)  # Top 25%

# Scan non-speech regions
for idx in range(len(times)):
    # Skip speech segments
    if speech_mask[idx]:
        continue

    # Skip low energy
    if energy[idx] < high_energy_threshold:
        continue

    # Classify sound type based on features
    if zcr[idx] > 0.1:              # High zero-crossing rate
        sound_type = "percussive"   # Drums, claps, hits
    elif energy[idx] > 0.2:         # Very high energy
        sound_type = "impact"       # Explosions, drops
    else:
        sound_type = "tonal"        # Music, sustained sounds

    # Normalize score
    score = min(energy[idx] / 0.3, 1.0)
```

**Parameters**:
```python
high_energy_percentile = 75.0      # Top 25%
min_sound_spacing = 0.5            # seconds between detections
percussive_zcr_threshold = 0.1     # ZCR for percussive classification
impact_energy_threshold = 0.2      # RMS for impact classification
score_normalization = 0.3          # Max expected energy
```

**Sound Type Classification**:
- `percussive`: High ZCR (> 0.1) = noisy, sharp attacks
- `impact`: Very high energy (> 0.2) = explosions, drops
- `tonal`: Otherwise = music, sustained sounds

**Output**:
```python
{
    "time": 8.7,
    "type": "non_speech_sound",
    "score": 0.75,
    "source": "audio",
    "metadata": {
        "sound_type": "percussive",  # Classification
        "energy": 0.22,              # RMS
        "zcr": 0.15                  # Zero-crossing rate
    }
}
```

#### 5. Unified Timeline Format

**Stream B Output**:
```python
{
    "time": float,           # Timestamp in seconds
    "type": str,             # "energy_peak", "spectral_change", etc.
    "score": float,          # Saliency [0.0, 1.0]
    "source": "audio",       # Always "audio" for Stream B
    "metadata": dict         # Type-specific details
}
```

**Integration**:
- All Stream B detections use same format
- Timeline sorted by timestamp
- Compatible with Stream A format for merging

---

### Performance Characteristics

**Stream A: Speech Semantics**
- Tone detection: ~100ms per segment (CPU, librosa pitch tracking)
- Style change detection: <10ms (simple delta calculation)
- Narrative analysis: 2-5s per transcript (GPT-4o-mini API call)
- **Total**: ~5-10s for 5min video (dominated by LLM call)

**Stream B: Audio Saliency**
- Energy peak detection: <50ms (numpy array operations)
- Spectral change detection: <50ms (numpy array operations)
- Silence-to-impact detection: <100ms (sliding window scan)
- Non-speech sound detection: <100ms (masked array scan)
- **Total**: <1s for 5min video (pure computation)

**Audio Feature Extraction** (prerequisite for Stream B):
- Librosa analysis: ~3-5s for 5min audio (CPU)
- Features: RMS, spectral centroid, pitch, ZCR, beats

**Speech Detection** (prerequisite for Stream A):
- Silero VAD: ~1-2s for 5min audio (CPU, PyTorch inference)
- Pitch variance filtering: ~0.5s per minute of speech

**Memory Usage**:
- Audio waveform (16kHz): ~960KB per minute
- Feature arrays: ~50KB per minute
- Model weights: Silero VAD (~2MB), FER+ (~100KB)

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

### Enhanced Endpoint (Current - with Stream A+B)

**Response now includes**:
```json
{
    "project_id": "550e8400-e29b-41d4-a716-446655440000",
    "frames": [...],
    "stream_a_moments": 12,
    "stream_b_moments": 18,
    "merged_moment_candidates": 25,
    "analysis_json_url": "gs://clickmoment-prod-assets/projects/{id}/analysis/adaptive_sampling_analysis.json",
    "pace_segments": [...],
    "processing_stats": {...},
    "summary": "..."
}
```

### Comprehensive Analysis JSON

**Saved to**: `gs://clickmoment-prod-assets/projects/{project_id}/analysis/adaptive_sampling_analysis.json`

**Complete structure**:
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_path": "gs://.../video.mp4",
  "processing_timestamp": "2025-01-28T12:34:56Z",
  "version": "1.0",

  "stream_a": {
    "enabled": true,
    "toned_segments": [
      {
        "start": 0.5,
        "end": 3.2,
        "tone": "excited",
        "confidence": 0.85,
        "features": {
          "pitch_mean": 220.5,
          "energy_mean": 0.15
        }
      }
    ],
    "style_changes": [
      {
        "time": 5.2,
        "from_tone": "calm",
        "to_tone": "excited",
        "importance": 0.75
      }
    ],
    "narrative_moments": [
      {
        "time": 5.2,
        "type": "hook_statement",
        "importance": 0.9,
        "text": "I can't believe this worked",
        "reason": "introduces surprise"
      }
    ],
    "timeline": [
      {
        "time": 5.2,
        "type": "hook_statement",
        "score": 0.9,
        "source": "speech",
        "metadata": {...}
      }
    ],
    "total_moments": 12
  },

  "stream_b": {
    "enabled": true,
    "energy_peaks": [
      {
        "time": 5.1,
        "type": "energy_peak",
        "score": 0.95,
        "source": "audio",
        "metadata": {
          "energy_delta": 0.18,
          "peak_ratio": 3.2
        }
      }
    ],
    "spectral_changes": [...],
    "silence_to_impact": [...],
    "non_speech_sounds": [...],
    "timeline": [...],
    "total_moments": 18
  },

  "visual_analysis": {
    "sample_frames": [
      {
        "timestamp": 5.2,
        "frame_path": "/path/to/frame.jpg",
        "has_face": true,
        "expression_intensity": 0.85,
        "eye_openness": 0.7,
        "mouth_openness": 0.3,
        "head_pose": {
          "pitch": 5.2,
          "yaw": -10.3,
          "roll": 2.1
        },
        "emotion_probs": {
          "neutral": 0.15,
          "happiness": 0.75,
          "surprise": 0.10
        }
      }
    ],
    "total_analyzed": 20,
    "faces_detected": 18
  },

  "merged_timeline": [
    {
      "time": 5.2,
      "score": 0.93,
      "sources": ["speech", "audio"],
      "types": ["hook_statement", "energy_peak"],
      "num_signals": 2,
      "features": {
        "speech": [...],
        "audio": [...],
        "visual": {...}
      },
      "metadata": {
        "cluster_size": 2,
        "cluster_moments": [...]
      }
    }
  ],

  "extracted_frames": [
    {
      "timestamp": 5.2,
      "frame_path": "gs://.../frame_5200ms.jpg",
      "moment_score": 0.93,
      "pace_score": 0.85,
      "pace_category": "high",
      "sources": ["speech", "audio"],
      "types": ["hook_statement", "energy_peak"]
    }
  ],

  "pace_analysis": {
    "segments": [...],
    "statistics": {...}
  },

  "processing_stats": {
    "audio_time": 12.3,
    "stream_analysis_time": 8.5,
    "total_time": 35.2
  },

  "audio_features": {
    "energy": [...],
    "spectral_brightness": [...],
    "times": [...]
  },

  "transcript": {
    "transcript": "Full transcript text...",
    "segments": [...],
    "duration": 120.5
  }
}
```

---

## File Structure

```
app/
├── ADAPTIVE_SAMPLING_DESIGN.md  ← This document
├── adaptive_sampling.py          ← Main orchestrator (✅ done, integrated Stream A+B)
├── audio_media.py                ← Audio extraction, transcription (✅ done)
├── speech_detection.py           ← Silero VAD, singing filter (✅ done)
├── speech_semantics.py           ← Stream A: Speech semantics (✅ done)
├── audio_saliency.py             ← Stream B: Audio saliency (✅ done)
├── analysis_output.py            ← JSON builder & timeline merger (✅ done)
├── face_analysis.py              ← MediaPipe + FER+ (✅ done)
├── pace_analysis.py              ← Multi-signal pace fusion (✅ done)
├── vision_media.py               ← Frame extraction utilities (✅ done)
└── main.py                       ← FastAPI endpoints (✅ done)
```

**Output Structure**:
```
gs://clickmoment-prod-assets/projects/{project_id}/
├── signals/
│   ├── audio/
│   │   ├── audio_speech.wav
│   │   └── audio_full.wav
│   └── frames/
│       ├── frame_0ms.jpg
│       ├── frame_5200ms.jpg
│       └── ...
└── analysis/
    └── adaptive_sampling_analysis.json  ← Comprehensive JSON output
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
