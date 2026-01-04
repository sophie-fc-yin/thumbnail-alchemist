# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================
import os

# GCS Buckets (consolidated - use these names throughout the codebase)
PROJECT_ASSETS_BUCKET = "clickmoment-prod-assets"
PROJECT_TEMP_BUCKET = "clickmoment-prod-temp"
# Aliases for backward compatibility and clarity
GCS_ASSETS_BUCKET = PROJECT_ASSETS_BUCKET
GCS_TEMP_BUCKET = PROJECT_TEMP_BUCKET

# ============================================================================
# VIDEO/AUDIO PROCESSING LIMITS
# ============================================================================
DEFAULT_MAX_DURATION_SECONDS = 1800  # 30 minutes - maximum video/audio duration for processing

# ============================================================================
# AUDIO EXTRACTION
# ============================================================================
DEFAULT_SAMPLE_RATE = 16000  # 16kHz for speech recognition
LOUDNORM_I = -16  # Integrated loudness target (LUFS)
LOUDNORM_TP = -1.5  # True peak (dBTP)
LOUDNORM_LRA = 11  # Loudness range (LU)

# ============================================================================
# SPEECH DETECTION (VAD Parameters)
# ============================================================================
# Silero VAD tuning for better speech detection:
# - Lower min_speech_duration_ms = more sensitive (catches short utterances)
# - Lower min_silence_duration_ms = less splitting at natural pauses
# - speech_pad_ms = padding around detected speech (standard: 30ms)
VAD_MIN_SPEECH_DURATION_MS = 100  # Reduced from 250ms for better sensitivity
VAD_MIN_SILENCE_DURATION_MS = 50  # Reduced from 100ms to avoid splitting at pauses
VAD_SPEECH_PAD_MS = 30  # Standard padding around speech segments

# Minimum segment duration for pitch analysis (needs longer segments than VAD)
MIN_PITCH_ANALYSIS_DURATION_MS = 300  # 300ms minimum for reliable pitch tracking

# Pitch analysis frequency range (Hz)
# Human voice range: ~80-400 Hz (male: 80-180, female: 165-400, children: up to 500)
# These bounds help filter out harmonics and noise, focus on fundamental frequency
PITCH_ANALYSIS_FMIN = 80  # Minimum frequency (Hz) - typical male voice lower bound
PITCH_ANALYSIS_FMAX = 400  # Maximum frequency (Hz) - typical female/child voice upper bound

# Legacy constants (kept for backward compatibility if needed)
SPEECH_CONFIDENCE_THRESHOLD = 0.5
SPEECH_ENERGY_THRESHOLD = 0.03
SPEECH_PITCH_VARIANCE_THRESHOLD = 5000.0  # typical speech: 100-10000

# ============================================================================
# TRANSCRIPTION
# ============================================================================
MAX_TRANSCRIPTION_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0  # seconds
TRANSCRIPTION_TIMEOUT = 300.0  # 5 minutes

# ============================================================================
# AUDIO FEATURES
# ============================================================================
AUDIO_FRAME_LENGTH_SECONDS = 0.1  # 100ms frames
AUDIO_HOP_LENGTH_SECONDS = 0.05  # 50ms hop (50% overlap)
PITCH_VARIANCE_WINDOW_SIZE = 20  # ~1 second windows

# ============================================================================
# TEXT IMPORTANCE
# ============================================================================
TEXT_IMPORTANCE_CLAIM = 1.0
TEXT_IMPORTANCE_EMPHASIS = 0.7
TEXT_IMPORTANCE_NEUTRAL = 0.4
TEXT_IMPORTANCE_FILLER = 0.1
TEXT_IMPORTANCE_MIN_CLAIM_WORDS = 10

# ============================================================================
# EMPHASIS SCORE
# ============================================================================
EMPHASIS_WINDOW_SECONDS = 5.0  # Window for rolling baseline
EMPHASIS_MIN_STD = 1e-6  # Avoid division by zero

# ============================================================================
# BGM PENALTY
# ============================================================================
BGM_MAX_VARIANCE = 0.05  # Threshold for max penalty
BGM_MIN_PENALTY = 0.2  # Minimum penalty multiplier
BGM_MAX_PENALTY = 1.0  # Maximum penalty multiplier

# ============================================================================
# TIMELINE EVENTS
# ============================================================================
SILENCE_THRESHOLD = 0.02  # Energy threshold for silence
MIN_PAUSE_DURATION = 1.0  # Only track pauses > 1 second
ENERGY_PEAK_PERCENTILE = 90  # Top 10% energy
ENERGY_PEAK_WINDOW_SIZE = 20  # ~1 second window
ENERGY_PEAK_MIN_SPACING = 2.0  # Minimum spacing between peaks (seconds)

# ============================================================================
# MUSIC DETECTION
# ============================================================================
MUSIC_ENERGY_THRESHOLD = 0.1
MUSIC_SCORE_THRESHOLD = 0.5
MUSIC_MIN_DURATION = 3.0  # Minimum 3 seconds

# ============================================================================
# DEFAULT KEYWORD LISTS
# ============================================================================
DEFAULT_CLAIM_KEYWORDS = [
    "discovered",
    "revealed",
    "found",
    "proved",
    "demonstrated",
    "breakthrough",
    "amazing",
    "incredible",
    "shocking",
    "secret",
    "truth",
    "fact",
    "evidence",
    "research shows",
    "study found",
    "this is",
    "here's why",
    "the reason",
    "turns out",
]

DEFAULT_EMPHASIS_KEYWORDS = [
    "but",
    "however",
    "although",
    "instead",
    "actually",
    "important",
    "key",
    "crucial",
    "critical",
    "essential",
    "remember",
    "note",
    "pay attention",
    "listen",
    "focus",
    "versus",
    "compared to",
    "unlike",
    "difference",
]

DEFAULT_FILLER_KEYWORDS = [
    "um",
    "uh",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "basically",
    "literally",
    "just",
]

# ============================================================================
# EMOTION DIMENSION TO TONE MAPPING
# ============================================================================
# Thresholds for mapping emotion dimensions (arousal, valence, dominance) to categorical tones
# These are heuristic values that should be tuned based on empirical testing
# Emotion dimensions are normalized to [0, 1] range:
# - arousal: 0=calm, 1=excited
# - valence: 0=negative, 1=positive
# - dominance: 0=submissive, 1=dominant

# Excited tone: high arousal + positive valence
EMOTION_EXCITED_AROUSAL_THRESHOLD = 0.6  # High arousal threshold
EMOTION_EXCITED_VALENCE_THRESHOLD = 0.5  # Positive valence threshold

# Calm tone: low arousal + neutral valence
EMOTION_CALM_AROUSAL_THRESHOLD = 0.4  # Low arousal threshold
EMOTION_CALM_VALENCE_NEUTRAL_RANGE = 0.2  # ±0.2 around 0.5 for neutral valence

# Dramatic tone: high arousal OR high dominance
EMOTION_DRAMATIC_AROUSAL_THRESHOLD = 0.6  # High arousal threshold
EMOTION_DRAMATIC_DOMINANCE_THRESHOLD = 0.7  # High dominance threshold

# Neutral tone: baseline confidence (used when no other category matches)
EMOTION_NEUTRAL_CONFIDENCE = 0.5

# Prosodic feature thresholds for tone classification
# Pitch normalization (Hz): typical range 100-300 Hz
TONE_PITCH_NORMALIZATION = 200.0  # 100Hz=low, 300Hz=high
TONE_PITCH_VARIANCE_NORMALIZATION = 50.0  # Typical variance range
TONE_PITCH_RANGE_NORMALIZATION = 100.0  # Typical range (Hz)

# Energy normalization (RMS)
TONE_ENERGY_NORMALIZATION = 0.2  # Typical RMS energy range
TONE_ENERGY_VARIANCE_NORMALIZATION = 0.05  # Typical energy variance

# Weight for combining emotion dimensions vs prosodic features
# Higher = more weight on emotion dimensions (ML model)
# Lower = more weight on prosodic features (pitch/energy)
TONE_EMOTION_WEIGHT = 0.7  # 70% emotion dimensions
TONE_PROSODIC_WEIGHT = 0.3  # 30% prosodic features

# ============================================================================
# SEGMENT IMPORTANCE CALCULATION
# ============================================================================
# Base importance scores by tone
TONE_IMPORTANCE_EXCITED = 0.8
TONE_IMPORTANCE_DRAMATIC = 0.75
TONE_IMPORTANCE_CALM = 0.4
TONE_IMPORTANCE_NEUTRAL = 0.3

# Emotion dimension boost weights (for importance calculation)
EMOTION_AROUSAL_WEIGHT = 0.6  # Arousal is most important
EMOTION_VALENCE_WEIGHT = 0.3
EMOTION_DOMINANCE_WEIGHT = 0.1
EMOTION_BOOST_WEIGHT = 0.2  # Max boost from emotion dimensions (0.2)

# Narrative context boost weight
NARRATIVE_BOOST_WEIGHT = 0.3  # Max boost from narrative moments (0.3)

# Final importance calculation weights
# These control how much each component contributes to the final importance score
IMPORTANCE_BASE_WEIGHT = 0.6  # Weight for base tone importance
IMPORTANCE_EMOTION_WEIGHT = 0.2  # Weight for emotion boost
IMPORTANCE_NARRATIVE_WEIGHT = 0.2  # Weight for narrative boost

# ============================================================================
# ADAPTIVE SAMPLING / FRAME EXTRACTION
# ============================================================================
# Pipeline Configuration
DEFAULT_MAX_FRAMES = 100
INITIAL_SAMPLE_RATIO = 5  # Sample 1/5 of max frames initially (min 20)
IMPORTANCE_SEGMENTATION_THRESHOLD = 0.2  # Importance change threshold for segmentation

# Dynamic max_frames calculation
FRAMES_PER_MINUTE = 10  # Target density: 1 frame every 6 seconds
MAX_VIDEO_LENGTH_MINUTES = 15  # Cap max_frames at 15 minutes worth
CALCULATED_MAX_FRAMES_CAP = FRAMES_PER_MINUTE * MAX_VIDEO_LENGTH_MINUTES  # 150 frames
MIN_FRAMES_FOR_ANALYSIS = 20  # Minimum frames for proper analysis

# Frame Extraction
DEFAULT_FRAME_INTERVAL_FALLBACK = 2.0  # seconds - fallback interval if importance calculation fails
SAMPLE_INTERVAL_SECONDS = 3.0  # Initial sparse sampling: 1 frame every 3 seconds
FFMPEG_OUTPUT_PATTERN = "sample_%03d.jpg"
SEGMENT_FRAME_PATTERN = "frame_%03d.jpg"

# Adaptive Sampling Interval Formula (Normal Distribution)
ADAPTIVE_SAMPLE_INTERVAL_MEAN = 3.0  # Mean/median interval in seconds (normal distribution center)
ADAPTIVE_SAMPLE_INTERVAL_STD_DEV = 1.5  # Standard deviation for normal distribution
ADAPTIVE_SAMPLE_INTERVAL_REF_DURATION = (
    240.0  # 4 minutes - reference duration for z-score calculation
)
ADAPTIVE_SAMPLE_INTERVAL_SCALE = 2.0  # Controls sensitivity of interval to duration changes
ADAPTIVE_SAMPLE_INTERVAL_MIN = 1.0  # Minimum interval (very short videos)
ADAPTIVE_SAMPLE_INTERVAL_MAX = 7.0  # Maximum interval (very long videos)

# ============================================================================
# ADAPTIVE SAMPLING INTERVALS (New System)
# ============================================================================
# Base intervals (before adaptive scaling based on video duration)
# PERFORMANCE OPTIMIZATION: Increased intervals to extract ~25-30% fewer frames
# This reduces vision analysis workload while maintaining quality for top moments
BASE_DENSE_INTERVAL_CRITICAL = 0.4  # Critical importance base: ~2.5 fps (was 0.3)
BASE_DENSE_INTERVAL_HIGH = 0.6  # High importance base: ~1.67 fps (was 0.5)
BASE_DENSE_INTERVAL_MEDIUM = 1.2  # Medium importance base: ~0.83 fps (was 1.0)
BASE_DENSE_INTERVAL_LOW = 2.0  # Low importance base: 0.5 fps (unchanged, rarely used)
BASE_DENSE_INTERVAL_MINIMAL = 3.0  # Minimal importance base: 0.33 fps (unchanged, rarely used)

# Adaptive interval scaling thresholds
ADAPTIVE_INTERVAL_SHORT_THRESHOLD = 120.0  # < 2 min: no scaling (scale = 1.0)
ADAPTIVE_INTERVAL_MEDIUM_THRESHOLD = 600.0  # 2-10 min: linear scale 1.0x → 2.0x
ADAPTIVE_INTERVAL_SCALE_MAX = 3.0  # Maximum scale factor (caps at 3.0x for long videos)

# Video FPS detection
DEFAULT_VIDEO_FPS = 30.0  # Fallback FPS if detection fails
FPS_MULTIPLIER_MIN_INTERVAL = 1.5  # Never sample faster than 1.5x video frame duration
FPS_VALID_RANGE_MIN = 15.0  # Minimum valid FPS
FPS_VALID_RANGE_MAX = 120.0  # Maximum valid FPS

# Short segment handling
SHORT_SEGMENT_THRESHOLD = 1.0  # Extract single frame at midpoint if segment < 1s

# ============================================================================
# DEPRECATED: Legacy Fixed Intervals (use BASE_* constants with adaptive scaling)
# ============================================================================
DENSE_INTERVAL_CRITICAL = (
    0.3  # DEPRECATED: Use BASE_DENSE_INTERVAL_CRITICAL with get_adaptive_intervals()
)
DENSE_INTERVAL_HIGH = 0.5  # DEPRECATED: Use BASE_DENSE_INTERVAL_HIGH with get_adaptive_intervals()
DENSE_INTERVAL_MEDIUM = (
    1.0  # DEPRECATED: Use BASE_DENSE_INTERVAL_MEDIUM with get_adaptive_intervals()
)
DENSE_INTERVAL_LOW = 2.0  # DEPRECATED: Use BASE_DENSE_INTERVAL_LOW with get_adaptive_intervals()
DENSE_INTERVAL_MINIMAL = (
    3.0  # DEPRECATED: Use BASE_DENSE_INTERVAL_MINIMAL with get_adaptive_intervals()
)
MIN_DENSE_SAMPLING_INTERVAL = 0.3  # DEPRECATED: No longer enforced as hard limit

# Segment Duration Limits
MIN_IMPORTANCE_SEGMENT_DURATION = 1.0  # Minimum duration for importance segments (seconds) - merges smaller segments (10x extraction threshold)
MAX_IMPORTANCE_SEGMENT_DURATION = (
    10.0  # Maximum duration for importance segments (seconds) - prevents overly long segments
)
VISUAL_CHANGE_LOW_THRESHOLD = (
    0.3  # Below this, merge segments (low visual change = longer segments)
)
VISUAL_CHANGE_HIGH_THRESHOLD = (
    0.4  # Above this, create boundaries (lowered from 0.6 to detect more shot changes)
)
VISUAL_CHANGE_SEPARATE_THRESHOLD = (
    0.6  # Above this, keep segments separate (high visual change = shorter segments)
)

# Parallel Processing Limits
MAX_CONCURRENT_SEGMENT_EXTRACTIONS = 10  # Maximum concurrent ffmpeg processes
MIN_SEGMENT_DURATION_FOR_EXTRACTION = 0.1  # Skip segments shorter than this (seconds)

# ============================================================================
# TWO-STAGE FILTERING (Performance Optimization)
# ============================================================================
# Stage 1: Quick filtering (face detection + basic quality) on all frames
# Stage 2: Full analysis (emotion, aesthetics, composition) only on top candidates
#
# The keep_ratio controls the tradeoff between speed and quality:
# - Higher ratio (0.70-0.80): Safer, less filtering, ~1.3-1.5x speedup
# - Medium ratio (0.55-0.65): Balanced, moderate filtering, ~1.7-2x speedup
# - Lower ratio (0.40-0.50): Aggressive, more filtering, ~2-2.5x speedup
#
# Recommendation: Start at 0.65 (conservative), then tune down if quality is maintained
TWO_STAGE_KEEP_RATIO = 0.70  # Keep top 65% of frames for Stage 2 full analysis
TWO_STAGE_MIN_FRAMES = 20  # Minimum frames to analyze (safety floor)
TWO_STAGE_MAX_FRAMES = 60  # Maximum frames to analyze (performance ceiling)

# Local Storage Paths
# On Cloud Run, use /tmp (only writable directory)
# In local/dev, use relative path for easier debugging
LOCAL_MEDIA_DIR = os.getenv(
    "LOCAL_MEDIA_DIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "thumbnail-alchemist-media")
)
TEMP_DIR_NAME = "temp"
SAMPLE_FRAMES_DIR = "sample_frames"
DOWNLOADED_SAMPLES_DIR = "downloaded_samples"
DERIVED_MEDIA_DIR = "derived-media"
FRAMES_DIR = "frames"
