# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================
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

# Local Storage Paths
LOCAL_MEDIA_DIR = "thumbnail-alchemist-media"
TEMP_DIR_NAME = "temp"
SAMPLE_FRAMES_DIR = "sample_frames"
DOWNLOADED_SAMPLES_DIR = "downloaded_samples"
DERIVED_MEDIA_DIR = "derived-media"
FRAMES_DIR = "frames"
