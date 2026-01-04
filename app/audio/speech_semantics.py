"""Stream A: Speech Semantics Analysis

Analyzes the semantic and narrative importance of speech to identify
moments that matter for thumbnail selection.

Key Functions:
- Tone detection (emotion classification on speech)
- Narrative context (important sentences via LLM)
- Speaking style changes (calm → excited transitions)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from google.genai import Client
from transformers import AutoModelForAudioClassification, AutoProcessor

from app.constants import (
    DEFAULT_SAMPLE_RATE,
    EMOTION_AROUSAL_WEIGHT,
    EMOTION_BOOST_WEIGHT,
    EMOTION_CALM_AROUSAL_THRESHOLD,
    EMOTION_CALM_VALENCE_NEUTRAL_RANGE,
    EMOTION_DOMINANCE_WEIGHT,
    EMOTION_DRAMATIC_AROUSAL_THRESHOLD,
    EMOTION_DRAMATIC_DOMINANCE_THRESHOLD,
    EMOTION_EXCITED_AROUSAL_THRESHOLD,
    EMOTION_EXCITED_VALENCE_THRESHOLD,
    EMOTION_NEUTRAL_CONFIDENCE,
    EMOTION_VALENCE_WEIGHT,
    IMPORTANCE_BASE_WEIGHT,
    IMPORTANCE_EMOTION_WEIGHT,
    IMPORTANCE_NARRATIVE_WEIGHT,
    MIN_PITCH_ANALYSIS_DURATION_MS,
    NARRATIVE_BOOST_WEIGHT,
    PITCH_ANALYSIS_FMAX,
    PITCH_ANALYSIS_FMIN,
    TONE_EMOTION_WEIGHT,
    TONE_ENERGY_NORMALIZATION,
    TONE_ENERGY_VARIANCE_NORMALIZATION,
    TONE_IMPORTANCE_CALM,
    TONE_IMPORTANCE_DRAMATIC,
    TONE_IMPORTANCE_EXCITED,
    TONE_IMPORTANCE_NEUTRAL,
    TONE_PITCH_NORMALIZATION,
    TONE_PITCH_RANGE_NORMALIZATION,
    TONE_PITCH_VARIANCE_NORMALIZATION,
    TONE_PROSODIC_WEIGHT,
)

logger = logging.getLogger(__name__)

# Singleton instance for model reuse
_speech_semantic_analyzer_instance: SpeechSemanticAnalyzer | None = None


def get_speech_semantic_analyzer() -> SpeechSemanticAnalyzer:
    """Get or create singleton SpeechSemanticAnalyzer instance.

    Returns:
        Shared SpeechSemanticAnalyzer instance with loaded models.
    """
    global _speech_semantic_analyzer_instance
    if _speech_semantic_analyzer_instance is None:
        logger.debug("Creating SpeechSemanticAnalyzer singleton instance")
        _speech_semantic_analyzer_instance = SpeechSemanticAnalyzer()
    return _speech_semantic_analyzer_instance


class SpeechSemanticAnalyzer:
    """
    Analyze speech for semantic importance and narrative value.

    Stream A focuses on WHAT is being said and WHY it matters:
    - Hook statements
    - Emotional peaks
    - Story beats
    - Important reveals
    """

    def __init__(self):
        """Initialize the analyzer with ML model for emotion detection."""
        self._emotion_model = None
        self._emotion_processor = None

    def _load_emotion_model(self):
        """Lazy load the emotion recognition model."""
        if self._emotion_model is None:
            model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self._emotion_model = AutoModelForAudioClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._emotion_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._emotion_model.eval()  # Set to evaluation mode
        return self._emotion_model, self._emotion_processor

    def detect_tone_on_segments(
        self,
        audio_path: Path | str,
        speech_segments: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        """
        Detect speaking tone for each speech segment.

        Analyzes prosody (pitch, energy, tempo) to classify tone:
        - Calm/neutral
        - Excited/enthusiastic
        - Serious/dramatic
        - Humorous/playful

        Args:
            audio_path: Path to audio file
            speech_segments: Speech segments from VAD with timestamps

        Returns:
            List of segments with tone classification:
            [
                {
                    "start": 0.5,
                    "end": 3.2,
                    "tone": "excited",
                    "confidence": 0.85,
                    "features": {
                        "pitch_mean": 220.5,
                        "pitch_variance": 45.2,
                        "energy_mean": 0.15,
                        "speaking_rate": 3.2  # syllables per second
                    }
                },
                ...
            ]
        """
        toned_segments = []

        # Load audio for segment-level analysis
        y, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE, mono=True)

        for segment in speech_segments:
            # Handle both start_ms/end_ms and start/end formats (transcript segments may use either)
            start = (
                segment.get("start_ms", segment.get("start", 0)) / 1000.0
                if segment.get("start_ms")
                else segment.get("start", 0)
            )
            end = (
                segment.get("end_ms", segment.get("end", 0)) / 1000.0
                if segment.get("end_ms")
                else segment.get("end", 0)
            )

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = y[start_sample:end_sample]

            # Check if segment is long enough for pitch analysis
            min_samples = int(sr * MIN_PITCH_ANALYSIS_DURATION_MS / 1000.0)
            if len(segment_audio) < min_samples:
                # Too short - use default neutral values
                toned_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "tone": "neutral",
                        "confidence": EMOTION_NEUTRAL_CONFIDENCE,
                        "emotion_dimensions": {"arousal": 0.5, "valence": 0.5, "dominance": 0.5},
                        "features": {
                            "pitch_mean": 0.0,
                            "pitch_variance": 0.0,
                            "pitch_range": 0.0,
                            "energy_mean": 0.0,
                            "energy_variance": 0.0,
                        },
                    }
                )
                continue

            # Extract prosodic features for this segment (within human voice range)
            pitches, magnitudes = librosa.piptrack(
                y=segment_audio, sr=sr, fmin=PITCH_ANALYSIS_FMIN, fmax=PITCH_ANALYSIS_FMAX
            )

            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) < 5:
                # Insufficient pitch data - use default neutral values with available energy
                rms = librosa.feature.rms(y=segment_audio)[0]
                energy_mean = float(np.mean(rms))
                energy_variance = float(np.var(rms))

                toned_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "tone": "neutral",
                        "confidence": EMOTION_NEUTRAL_CONFIDENCE,
                        "emotion_dimensions": {"arousal": 0.5, "valence": 0.5, "dominance": 0.5},
                        "features": {
                            "pitch_mean": 0.0,
                            "pitch_variance": 0.0,
                            "pitch_range": 0.0,
                            "energy_mean": energy_mean,
                            "energy_variance": energy_variance,
                        },
                    }
                )
                continue

            # Calculate features
            pitch_mean = float(np.mean(pitch_values))
            pitch_variance = float(np.var(pitch_values))
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values))

            # Energy
            rms = librosa.feature.rms(y=segment_audio)[0]
            energy_mean = float(np.mean(rms))
            energy_variance = float(np.var(rms))

            # Detect emotion intensity using ML model
            emotion_dims = self._detect_emotion_intensity(
                segment_audio=segment_audio,
                sample_rate=sr,
            )

            # Classify tone using emotion dimensions + prosodic features
            tone, confidence = self._classify_tone(
                emotion_dimensions=emotion_dims,
                pitch_mean=pitch_mean,
                pitch_variance=pitch_variance,
                pitch_range=pitch_range,
                energy_mean=energy_mean,
                energy_variance=energy_variance,
            )

            toned_segments.append(
                {
                    "start": start,  # Use normalized start/end from above
                    "end": end,
                    "tone": tone,
                    "confidence": confidence,
                    "emotion_dimensions": emotion_dims,
                    "features": {
                        "pitch_mean": pitch_mean,
                        "pitch_variance": pitch_variance,
                        "pitch_range": pitch_range,
                        "energy_mean": energy_mean,
                        "energy_variance": energy_variance,
                    },
                }
            )

        return toned_segments

    def _detect_emotion_intensity(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """
        Detect emotion intensity using ML model (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim).

        The model outputs dimensional emotion predictions:
        - Arousal: calm (low) to excited (high)
        - Valence: negative to positive
        - Dominance: submissive (low) to dominant (high)

        Args:
            segment_audio: Audio samples for the segment
            sample_rate: Sample rate of the audio

        Returns:
            Dictionary with emotion dimensions:
            {
                "arousal": float,    # 0=calm, 1=excited
                "valence": float,     # 0=negative, 1=positive
                "dominance": float,   # 0=submissive, 1=dominant
            }
        """
        try:
            model, processor = self._load_emotion_model()

            # Convert to tensor and ensure correct format
            if isinstance(segment_audio, np.ndarray):
                audio_tensor = torch.from_numpy(segment_audio).float()
            else:
                audio_tensor = segment_audio.float()

            # Ensure mono and correct shape
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()

            # Process audio for the model (resample if needed, model expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)

            # Process with feature extractor
            inputs = processor(
                audio_tensor.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits.squeeze().cpu().numpy()

            # Map logits using model labels when available (audeering uses V/A/D).
            id_to_label = getattr(model.config, "id2label", {}) or {}
            label_to_value = {
                str(id_to_label.get(idx, idx)).lower(): float(value)
                for idx, value in enumerate(predictions)
            }
            arousal = label_to_value.get("arousal", predictions[0] if len(predictions) > 0 else 0.0)
            valence = label_to_value.get("valence", predictions[1] if len(predictions) > 1 else 0.0)
            dominance = label_to_value.get(
                "dominance", predictions[2] if len(predictions) > 2 else 0.0
            )

            # Normalize to [0, 1] range if needed (model may output [-1, 1])
            arousal_norm = (arousal + 1.0) / 2.0 if arousal < 0 else arousal
            valence_norm = (valence + 1.0) / 2.0 if valence < 0 else valence
            dominance_norm = (dominance + 1.0) / 2.0 if dominance < 0 else dominance

            emotion_dims = {
                "arousal": float(arousal_norm),  # 0=calm, 1=excited
                "valence": float(valence_norm),  # 0=negative, 1=positive
                "dominance": float(dominance_norm),  # 0=submissive, 1=dominant
            }

            return emotion_dims

        except Exception as e:
            # Fallback to neutral if model fails
            logger.warning("Emotion detection failed: %s, using neutral", e)
            return {
                "arousal": 0.5,
                "valence": 0.5,
                "dominance": 0.5,
            }

    def _classify_tone(
        self,
        emotion_dimensions: dict[str, float],
        pitch_mean: float,
        pitch_variance: float,
        pitch_range: float,
        energy_mean: float,
        energy_variance: float,
    ) -> tuple[str, float]:
        """
        Classify tone category using emotion dimensions + prosodic features.

        Combines ML-based emotion dimensions with traditional prosodic features
        (pitch, energy) for more robust tone classification.

        Args:
            emotion_dimensions: Dict with arousal, valence, dominance (from ML model)
            pitch_mean: Average pitch (Hz)
            pitch_variance: Pitch variance (Hz²)
            pitch_range: Pitch range (Hz)
            energy_mean: Average energy (RMS)
            energy_variance: Energy variance

        Returns:
            Tuple of (tone, confidence):
            - tone: "excited", "calm", "dramatic", or "neutral"
            - confidence: Confidence score [0, 1]
        """
        arousal = emotion_dimensions.get("arousal", 0.5)
        valence = emotion_dimensions.get("valence", 0.5)
        dominance = emotion_dimensions.get("dominance", 0.5)

        # Normalize prosodic features for scoring
        pitch_norm = min(pitch_mean / TONE_PITCH_NORMALIZATION, 2.0)  # 100Hz=low, 300Hz=high
        pitch_variance_norm = min(pitch_variance / TONE_PITCH_VARIANCE_NORMALIZATION, 2.0)
        pitch_range_norm = min(pitch_range / TONE_PITCH_RANGE_NORMALIZATION, 2.0)
        energy_norm = min(energy_mean / TONE_ENERGY_NORMALIZATION, 2.0)
        energy_variance_norm = min(energy_variance / TONE_ENERGY_VARIANCE_NORMALIZATION, 2.0)

        # Calculate emotion-based scores (from ML model)
        excited_emotion_score = (
            0.4 * arousal + 0.3 * valence + 0.3 * dominance
            if arousal > EMOTION_EXCITED_AROUSAL_THRESHOLD
            and valence > EMOTION_EXCITED_VALENCE_THRESHOLD
            else 0.0
        )

        calm_emotion_score = (
            0.5 * (1.0 - arousal) + 0.5 * (1.0 - abs(valence - 0.5) * 2.0)
            if arousal < EMOTION_CALM_AROUSAL_THRESHOLD
            and abs(valence - 0.5) < EMOTION_CALM_VALENCE_NEUTRAL_RANGE
            else 0.0
        )

        dramatic_emotion_score = (
            0.5 * arousal + 0.5 * dominance
            if arousal > EMOTION_DRAMATIC_AROUSAL_THRESHOLD
            or dominance > EMOTION_DRAMATIC_DOMINANCE_THRESHOLD
            else 0.0
        )

        neutral_emotion_score = 0.5  # Baseline

        # Calculate prosodic-based scores (from pitch/energy)
        excited_prosodic_score = (
            0.3 * pitch_norm
            + 0.25 * pitch_variance_norm
            + 0.25 * energy_norm
            + 0.2 * energy_variance_norm
        )

        calm_prosodic_score = (
            0.3 * (2.0 - pitch_norm)
            + 0.25 * (2.0 - pitch_variance_norm)
            + 0.25 * (2.0 - energy_norm)
            + 0.2 * (2.0 - energy_variance_norm)
        )

        dramatic_prosodic_score = 0.5 * pitch_range_norm + 0.5 * energy_variance_norm

        neutral_prosodic_score = 1.0  # Baseline

        # Combine emotion and prosodic scores with weights
        excited_score = (
            TONE_EMOTION_WEIGHT * excited_emotion_score
            + TONE_PROSODIC_WEIGHT * excited_prosodic_score
        )

        calm_score = (
            TONE_EMOTION_WEIGHT * calm_emotion_score + TONE_PROSODIC_WEIGHT * calm_prosodic_score
        )

        dramatic_score = (
            TONE_EMOTION_WEIGHT * dramatic_emotion_score
            + TONE_PROSODIC_WEIGHT * dramatic_prosodic_score
        )

        neutral_score = (
            TONE_EMOTION_WEIGHT * neutral_emotion_score
            + TONE_PROSODIC_WEIGHT * neutral_prosodic_score
        )

        # Select tone with highest combined score
        scores = {
            "excited": excited_score,
            "calm": calm_score,
            "dramatic": dramatic_score,
            "neutral": neutral_score,
        }

        tone = max(scores, key=scores.get)
        confidence = min(scores[tone] / 2.0, 1.0)  # Normalize to [0, 1]

        return tone, confidence


async def analyze_narrative_context(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Analyze transcript for narrative importance using LLM.

    Identifies:
    - Hook statements ("I can't believe...", "This changed everything...")
    - Emotional peaks (excitement, surprise, tension)
    - Story beats (setup, conflict, resolution)
    - Key reveals or information

    Args:
        transcript: Full transcript text
        segments: Transcript segments with timestamps

    Returns:
        List of important narrative moments (same format as input segments):
        [
            {
                "start": 5.0,
                "end": 7.2,
                "type": "hook_statement",
                "importance": 0.9,
                "text": "I can't believe this actually worked",
                "reason": "introduces conflict/surprise"
            },
            ...
        ]
    """
    # Prepare transcript with segment indices for precise reference
    timestamped_transcript = ""
    for idx, segment in enumerate(segments):
        start_time = segment.get("start_ms", segment.get("start", 0)) / 1000.0
        text = segment.get("text", "")
        timestamped_transcript += f"[Segment {idx}] [{start_time:.1f}s] {text}\n"

    # DEBUG: Log transcript structure
    logger.debug("Building narrative context from %d segments", len(segments))
    if segments:
        logger.debug("Sample segment structure: %s", segments[0])
    logger.debug("Timestamped transcript length: %d chars", len(timestamped_transcript))

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, skipping narrative context analysis")
            return []

        client = Client(api_key=api_key)

        # Create comprehensive, context-aware prompt
        full_prompt = f"""You are analyzing a complete video transcript to score EVERY segment for narrative importance in thumbnail selection.

CRITICAL REQUIREMENTS:
1. You MUST score ALL {len(segments)} segments (0 through {len(segments) - 1})
2. Return exactly {len(segments)} entries in the JSON array - one for each segment
3. Reference segments by their SEGMENT NUMBER only
4. DO NOT skip any segments - score every single one

TRANSCRIPT WITH SEGMENT NUMBERS:
{timestamped_transcript}

TASK:

STEP 1: Understand the Complete Story
First, read and understand the ENTIRE transcript to grasp:
- The complete narrative arc and story structure
- The overall emotional journey and tone
- Key themes, conflicts, and resolutions
- The context and flow of the entire video

STEP 2: Score EVERY Segment
Score each segment's narrative importance relative to the overall story:
- How compelling is this segment compared to others?
- Does it contain a hook, emotional peak, story beat, or reveal?
- Would this segment make a good thumbnail moment?

Classify each segment:
- "hook_statement": Surprising, intriguing, attention-grabbing statements
- "emotional_peak": High excitement, tension, surprise, emotional intensity
- "story_beat": Key turning points, conflicts, resolutions, narrative shifts
- "reveal": Critical information, discoveries, plot developments
- "neutral": Standard content without special narrative significance

For EACH segment (0 through {len(segments) - 1}), provide:
- segment_index: The segment number (must include ALL segments 0-{len(segments) - 1})
- type: One of: "hook_statement", "emotional_peak", "story_beat", "reveal", "neutral"
- importance: Score 0.0-1.0 based on narrative importance for thumbnails
  - 0.0-0.3: Low importance (filler, transitions, standard content)
  - 0.4-0.6: Medium importance (relevant but not peak moments)
  - 0.7-0.9: High importance (compelling, attention-grabbing)
  - 1.0: Critical moment (must-see, defining moment)
- reason: Brief explanation of the score

Return ONLY valid JSON with exactly {len(segments)} entries (no markdown, no code blocks):
[
  {{"segment_index": 0, "type": "neutral", "importance": 0.3, "reason": "opening context"}},
  {{"segment_index": 1, "type": "hook_statement", "importance": 0.7, "reason": "surprising statement"}},
  ...
  {{"segment_index": {len(segments) - 1}, "type": "neutral", "importance": 0.2, "reason": "closing remark"}}
]"""

        response = client.models.generate_content(
            model="gemini-2.5-pro",  # Pro model for better instruction following
            contents=full_prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,  # Zero temperature - strict adherence only
            },
        )

        # Parse response
        result_text = response.text.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(line for line in lines if not line.startswith("```"))

        narrative_moments = json.loads(result_text)

        # DEBUG: Log what LLM returned
        logger.debug("LLM returned %d narrative moments", len(narrative_moments))
        if narrative_moments:
            logger.debug("First moment sample: %s", narrative_moments[0])
            # Log all segment indices to detect if LLM is providing valid references
            segment_indices = [m.get("segment_index", -1) for m in narrative_moments]
            logger.debug("All LLM segment indices: %s", segment_indices)

        # VALIDATION: Verify segment indices are valid and extract segment data
        validated_moments = []

        for idx, moment in enumerate(narrative_moments):
            segment_index = moment.get("segment_index", -1)

            logger.debug(
                "Processing moment %d/%d: segment_index=%d, type=%s",
                idx + 1,
                len(narrative_moments),
                segment_index,
                moment.get("type", "unknown"),
            )

            # Validate segment index is within bounds
            if segment_index < 0 or segment_index >= len(segments):
                logger.debug(
                    "REJECTED: Invalid segment_index %d (valid range: 0-%d)",
                    segment_index,
                    len(segments) - 1,
                )
                continue

            # Get the referenced segment
            segment = segments[segment_index]
            seg_start = (
                segment.get("start_ms", segment.get("start", 0)) / 1000.0
                if segment.get("start_ms")
                else segment.get("start", 0)
            )
            seg_end = (
                segment.get("end_ms", segment.get("end", 0)) / 1000.0
                if segment.get("end_ms")
                else segment.get("end", 0)
            )
            segment_text = segment.get("text", "")

            # Create validated moment with actual segment data
            validated_moments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "type": moment.get("type"),
                    "importance": moment.get("importance", 0.0),
                    "text": segment_text,  # Use actual segment text, not LLM-generated quote
                    "reason": moment.get("reason", ""),
                    "time": seg_start,  # Use segment start as moment time
                }
            )
            logger.debug(
                "VALIDATED: Segment %d [%.2fs-%.2fs] text='%s...'",
                segment_index,
                seg_start,
                seg_end,
                segment_text[:30] if segment_text else "NO_TEXT",
            )

        rejected_count = len(narrative_moments) - len(validated_moments)
        if rejected_count > 0:
            logger.warning(
                "Rejected %d hallucinated or unmatched narrative moments", rejected_count
            )

        logger.info("Identified %d validated narrative moments via LLM", len(validated_moments))

        return validated_moments

    except Exception as e:
        logger.error("Failed to analyze narrative context: %s", e, exc_info=True)
        return []


async def analyze_speech(
    audio_path: Path | str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Complete Stream A analysis: comprehensive speech analysis.

    Analyzes speech from multiple dimensions:
    - Prosodic: tone detection (emotion classification)
    - Semantic: narrative importance via LLM

    Args:
        audio_path: Path to speech audio file
        segments: Transcript segments with timestamps (used for both tone detection and narrative analysis)

    Returns:
        List of toned segments with all features:
        [
            {
                "start": float,
                "end": float,
                "text": str,
                "tone": str,                    # "excited", "dramatic", "calm", "neutral"
                "emotion_dimensions": {...},   # arousal, valence, dominance
                "features": {...},              # prosodic features (pitch, energy, etc.)
                "narrative_context": [...],     # LLM-identified important moments
                "importance": float,            # Calculated importance score [0, 1]
            },
            ...
        ]
    """
    analyzer = get_speech_semantic_analyzer()

    # 1. Detect tone on each segment
    logger.debug("Detecting tone on segments...")
    toned_segments = analyzer.detect_tone_on_segments(
        audio_path=audio_path,
        speech_segments=segments,  # Same segments used for tone detection
    )

    # 2. Analyze narrative context with LLM
    logger.debug("Analyzing narrative context with LLM...")
    narrative_moments = await analyze_narrative_context(
        segments=segments  # Same segments used for narrative analysis
    )

    # 3. Process segments: append narrative context and calculate importance
    # DEBUG: Log matching info
    logger.debug(
        "Matching %d narrative moments to %d toned segments",
        len(narrative_moments),
        len(toned_segments),
    )
    if narrative_moments:
        logger.debug(
            "Sample narrative moment: start=%.2f, end=%.2f",
            narrative_moments[0].get("start", 0),
            narrative_moments[0].get("end", 0),
        )
    if toned_segments:
        logger.debug(
            "Sample toned segment: start=%.2f, end=%.2f",
            toned_segments[0].get("start", 0),
            toned_segments[0].get("end", 0),
        )

    matched_moments_count = 0
    for segment in toned_segments:
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", 0.0)

        # Find and append matching narrative moments (same start/end format as segments)
        segment["narrative_context"] = []
        for moment in narrative_moments:
            # Narrative moments have start/end from their source segment - direct match
            moment_start = moment.get("start", 0.0)
            moment_end = moment.get("end", 0.0)
            if moment_start == segment_start and moment_end == segment_end:
                segment["narrative_context"].append(
                    {
                        "type": moment.get("type"),
                        "importance": moment.get("importance", 0.0),
                        "text": moment.get("text", ""),
                        "reason": moment.get("reason", ""),
                        "time": moment.get(
                            "time", segment_start
                        ),  # Specific timestamp within segment
                    }
                )
                matched_moments_count += 1
                logger.debug(
                    "MATCHED: Narrative moment [%.2f-%.2f] to toned segment [%.2f-%.2f]",
                    moment_start,
                    moment_end,
                    segment_start,
                    segment_end,
                )

        # Calculate importance score for this segment
        tone = segment.get("tone", "neutral")
        tone_scores = {
            "excited": TONE_IMPORTANCE_EXCITED,
            "dramatic": TONE_IMPORTANCE_DRAMATIC,
            "calm": TONE_IMPORTANCE_CALM,
            "neutral": TONE_IMPORTANCE_NEUTRAL,
        }
        base_importance = tone_scores.get(tone, TONE_IMPORTANCE_NEUTRAL)

        # Boost from emotion dimensions
        emotion_dims = segment.get("emotion_dimensions", {})
        arousal = emotion_dims.get("arousal", 0.5)
        valence = emotion_dims.get("valence", 0.5)
        dominance = emotion_dims.get("dominance", 0.5)

        # Emotion boost: weighted combination of dimensions
        emotion_boost = (
            arousal * EMOTION_AROUSAL_WEIGHT
            + valence * EMOTION_VALENCE_WEIGHT
            + dominance * EMOTION_DOMINANCE_WEIGHT
        ) * EMOTION_BOOST_WEIGHT

        # Boost from narrative context
        narrative_boost = 0.0
        narrative_context = segment.get("narrative_context", [])
        if narrative_context:
            max_narrative_importance = max(
                [ctx.get("importance", 0.0) for ctx in narrative_context]
            )
            narrative_boost = max_narrative_importance * NARRATIVE_BOOST_WEIGHT

        # Calculate final importance using weighted combination (clamped to [0, 1])
        importance = min(
            base_importance * IMPORTANCE_BASE_WEIGHT
            + emotion_boost * IMPORTANCE_EMOTION_WEIGHT
            + narrative_boost * IMPORTANCE_NARRATIVE_WEIGHT,
            1.0,
        )
        segment["importance"] = float(importance)

    # Log matching statistics
    logger.debug(
        "Matched %d narrative moments to toned segments (unmatched: %d)",
        matched_moments_count,
        len(narrative_moments) - matched_moments_count,
    )

    logger.info(
        "Stream A analysis complete: %d toned segments, %d narrative moments",
        len(toned_segments),
        len(narrative_moments),
    )

    return toned_segments
