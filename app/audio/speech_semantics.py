"""Stream A: Speech Semantics Analysis

Analyzes the semantic and narrative importance of speech to identify
moments that matter for thumbnail selection.

Key Functions:
- Tone detection (emotion classification on speech)
- Narrative context (important sentences via LLM)
- Speaking style changes (calm → excited transitions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np


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
        """Initialize semantic analyzer."""
        pass

    def detect_tone_on_segments(
        self,
        audio_path: Path | str,
        speech_segments: list[dict[str, float]],
        prosody_features: dict[str, Any],
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
            prosody_features: Output from analyze_audio_features()

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
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        for segment in speech_segments:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            segment_audio = y[start_sample:end_sample]

            # Skip very short segments
            if len(segment_audio) < sr * 0.3:  # Less than 300ms
                continue

            # Extract prosodic features for this segment
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr, fmin=80, fmax=400)

            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) < 5:
                continue

            # Calculate features
            pitch_mean = float(np.mean(pitch_values))
            pitch_variance = float(np.var(pitch_values))
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values))

            # Energy
            rms = librosa.feature.rms(y=segment_audio)[0]
            energy_mean = float(np.mean(rms))
            energy_variance = float(np.var(rms))

            # Speaking rate (approximate using zero crossing rate)
            zcr = librosa.feature.zero_crossing_rate(y=segment_audio)[0]
            speaking_rate = float(np.mean(zcr)) * 10  # Rough approximation

            # Classify tone based on features
            tone, confidence = self._classify_tone(
                pitch_mean=pitch_mean,
                pitch_variance=pitch_variance,
                pitch_range=pitch_range,
                energy_mean=energy_mean,
                energy_variance=energy_variance,
                speaking_rate=speaking_rate,
            )

            toned_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "tone": tone,
                    "confidence": confidence,
                    "features": {
                        "pitch_mean": pitch_mean,
                        "pitch_variance": pitch_variance,
                        "pitch_range": pitch_range,
                        "energy_mean": energy_mean,
                        "energy_variance": energy_variance,
                        "speaking_rate": speaking_rate,
                    },
                }
            )

        return toned_segments

    def _classify_tone(
        self,
        pitch_mean: float,
        pitch_variance: float,
        pitch_range: float,
        energy_mean: float,
        energy_variance: float,
        speaking_rate: float,
    ) -> tuple[str, float]:
        """
        Classify speaking tone from prosodic features.

        Rule-based classification (can be replaced with ML later):
        - Excited: high pitch, high variance, high energy, fast rate
        - Calm: low pitch, low variance, low energy, slow rate
        - Dramatic: high pitch range, high energy variance
        - Neutral: middle values

        Returns:
            (tone, confidence)
        """
        # Normalize features (rough normalization)
        pitch_norm = min(pitch_mean / 200.0, 2.0)  # 100Hz = low, 300Hz = high
        variance_norm = min(pitch_variance / 50.0, 2.0)
        energy_norm = min(energy_mean / 0.2, 2.0)
        rate_norm = min(speaking_rate / 2.0, 2.0)

        # Calculate tone scores
        excited_score = (
            0.3 * pitch_norm + 0.25 * variance_norm + 0.25 * energy_norm + 0.2 * rate_norm
        )

        calm_score = (
            0.3 * (2.0 - pitch_norm)
            + 0.25 * (2.0 - variance_norm)
            + 0.25 * (2.0 - energy_norm)
            + 0.2 * (2.0 - rate_norm)
        )

        dramatic_score = 0.5 * min(pitch_range / 100.0, 2.0) + 0.5 * min(
            energy_variance / 0.05, 2.0
        )

        # Select highest score
        scores = {
            "excited": excited_score,
            "calm": calm_score,
            "dramatic": dramatic_score,
            "neutral": 1.0,  # Baseline
        }

        tone = max(scores, key=scores.get)
        confidence = min(scores[tone] / 2.0, 1.0)  # Normalize to [0, 1]

        return tone, confidence

    def detect_style_changes(
        self, toned_segments: list[dict[str, Any]], threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """
        Detect speaking style transitions (calm → excited, etc.).

        These transitions often indicate important moments:
        - Calm → Excited: something interesting is happening
        - Normal → Dramatic: tension or reveal coming
        - Slow → Fast: urgency or excitement

        Args:
            toned_segments: Segments with tone classification
            threshold: Minimum change to consider a transition

        Returns:
            List of style change moments:
            [
                {
                    "time": 5.2,
                    "from_tone": "calm",
                    "to_tone": "excited",
                    "importance": 0.85,
                    "reason": "tone_shift"
                },
                ...
            ]
        """
        style_changes = []

        for i in range(1, len(toned_segments)):
            prev_segment = toned_segments[i - 1]
            curr_segment = toned_segments[i]

            # Check if tone changed
            if prev_segment["tone"] != curr_segment["tone"]:
                # Calculate feature deltas
                prev_features = prev_segment["features"]
                curr_features = curr_segment["features"]

                pitch_delta = abs(curr_features["pitch_mean"] - prev_features["pitch_mean"])
                energy_delta = abs(curr_features["energy_mean"] - prev_features["energy_mean"])
                rate_delta = abs(curr_features["speaking_rate"] - prev_features["speaking_rate"])

                # Normalize deltas
                pitch_delta_norm = min(pitch_delta / 50.0, 1.0)
                energy_delta_norm = min(energy_delta / 0.1, 1.0)
                rate_delta_norm = min(rate_delta / 1.0, 1.0)

                # Calculate importance
                importance = (
                    0.4 * pitch_delta_norm + 0.3 * energy_delta_norm + 0.3 * rate_delta_norm
                )

                if importance > threshold:
                    style_changes.append(
                        {
                            "time": curr_segment["start"],
                            "from_tone": prev_segment["tone"],
                            "to_tone": curr_segment["tone"],
                            "importance": float(importance),
                            "reason": "tone_shift",
                            "metadata": {
                                "pitch_delta": float(pitch_delta),
                                "energy_delta": float(energy_delta),
                                "rate_delta": float(rate_delta),
                            },
                        }
                    )

        return style_changes


async def analyze_narrative_context(
    transcript: str, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
        List of important narrative moments:
        [
            {
                "time": 5.2,
                "type": "hook_statement",
                "importance": 0.9,
                "text": "I can't believe this actually worked",
                "reason": "introduces conflict/surprise"
            },
            ...
        ]
    """
    import os

    from openai import OpenAI

    # Prepare transcript with timestamps
    timestamped_transcript = ""
    for segment in segments:
        start_time = segment.get("start_ms", segment.get("start", 0)) / 1000.0
        text = segment.get("text", "")
        timestamped_transcript += f"[{start_time:.1f}s] {text}\n"

    # Create LLM prompt
    prompt = f"""You are analyzing a video transcript to identify important narrative moments. You MUST follow these rules STRICTLY:

1. ONLY use EXACT quotes from the transcript below - do NOT paraphrase, summarize, or create new text
2. The "text" field MUST be a word-for-word copy from the transcript
3. If you cannot find compelling moments, return an empty array: []
4. DO NOT make up generic examples or placeholder text
5. Every quote you return will be validated against the original transcript

Transcript:
{timestamped_transcript}

Task: Identify moments that are genuinely compelling for a thumbnail:
- Hook statements (surprising, intriguing, attention-grabbing)
- Emotional peaks (excitement, tension, revelation)
- Story beats (key turning points)
- Important reveals or information

For each moment provide:
- time: Timestamp in seconds (float)
- type: One of: "hook_statement", "emotional_peak", "story_beat", "reveal"
- importance: Score 0.0-1.0 based on how compelling this moment is
- text: EXACT QUOTE from the transcript (word-for-word, no changes)
- reason: Brief explanation of why this specific moment matters

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
[
  {{"time": 5.2, "type": "hook_statement", "importance": 0.9, "text": "exact quote from transcript", "reason": "why it matters"}},
  {{"time": 15.8, "type": "emotional_peak", "importance": 0.85, "text": "exact quote from transcript", "reason": "why it matters"}}
]

If no compelling moments exist, return: []"""

    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for this task
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing video transcripts to identify compelling moments. You MUST use exact quotes from the transcript - never paraphrase or create text. All quotes will be validated. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Very low temperature to reduce hallucination
        )

        # Parse response
        import json

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(line for line in lines if not line.startswith("```"))

        narrative_moments = json.loads(result_text)

        # VALIDATION: Verify all quotes actually exist in the transcript
        validated_moments = []
        full_transcript_text = " ".join([seg.get("text", "") for seg in segments])

        for moment in narrative_moments:
            quote = moment.get("text", "").strip()

            # Check if quote exists in transcript (allow some whitespace flexibility)
            if quote and quote in full_transcript_text:
                validated_moments.append(moment)
            else:
                # Try to find it with normalized whitespace
                normalized_quote = " ".join(quote.split())
                normalized_transcript = " ".join(full_transcript_text.split())

                if normalized_quote in normalized_transcript:
                    validated_moments.append(moment)
                else:
                    print(f"[Stream A] WARNING: Rejected hallucinated quote: '{quote[:50]}...'")

        rejected_count = len(narrative_moments) - len(validated_moments)
        if rejected_count > 0:
            print(f"[Stream A] Rejected {rejected_count} hallucinated quotes")

        print(f"[Stream A] Identified {len(validated_moments)} validated narrative moments via LLM")

        return validated_moments

    except Exception as e:
        print(f"[Stream A] Failed to analyze narrative context: {e}")
        return []


async def analyze_speech_semantics(
    audio_path: Path | str,
    transcript: str,
    transcript_segments: list[dict[str, Any]],
    speech_segments: list[dict[str, float]],
    prosody_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Complete Stream A analysis: semantic understanding of speech.

    Args:
        audio_path: Path to speech audio file
        transcript: Full transcript text
        transcript_segments: Transcript segments with timestamps
        speech_segments: Speech segments from VAD
        prosody_features: Audio features from analyze_audio_features()

    Returns:
        Stream A analysis results:
        {
            "toned_segments": [...],     # Segments with tone classification
            "style_changes": [...],       # Speaking style transitions
            "narrative_moments": [...],   # LLM-identified important moments
            "importance_timeline": [...], # Time-ordered moments with scores
        }
    """
    analyzer = SpeechSemanticAnalyzer()

    # 1. Detect tone on each speech segment
    print("[Stream A] Detecting tone on speech segments...")
    toned_segments = analyzer.detect_tone_on_segments(
        audio_path=audio_path,
        speech_segments=speech_segments,
        prosody_features=prosody_features,
    )

    # 2. Detect speaking style changes
    print("[Stream A] Detecting style changes...")
    style_changes = analyzer.detect_style_changes(toned_segments)

    # 3. Analyze narrative context with LLM
    print("[Stream A] Analyzing narrative context with LLM...")
    narrative_moments = await analyze_narrative_context(
        transcript=transcript, segments=transcript_segments
    )

    # 4. Create unified importance timeline
    importance_timeline = []

    # Add style changes
    for change in style_changes:
        importance_timeline.append(
            {
                "time": change["time"],
                "type": "style_change",
                "score": change["importance"],  # Use "score" for unified format
                "source": "speech",  # Unified source name
                "metadata": change,
            }
        )

    # Add narrative moments
    for moment in narrative_moments:
        importance_timeline.append(
            {
                "time": moment["time"],
                "type": moment["type"],
                "score": moment["importance"],  # Use "score" for unified format
                "source": "speech",  # Unified source name
                "metadata": moment,
            }
        )

    # Sort by time
    importance_timeline.sort(key=lambda x: x["time"])

    print(
        f"[Stream A] Analysis complete: "
        f"{len(toned_segments)} toned segments, "
        f"{len(style_changes)} style changes, "
        f"{len(narrative_moments)} narrative moments"
    )

    return {
        "toned_segments": toned_segments,
        "style_changes": style_changes,
        "narrative_moments": narrative_moments,
        "importance_timeline": importance_timeline,
    }
