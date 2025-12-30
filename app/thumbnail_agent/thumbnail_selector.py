"""Thumbnail Selection Agent - Main orchestrator.

This agent selects the best thumbnail frame from adaptive sampling results using:
- Phase 1: Contextual scoring (niche-aware aesthetics + goal-aligned psychology)
- Phase 2: Gemini 2.5 Flash for creative final decision

Cost: ~$0.0023 per selection
Speed: ~2-3 seconds
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google.genai import Client
from PIL import Image

from app.thumbnail_agent.contextual_scoring import (
    ContextualScoringCriteria,
    get_contextual_scores,
)
from app.thumbnail_agent.scoring_weights import get_scoring_weights


class ThumbnailSelector:
    """
    Main thumbnail selection agent.

    Uses hybrid approach:
    1. Contextual scoring for all frames (niche + goal aware)
    2. Gemini 2.5 Flash for final creative decision

    Example:
        >>> selector = ThumbnailSelector()
        >>> result = await selector.select_best_thumbnail(
        ...     frames=extracted_frames,
        ...     creative_brief=brief,
        ...     channel_profile=profile
        ... )
        >>> print(result["selected_frame_path"])
        >>> print(result["reasoning"])
    """

    def __init__(self, use_pro: bool = False):
        """
        Initialize thumbnail selector.

        Args:
            use_pro: If True, use Gemini 2.5 Pro ($0.007).
                     If False, use Gemini 2.5 Flash ($0.0023).
        """
        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your API key from https://ai.google.dev/"
            )

        # Create Gemini client
        self.client = Client(api_key=api_key)

        # Select model
        self.model_name = "gemini-2.5-pro" if use_pro else "gemini-2.5-flash"

        print(f"[ThumbnailSelector] Initialized with {self.model_name}")

    async def select_best_thumbnail(
        self,
        frames: list[dict[str, Any]],
        creative_brief: dict[str, Any],
        channel_profile: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Select the best thumbnail frame from candidates.

        Args:
            frames: List of frame metadata from adaptive sampling
                Each frame should have:
                - path: Local path or GCS URL
                - timestamp: Time in video
                - frame_number: Frame index
                - visual_analysis: Face/emotion data
                - moment_score: Combined audio+visual score

            creative_brief: Creator's requirements
                - video_title: Title of the video
                - primary_message: Main point of video
                - target_emotion: Desired viewer emotion
                - primary_goal: maximize_ctr | grow_subscribers | brand_building
                - tone: professional | casual | energetic | calm
                - must_include: Required elements
                - avoid_moments: Moments to avoid

            channel_profile: Channel metadata
                - niche: Channel category (beauty, tech, gaming, etc.)
                - personality: Brand personality traits
                - visual_style: Preferred aesthetic
                - top_performing_thumbnails: Examples of successful thumbnails

        Returns:
            {
                "selected_frame_path": str,
                "selected_frame_url": str,
                "selected_frame_timestamp": float,
                "selected_frame_number": int,
                "selected_frame": int,
                "confidence": float (0-1),
                "reasoning": {
                    "summary": str,
                    "visual_analysis": str,
                    "score_alignment": str,
                    "niche_fit": str,
                    "goal_optimization": str,
                    "psychology_triggers": str
                },
                "key_strengths": list[str],
                "comparative_analysis": {
                    "runner_up": str,
                    "score_vs_visual": str,
                    "weaknesses_avoided": str
                },
                "creator_message": str,
                "quantitative_scores": dict,
                "all_frame_scores": list[dict],
                "gemini_model": str,
                "cost_usd": float
            }
        """
        print(f"\n{'='*70}")
        print("THUMBNAIL SELECTION AGENT - Starting")
        print(f"{'='*70}")
        print(f"Frames to analyze: {len(frames)}")
        print(f"Channel niche: {channel_profile.get('niche')}")
        print(f"Primary goal: {creative_brief.get('primary_goal')}")
        print(f"Model: {self.model_name}")

        # ====================================================================
        # PHASE 1: Contextual Scoring (FREE, ~500ms)
        # ====================================================================
        print(f"\n{'─'*70}")
        print("PHASE 1: Contextual Scoring (Niche + Goal Aware)")
        print(f"{'─'*70}")

        scored_frames = await self._phase1_contextual_scoring(
            frames, creative_brief, channel_profile
        )

        # Sort by score
        scored_frames.sort(key=lambda x: x["total_score"], reverse=True)

        print("\nTop 5 scores:")
        for i, frame in enumerate(scored_frames[:5], 1):
            print(f"  {i}. Frame {frame['frame_number']}: {frame['total_score']:.3f}")

        # ====================================================================
        # PHASE 2: Gemini 2.5 Flash Final Decision ($0.0023, ~2s)
        # ====================================================================
        print(f"\n{'─'*70}")
        print("PHASE 2: Gemini 2.5 Flash Creative Decision")
        print(f"{'─'*70}")

        result = await self._phase2_gemini_decision(scored_frames, creative_brief, channel_profile)

        # Add quantitative scores
        selected_idx = result["selected_frame"] - 1
        result["quantitative_scores"] = scored_frames[selected_idx]
        result["all_frame_scores"] = scored_frames
        result["gemini_model"] = self.model_name

        # Estimate cost
        result["cost_usd"] = self._estimate_cost(len(frames))

        print(f"\n{'='*70}")
        print("THUMBNAIL SELECTION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Selected: Frame {result['selected_frame']}")
        print(f"📊 Confidence: {result['confidence']:.0%}")
        print(f"💰 Cost: ${result['cost_usd']:.4f}")
        print(f"⚡ Model: {self.model_name}")
        print(f"\n💡 {result['creator_message']}")

        return result

    async def _phase1_contextual_scoring(
        self,
        frames: list[dict[str, Any]],
        brief: dict[str, Any],
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Phase 1: Compute contextual scores for all frames.

        Uses niche-specific aesthetic criteria and goal-aligned psychology.

        Returns:
            List of frames with computed scores
        """
        # Get niche-specific weights
        weights = get_scoring_weights(profile)

        print(f"Using {profile.get('niche', 'default')} niche weights:")
        print(f"  Aesthetic: {weights['aesthetic_quality']:.0%}")
        print(f"  Psychology: {weights['psychology_score']:.0%}")
        print(f"  Editability: {weights['editability']:.0%}")
        print(f"  Face: {weights['face_quality']:.0%}")

        scored_frames = []

        for frame in frames:
            # Extract features from frame metadata
            visual_analysis = frame.get("visual_analysis", {})

            # Build frame features for contextual scoring
            frame_features = self._extract_frame_features(frame, visual_analysis)

            # Detect psychology triggers
            detected_triggers = self._detect_psychology_triggers(frame, visual_analysis, brief)

            # Get contextual scores (niche + goal aware)
            contextual_scores = get_contextual_scores(
                frame_features=frame_features,
                detected_triggers=detected_triggers,
                channel_profile=profile,
                creative_brief=brief,
            )

            # Get editability score (NEW - workability for creators)
            editability_eval = ContextualScoringCriteria.evaluate_editability(
                frame_features=frame_features,
                visual_analysis=visual_analysis,
                niche=profile.get("niche", "general"),
            )
            editability_score = editability_eval["overall_editability"]

            # Get other component scores
            face_quality = self._compute_face_quality(visual_analysis)
            originality = 0.75  # Placeholder - would compare to existing thumbnails
            composition = frame_features.get("composition_score", 0.70)
            technical = frame_features.get("technical_quality", 0.80)

            # Creator alignment score
            creator_alignment = self._compute_creator_alignment(
                frame, visual_analysis, brief, profile
            )

            # Weighted total score
            total_score = (
                weights["creator_alignment"] * creator_alignment
                + weights["aesthetic_quality"] * contextual_scores["aesthetic_score"]
                + weights["psychology_score"] * contextual_scores["psychology_score"]
                + weights["editability"] * editability_score
                + weights["face_quality"] * face_quality
                + weights["originality"] * originality
                + weights["composition"] * composition
                + weights["technical_quality"] * technical
            )

            scored_frames.append(
                {
                    "frame_number": frame.get("frame_number", 0),
                    "timestamp": frame.get("timestamp", 0.0),
                    "path": frame.get("path"),
                    "url": frame.get("url"),
                    "total_score": total_score,
                    "creator_alignment": creator_alignment,
                    "aesthetic_score": contextual_scores["aesthetic_score"],
                    "psychology_score": contextual_scores["psychology_score"],
                    "editability": editability_score,
                    "editability_details": editability_eval,
                    "face_quality": face_quality,
                    "originality": originality,
                    "composition": composition,
                    "technical": technical,
                    "detected_triggers": detected_triggers,
                    "emotion": visual_analysis.get("dominant_emotion", "unknown"),
                    "expression_intensity": visual_analysis.get("expression_intensity", 0.0),
                    "aesthetic_details": contextual_scores.get("aesthetic_details", {}),
                    "psychology_details": contextual_scores.get("psychology_details", {}),
                }
            )

        return scored_frames

    async def _phase2_gemini_decision(
        self,
        scored_frames: list[dict[str, Any]],
        brief: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Phase 2: Gemini 2.5 Flash analyzes all frames + scores for final decision.

        Cost: ~$0.0023
        Time: ~2s
        """
        # Build comprehensive prompt with all frame scores
        prompt = self._build_gemini_prompt(scored_frames, brief, profile)

        # Load images
        images = []
        for i, frame in enumerate(scored_frames, 1):
            try:
                # Try local path first
                path = frame.get("path")
                if path and Path(path).exists():
                    img = Image.open(path)
                    images.append(img)
                else:
                    print(f"⚠️  Frame {i}: Could not load from {path}")
                    # TODO: Handle GCS URLs - download temporarily
                    images.append(None)
            except Exception as e:
                print(f"⚠️  Frame {i}: Error loading image: {e}")
                images.append(None)

        # Filter out None images
        valid_images = [img for img in images if img is not None]

        if not valid_images:
            raise ValueError("No valid images could be loaded for Gemini analysis")

        print(f"Sending {len(valid_images)} frames to Gemini 2.5 Flash...")

        # Generate response using new API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, *valid_images],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.4,  # Lower for more consistent decisions
            },
        )

        # Parse JSON response
        result = json.loads(response.text)

        # Add frame metadata
        selected_idx = result["selected_frame"] - 1
        selected_frame = scored_frames[selected_idx]

        result["selected_frame_path"] = selected_frame["path"]
        result["selected_frame_url"] = selected_frame.get("url")
        result["selected_frame_timestamp"] = selected_frame["timestamp"]
        result["selected_frame_number"] = selected_frame["frame_number"]

        return result

    def _build_gemini_prompt(
        self,
        scored_frames: list[dict[str, Any]],
        brief: dict[str, Any],
        profile: dict[str, Any],
    ) -> str:
        """Build comprehensive prompt for Gemini with all frame scores."""

        # Build frame summaries
        frame_summaries = []
        for i, frame in enumerate(scored_frames, 1):
            # Extract editability details
            edit_details = frame.get("editability_details", {})
            text_space = edit_details.get("text_overlay_space", 0.0)
            emotion_resilience = edit_details.get("emotion_resilience", 0.0)

            summary = f"""
**Frame {i}** (Timestamp: {frame['timestamp']:.1f}s, Overall Score: {frame['total_score']:.2f})
- Aesthetic (niche-adjusted): {frame['aesthetic_score']:.2f}
- Psychology (goal-aligned): {frame['psychology_score']:.2f}
- Editability (workability): {frame['editability']:.2f}
  → Text overlay space: {text_space:.2f}
  → Emotion resilience: {emotion_resilience:.2f}
- Creator Alignment: {frame['creator_alignment']:.2f}
- Face Quality: {frame['face_quality']:.2f}
- Emotion: {frame['emotion']} (intensity: {frame['expression_intensity']:.2f})
- Detected Triggers: {', '.join(frame['detected_triggers'][:4]) if frame['detected_triggers'] else 'none'}
"""
            frame_summaries.append(summary)

        prompt = f"""You are a YouTube thumbnail optimization expert with deep knowledge of viewer psychology and creator branding.

I've analyzed {len(scored_frames)} frames using specialized models with niche-specific criteria. Each frame has quantitative scores that are already optimized for this channel's niche and goals. Now I need your creative judgment to select the BEST frame.

## Creator Context

**Video Title**: "{brief.get('video_title', 'Untitled')}"
**Primary Message**: "{brief.get('primary_message', 'Not specified')}"
**Target Emotion**: {brief.get('target_emotion', 'Not specified')}
**Primary Goal**: {brief.get('primary_goal', 'maximize_ctr')} (maximize CTR, grow subscribers, or brand building)

**Channel Profile**:
- Niche: {profile.get('niche', 'general')}
- Personality: {profile.get('personality', 'Not specified')}
- Tone: {brief.get('tone', 'professional')}
- Visual Style: {profile.get('visual_style', 'Not specified')}

**Important Context**:
- Aesthetic scores are adjusted for {profile.get('niche', 'general')} niche standards
- Psychology scores prioritize {brief.get('primary_goal', 'CTR')} triggers
- All scores use niche-specific weights

## Frame Analysis

{''.join(frame_summaries)}

## Your Task

Analyze ALL {len(scored_frames)} frames visually alongside their quantitative scores. Select the BEST frame for this specific video.

**Consider**:
1. **Title/Thumbnail Synergy**: Does it complement "{brief.get('video_title')}"?
2. **Niche Fit**: Does it match {profile.get('niche')} aesthetic standards?
3. **Goal Alignment**: Will it maximize {brief.get('primary_goal')}?
4. **Editability (CRITICAL)**: Can this frame survive cropping, zooming, and text overlays while maintaining emotional impact? Is there space for title text? Will the emotion remain clear after edits?
5. **Visual + Data**: Do your visual observations confirm or contradict the scores?
6. **Psychological Impact**: What emotions/triggers will drive clicks?
7. **Brand Consistency**: Does it fit the creator's personality?
8. **Uniqueness**: Will it stand out in search results and recommended feeds?

**Important**: The quantitative scores are already niche-optimized, but trust your visual judgment if you see something the models missed.

Return your analysis in this exact JSON format:
{{
  "selected_frame": <number 1 to {len(scored_frames)}>,
  "confidence": <0.0 to 1.0>,
  "reasoning": {{
    "summary": "1-2 sentences explaining why this frame is THE best choice overall",
    "visual_analysis": "Detailed explanation of what you see in the image that makes it work (facial expression, composition, lighting, etc.)",
    "editability_assessment": "CRITICAL: How well can this frame survive cropping, zooming, text overlays? Where can creators add title text? Will emotion remain clear after edits?",
    "score_alignment": "How your visual assessment aligns with or diverges from the quantitative scores",
    "niche_fit": "Why this frame specifically works for the {profile.get('niche', 'general')} niche",
    "goal_optimization": "How this frame will help achieve the {brief.get('primary_goal', 'goal')} goal",
    "psychology_triggers": "Which psychological triggers this frame activates and why they'll drive clicks"
  }},
  "key_strengths": ["strength1", "strength2", "strength3", "strength4"],
  "comparative_analysis": {{
    "runner_up": "Which frame was second choice and why it fell short",
    "score_vs_visual": "If the highest-scoring frame wasn't selected, explain why visual judgment overrode the score",
    "weaknesses_avoided": "What weaknesses in other frames this selection avoids"
  }},
  "creator_message": "2-3 sentences explaining to the creator why this frame will help achieve their {brief.get('primary_goal', 'goal')}, what makes it stand out, and any suggestions for text overlay placement"
}}"""

        return prompt

    def _extract_frame_features(
        self, frame: dict[str, Any], visual_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract features from frame for contextual scoring."""
        # This is a simplified version - in production would have actual detectors

        # Mock features based on available data
        emotion = visual_analysis.get("dominant_emotion", "neutral")
        expression_intensity = visual_analysis.get("expression_intensity", 0.5)

        # Infer lighting/style from emotion and intensity
        # (In production, would use actual computer vision models)
        features = {
            "lighting": ["natural"],
            "color_palette": ["natural"],
            "composition": ["centered_subject"] if expression_intensity > 0.6 else ["standard"],
            "composition_score": 0.70,
            "technical_quality": 0.80,
        }

        # Adjust based on emotion
        if emotion in ["joy", "excitement"]:
            features["lighting"].append("bright")
            features["color_palette"].append("warm_tones")
        elif emotion in ["surprise", "shock"]:
            features["lighting"].append("dramatic")
            features["color_palette"].append("high_contrast")
        elif emotion in ["serious", "focused"]:
            features["lighting"].append("even")
            features["polish_level"] = ["professional"]

        return features

    def _detect_psychology_triggers(
        self,
        frame: dict[str, Any],
        visual_analysis: dict[str, Any],
        brief: dict[str, Any],
    ) -> list[str]:
        """Detect psychology triggers in frame."""
        triggers = []

        emotion = visual_analysis.get("dominant_emotion", "neutral")
        expression_intensity = visual_analysis.get("expression_intensity", 0.0)

        # Emotional contagion (strong emotions)
        if expression_intensity > 0.7:
            triggers.append("emotional_contagion")

        # Specific emotion triggers
        if emotion in ["surprise", "shock"]:
            triggers.append("surprise")
            triggers.append("curiosity_gap")
        elif emotion in ["joy", "excitement"]:
            triggers.append("aspiration")
            triggers.append("emotional_contagion")
        elif emotion == "serious":
            triggers.append("authority")
        elif emotion == "fear":
            triggers.append("pattern_interrupt")

        # Context-based triggers
        target_emotion = brief.get("target_emotion", "").lower()
        if target_emotion and target_emotion in emotion:
            triggers.append("authenticity")

        return list(set(triggers))  # Remove duplicates

    def _compute_face_quality(self, visual_analysis: dict[str, Any]) -> float:
        """Compute face quality score."""
        if not visual_analysis:
            return 0.5

        # Use expression intensity as proxy for face quality
        expression_intensity = visual_analysis.get("expression_intensity", 0.5)

        # Bonus for clear emotions
        emotion = visual_analysis.get("dominant_emotion", "neutral")
        emotion_bonus = 0.1 if emotion != "neutral" else 0.0

        return min(expression_intensity + emotion_bonus, 1.0)

    def _compute_creator_alignment(
        self,
        frame: dict[str, Any],
        visual_analysis: dict[str, Any],
        brief: dict[str, Any],
        profile: dict[str, Any],
    ) -> float:
        """Compute how well frame aligns with creator's brief."""
        score = 0.7  # Base score

        # Check target emotion match
        target_emotion = brief.get("target_emotion", "").lower()
        detected_emotion = visual_analysis.get("dominant_emotion", "").lower()

        if target_emotion and target_emotion in detected_emotion:
            score += 0.15

        # Check expression intensity for energetic vs calm
        tone = brief.get("tone", "").lower()
        expression_intensity = visual_analysis.get("expression_intensity", 0.5)

        if tone == "energetic" and expression_intensity > 0.7:
            score += 0.1
        elif tone == "calm" and expression_intensity < 0.4:
            score += 0.1
        elif tone == "professional" and 0.4 <= expression_intensity <= 0.7:
            score += 0.1

        return min(score, 1.0)

    def _estimate_cost(self, num_frames: int) -> float:
        """
        Estimate API cost for selection.

        Gemini 2.5 Flash:
        - Input: $0.30 per 1M tokens
        - Output: $2.50 per 1M tokens
        - Image: ~258 tokens each
        - Prompt: ~1000 tokens
        """
        if self.model_name == "gemini-2.5-pro":
            # Gemini 2.5 Pro pricing
            input_cost_per_m = 1.25
            output_cost_per_m = 10.00
        else:
            # Gemini 2.5 Flash pricing
            input_cost_per_m = 0.30
            output_cost_per_m = 2.50

        # Calculate tokens
        image_tokens = num_frames * 258
        prompt_tokens = 1000
        total_input_tokens = image_tokens + prompt_tokens
        output_tokens = 500

        # Calculate cost
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m

        return input_cost + output_cost
