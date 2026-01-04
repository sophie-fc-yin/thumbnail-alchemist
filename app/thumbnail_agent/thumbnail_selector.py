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

import cv2
import numpy as np
from google.genai import Client
from google.genai.types import Part
from PIL import Image

from app.thumbnail_agent.contextual_scoring import (
    ContextualScoringCriteria,
    get_contextual_scores,
)
from app.thumbnail_agent.scoring_weights import determine_optimal_weights


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

    # --------------------------------------------------------------------
    # Phase-1 language guardrail (diagnostic, non-prescriptive)
    # --------------------------------------------------------------------
    _BANNED_PHASE1_WORDS = (
        "should",
        "avoid",
        "best",
        "optimal",
        "recommended",
        "strategy",
        "tactic",
    )

    @classmethod
    def _sanitize_phase1_text(cls, value: Any) -> Any:
        """
        Remove banned Phase-1 words from creator-facing strings.

        This is a conservative guardrail: it prefers slightly awkward phrasing
        over drifting into prescriptive/teaching language.
        """
        if isinstance(value, str):
            lowered = value
            for w in cls._BANNED_PHASE1_WORDS:
                lowered = lowered.replace(w, "")
                lowered = lowered.replace(w.title(), "")
                lowered = lowered.replace(w.upper(), "")
            # Collapse whitespace
            lowered = " ".join(lowered.split())
            return lowered
        if isinstance(value, list):
            return [cls._sanitize_phase1_text(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._sanitize_phase1_text(v) for k, v in value.items()}
        return value

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
                Each frame MUST have:
                - url: GCS URL (gs://...) - REQUIRED for Gemini
                - timestamp: Time in video (seconds)
                - frame_number: Frame index
                - visual_analysis: Face/emotion data dict
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

        # Deduplicate near-identical frames by timestamp bucket (keep highest-score exemplar)
        scored_frames = self._dedupe_frames_by_timestamp(scored_frames, bucket_seconds=0.1)
        # Additional visual dedupe using aHash to avoid surfacing visually identical frames
        scored_frames = self._dedupe_by_perceptual_hash(scored_frames, max_distance=5)

        print("\nTop 5 scores (deduped):")
        for i, frame in enumerate(scored_frames[:5], 1):
            print(f"  {i}. Frame {frame['frame_number']}: {frame['total_score']:.3f}")

        # ====================================================================
        # Filter to top 10 frames for Gemini analysis, with temporal diversity
        # ====================================================================
        TOP_N_FOR_GEMINI = 10
        # Use balanced temporal diversity to ensure coverage across entire video
        diversified = self._diversify_by_time_balanced(
            scored_frames, target_count=TOP_N_FOR_GEMINI, min_gap_seconds=2.0
        )
        top_frames = diversified[:TOP_N_FOR_GEMINI]

        print(f"\nFiltered to top {len(top_frames)} frames for Gemini analysis")
        if top_frames:
            timestamps = [f["timestamp"] for f in top_frames]
            print(f"Timestamp range: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
            print(
                f"Timestamps: {', '.join(f'{t:.1f}s' for t in sorted(timestamps)[:5])}{'...' if len(timestamps) > 5 else ''}"
            )

            # AUDIT: Print detailed frame metadata for debugging
            print("\n[AUDIT] Frames being sent to Gemini:")
            for i, frame in enumerate(top_frames[:5], 1):
                print(f"  {i}. Frame #{frame.get('frame_number')} @ {frame.get('timestamp'):.2f}s")
                print(f"     Path: {frame.get('local_path', 'MISSING')[-50:]}")  # Last 50 chars
                print(f"     Score: {frame.get('total_score', 0):.3f}")

        # ====================================================================
        # PHASE 2: Gemini 2.5 Flash Final Decision ($0.0023, ~2s)
        # ====================================================================
        print(f"\n{'─'*70}")
        print("PHASE 2: Gemini 2.5 Flash Creative Decision")
        print(f"{'─'*70}")

        result = await self._phase2_gemini_decision(top_frames, creative_brief, channel_profile)

        # Add metadata to the result
        result["all_frame_scores"] = scored_frames  # Keep all scores (deduped) for reference
        result["gemini_model"] = self.model_name
        result["frames_sent_to_gemini"] = top_frames  # Frames sent to Gemini for analysis

        # Estimate cost based on frames actually sent to Gemini
        result["cost_usd"] = self._estimate_cost(len(top_frames))

        # Sanitize creator-facing fields for Phase-1 copy constraints
        # Keep debug intact (developer-facing).
        if isinstance(result, dict):
            debug = result.get("debug")
            result["moments"] = self._sanitize_phase1_text(result.get("moments", []))
            result["meta"] = self._sanitize_phase1_text(result.get("meta", {}))
            if debug is not None:
                result["debug"] = debug

        # Extract top moment for logging (if present)
        moments = result.get("moments", []) if isinstance(result, dict) else []
        first_moment = moments[0] if moments else {}
        meta = result.get("meta", {}) if isinstance(result, dict) else {}

        print(f"\n{'='*70}")
        print("THUMBNAIL SELECTION COMPLETE")
        print(f"{'='*70}")
        print(f"✨ Top moment: {first_moment.get('frame_id', 'N/A')}")
        print(f"🧭 Selection note: {meta.get('selection_note', 'n/a')}")
        print(f"💰 Cost: ${result['cost_usd']:.4f}")
        print(f"⚡ Model: {self.model_name}")

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
        # Get optimal weights from agent (falls back to niche preset if fails)
        weights = await determine_optimal_weights(brief, profile, model_name=self.model_name)

        print(f"Scoring weights for {profile.get('niche', 'default')}:")
        print(f"  Aesthetic: {weights['aesthetic_quality']:.0%}")
        print(f"  Psychology: {weights['psychology_score']:.0%}")
        print(f"  Editability: {weights['editability']:.0%}")
        print(f"  Face: {weights['face_quality']:.0%}")
        print(f"  Creator Alignment: {weights['creator_alignment']:.0%}")
        print(f"  Moment Importance: {weights['moment_importance']:.0%}")

        scored_frames = []

        print(
            f"\n[DEBUG] First frame visual_analysis keys: {list(frames[0].get('visual_analysis', {}).keys()) if frames else 'NO FRAMES'}"
        )
        print(
            f"[DEBUG] First frame visual_analysis sample: {frames[0].get('visual_analysis', {}) if frames else 'NO FRAMES'}"
        )

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

            # Get other component scores (all computed from real data, no placeholders)
            face_quality = self._compute_face_quality(visual_analysis)
            composition = self._compute_composition_score(visual_analysis)
            technical = self._compute_technical_quality(visual_analysis)

            # Creator alignment score
            creator_alignment = self._compute_creator_alignment(
                frame, visual_analysis, brief, profile
            )

            # Moment importance score from adaptive sampling (audio+visual saliency)
            # This is the core ClickMoment signal: “is this moment psychologically ready already?”
            moment_importance = float(frame.get("moment_score") or 0.0)
            if moment_importance > 1.0:
                # Defensive normalization if upstream returns non-[0,1]
                moment_importance = min(moment_importance / 10.0, 1.0)

            # Weighted total score (originality removed - no comparison data)
            total_score = (
                weights.get("moment_importance", 0.0) * moment_importance
                + weights["creator_alignment"] * creator_alignment
                + weights["aesthetic_quality"] * contextual_scores["aesthetic_score"]
                + weights["psychology_score"] * contextual_scores["psychology_score"]
                + weights["editability"] * editability_score
                + weights["face_quality"] * face_quality
                + weights["composition"] * composition
                + weights["technical_quality"] * technical
            )

            # Debug: Print component scores for first few frames to diagnose identical scores
            if len(scored_frames) < 5:
                image_qual = frame_features.get("image_quality", {})
                print(f"\nFrame {frame.get('frame_number')} @ {frame.get('timestamp'):.1f}s:")
                print(
                    f"  Subject brightness: {image_qual.get('subject_brightness', 0):.3f} {'❌ TOO DARK' if image_qual.get('is_too_dark') else '✓'}"
                )
                print(f"  Creator alignment: {creator_alignment:.3f}")
                print(f"  Aesthetic: {contextual_scores['aesthetic_score']:.3f}")
                print(
                    f"  Psychology: {contextual_scores['psychology_score']:.3f} (triggers: {len(detected_triggers)})"
                )
                print(f"  Editability: {editability_score:.3f}")
                print(f"  Face quality: {face_quality:.3f}")
                print(f"  Composition: {composition:.3f}")
                print(f"  Technical: {technical:.3f}")
                print(f"  → Total: {total_score:.3f}")

            scored_frames.append(
                {
                    "frame_number": frame.get("frame_number", 0),
                    "timestamp": frame.get("timestamp", 0.0),
                    "local_path": frame.get("local_path"),  # Local path for Gemini
                    "url": frame.get("url"),  # GCS URL for debugging
                    "total_score": total_score,
                    "moment_importance": moment_importance,
                    "creator_alignment": creator_alignment,
                    "aesthetic_score": contextual_scores["aesthetic_score"],
                    "psychology_score": contextual_scores["psychology_score"],
                    "editability": editability_score,
                    "editability_details": editability_eval,
                    "face_quality": face_quality,
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
        Phase 2: Gemini 2.5 Flash analyzes top-scored frames for final decision.

        Args:
            scored_frames: Top N frames from Phase 1 scoring (typically top 10)
            brief: Creative brief with video context
            profile: Channel profile with niche and style

        Returns:
            Selection result with chosen frame and reasoning

        Cost: ~$0.0023 (for 10 frames)
        Time: ~2s
        """
        # Build comprehensive prompt with all frame scores
        prompt = self._build_gemini_prompt(scored_frames, brief, profile)

        # Load images from local disk and build image parts
        image_parts = []

        print(f"\n[AUDIT] Loading {len(scored_frames)} frames for Gemini...")

        for i, frame in enumerate(scored_frames, 1):
            try:
                # Frames MUST have 'local_path' field
                if "local_path" not in frame:
                    print(f"[ERROR] Frame {i} available keys: {list(frame.keys())}")
                    raise ValueError(f"Frame {i} missing required 'local_path' field")

                local_path = frame["local_path"]
                frame_num = frame.get("frame_number", "?")
                timestamp = frame.get("timestamp", 0.0)

                # AUDIT: Log every frame being loaded
                print(
                    f"[AUDIT] Loading Frame {i}: frame_number={frame_num}, timestamp={timestamp:.2f}s, path=...{local_path[-40:]}"
                )

                # Verify file exists
                if not Path(local_path).exists():
                    raise ValueError(f"Frame {i} local file not found: {local_path}")

                # Read image data from local disk
                with open(local_path, "rb") as f:
                    image_data = f.read()

                # Create Part from inline image data
                image_part = Part.from_bytes(data=image_data, mime_type="image/jpeg")
                image_parts.append(image_part)
                print(f"[AUDIT] ✓ Loaded successfully ({len(image_data)} bytes)")

            except Exception as e:
                print(f"⚠️  Frame {i}: Error loading image: {e}")
                # Continue without this frame

        if not image_parts:
            raise ValueError("No valid frames could be loaded for Gemini analysis")

        print(f"Sending {len(image_parts)} frames to Gemini 2.5 Flash (from local disk)...")

        # Generate response using new API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, *image_parts],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,  # Low temperature for factual, deterministic output
            },
        )

        # Parse JSON response
        print(f"[DEBUG] Gemini response text: {response.text[:500]}...")
        result = json.loads(response.text)
        print(f"[DEBUG] Parsed JSON keys: {list(result.keys())}")

        # Verify response has expected Phase-1 diagnostic structure
        required_keys = ["moments", "meta", "debug"]
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"[ERROR] Gemini response missing keys: {missing_keys}. Full response: {result}")
            raise ValueError(f"Gemini response missing required fields: {missing_keys}")

        print(
            f"✅ Gemini returned Phase-1 insights with {len(result.get('debug', {}).get('all_frames_scored', []))} frames analyzed"
        )

        # Return the complete advisory response
        return result

    def _build_gemini_prompt(
        self,
        scored_frames: list[dict[str, Any]],
        brief: dict[str, Any],
        profile: dict[str, Any],
    ) -> str:
        """Build Phase-1 diagnostic prompt (observational) + debug scores."""

        # Build frame summaries with FULL scoring details for analysis
        frame_summaries = []
        for i, frame in enumerate(scored_frames, 1):
            edit_details = frame.get("editability_details", {})

            summary = f"""
**Frame {i}** (Timestamp: {frame['timestamp']:.1f}s)
SCORES (for internal analysis):
- Total: {frame['total_score']:.2f}
- Moment importance: {frame.get('moment_importance', 0.0):.2f}
- Aesthetic: {frame['aesthetic_score']:.2f}
- Psychology: {frame['psychology_score']:.2f}
- Editability: {frame['editability']:.2f} (emotion resilience {edit_details.get('emotion_resilience', 0):.2f}, text space {edit_details.get('text_overlay_space', 0):.2f})
- Face Quality: {frame['face_quality']:.2f}
- Creator Alignment: {frame['creator_alignment']:.2f}
FEATURES:
- Emotion: {frame['emotion']} (intensity: {frame['expression_intensity']:.2f})
- Triggers: {', '.join(frame['detected_triggers'][:4]) if frame['detected_triggers'] else 'none'}
- Visual quality: Good readability on mobile
"""
            frame_summaries.append(summary)

        prompt = f"""You are ClickMoment (Phase 1). Your job is diagnostic, not prescriptive.

Core positioning: You surface the moments in a video that are already psychologically ready to earn clicks.

You do NOT teach thumbnail design. You do NOT give instructions. You only describe what is already present.

You bring deep expertise in YouTube viewer attention psychology and thumbnail performance patterns, but you apply it only to **recognize** existing signals. Do not let this expertise become prescriptive; stay observational in the output.

## How to use scores (internal only)
- Use scores to decide which frames are strongest moments (scores guide; images decide)
- Prefer frames with high moment importance + clear emotional signal + fast readability

## Non‑negotiable prohibitions (creator-facing)
- Do NOT use: should, avoid, best, optimal, recommended, strategy, tactic
- Do NOT give instructions (no “add/move/resize/put text”)
- Do NOT write overlay copy
- Do NOT mention numeric scores, ranks, or performance predictions (CTR, views)
- Do NOT judge the creator; keep it observational and fast

## Your task
Return EXACTLY 3 moments that already feel clickable, each explained using these 4 pillars only:
1) Emotional signal detection
2) Curiosity gap validation (already present; not created)
3) Attention signal density (guardrail)
4) Readability & speed (mobile + ~2-second scan)

CRITICAL: You MUST return exactly 3 moments. If fewer than 3 strong candidates exist, include the best available options with honest assessment.

**Video Title**: "{brief.get('video_title', 'Untitled')}"
**Primary Message**: "{brief.get('primary_message', 'Not specified')}"
**Target Emotion**: {brief.get('target_emotion', 'Not specified')}
**Primary Goal**: {brief.get('primary_goal', 'maximize_ctr')}

**Channel Niche**: {profile.get('niche', 'general')} | {brief.get('tone', 'professional')} tone

**Context (lightweight, observational)**:
{self._get_niche_context(profile.get('niche', 'general'))}

## Frame Analysis

{''.join(frame_summaries)}

## Output Format (STRICT JSON)

Return EXACTLY this structure:

{{
  "moments": [
    {{
      "frame_id": "Frame X",
      "timestamp": "X.Xs",
      "moment_summary": "MAX ~18 WORDS - observational, no advice",
      "viewer_feel": "What the viewer likely feels at a glance (fast, emotional)",
      "why_this_reads": [
        "The emotion is immediately clear at a glance",
        "The moment hints at a story without explaining it",
        "The subject is readable on mobile"
      ],
      "pillars": {{
        "emotional_signal": "What emotion/tension is visible and how clearly it reads",
        "curiosity_gap": "How it hints at an outcome without resolving it (or 'not present')",
        "attention_signals": ["face", "motion blur", "high contrast"],
        "readability_speed": "Does it survive small size and a ~2-second scan? Why?"
      }},
      "optional_note": "Optional. Only if capture is weak: recreating the same moment may help."
    }}
  ],
  "meta": {{
    "selection_note": "Brief, non-quantitative statement of why these were surfaced (no confidence labels).",
    "positioning": "ClickMoment finds moments that already deserve to be thumbnails.",
    "note": "Short supportive note that the creator is choosing a moment, not being taught design."
  }},
  "debug": {{
    "all_frames_scored": [
      {{
        "frame_id": "Frame 1",
        "timestamp": "X.Xs",
        "total_score": 0.XX,
        "moment_importance": 0.XX,
        "aesthetic": 0.XX,
        "psychology": 0.XX,
        "editability": 0.XX,
        "face_quality": 0.XX,
        "creator_alignment": 0.XX,
        "emotion": "emotion_name",
        "expression_intensity": 0.XX,
        "triggers": ["list", "of", "triggers"],
        "why_chosen_or_not": "Brief technical note"
      }}
    ],
    "scoring_notes": "Brief explanation of how scores influenced your choices"
  }}
}}

## Rules
1. **moment_summary**: MAX ~18 words
2. **Tone**: Observational intelligence. Cause → effect. Viewer perception.
3. **Language**: Plain English, no jargon, no "best practices".
4. **No prescriptive advice**: describe only what is already in the frame.
5. **Optional note**: allowed only when the moment is strong but capture is weak; do not invent.
6. **Be factual and literal**: Describe only what you actually see in the images. Avoid speculation, interpretation, or creative descriptions.

CRITICAL: Base analysis on VISUAL assessment of actual images, not just score data. Stick to observable facts, not creative interpretations."""

        return prompt

    def _get_niche_context(self, niche: str) -> str:
        """Provide light niche context (observational, non-prescriptive)."""

        # Map niche variations to standard categories
        niche_lower = niche.lower()

        niche_contexts = {
            "gaming": """
- Fast, high-energy moments and clear reactions often read instantly on mobile
- The “what just happened?” feeling matters more than visual polish""",
            "tech": """
- Frames that show the subject + a clear “result” or reaction tend to read quickly""",
            "beauty": """
- Moments with obvious transformation/emotion tend to be instantly legible""",
            "commentary": """
- A clear, readable expression and a strong “take” can carry the moment alone""",
            "cooking": """
- The moment reads when the food/result is instantly recognizable""",
            "educational": """
- A clear “question → answer is coming” feeling can be visible in the moment itself""",
        }

        # Find matching niche context
        for key in niche_contexts:
            if key in niche_lower:
                return niche_contexts[key]

        # Default/general guidance
        return """
- Favor moments that read instantly (emotion + focal point) and still leave a question unanswered."""

    def _analyze_image_quality(self, image_path: str) -> dict[str, Any]:
        """
        Analyze actual image quality metrics from pixel data.

        CRITICAL: Analyzes center region (where main subject usually is).
        A bright subject with dark background is GOOD for thumbnails.

        Returns:
            {
                "brightness": float [0, 1],  # Overall frame brightness
                "subject_brightness": float [0, 1],  # Center region (main subject) brightness
                "contrast": float [0, 1],    # 0 = flat, 1 = high contrast
                "is_too_dark": bool,          # True if SUBJECT is too dark
                "is_too_bright": bool,        # True if subject is overexposed
                "mean_brightness": float      # Raw mean brightness [0, 255]
            }
        """
        try:
            # Check if file exists before attempting to read
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "brightness": 0.5,
                    "subject_brightness": 0.5,
                    "contrast": 0.5,
                    "is_too_dark": False,
                    "is_too_bright": False,
                    "mean_brightness": 128,
                }

            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    "brightness": 0.5,
                    "subject_brightness": 0.5,
                    "contrast": 0.5,
                    "is_too_dark": False,
                    "is_too_bright": False,
                    "mean_brightness": 128,
                }

            # Convert to grayscale for luminance analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Calculate overall frame brightness
            mean_brightness = float(np.mean(gray))
            brightness_normalized = mean_brightness / 255.0

            # CRITICAL: Analyze center region (40% of frame) where main subject typically is
            # This works for faces, products, animals, or any centered subject
            center_y_start = int(h * 0.3)
            center_y_end = int(h * 0.7)
            center_x_start = int(w * 0.3)
            center_x_end = int(w * 0.7)

            center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
            subject_brightness_raw = float(np.mean(center_region))
            subject_brightness = subject_brightness_raw / 255.0

            # Calculate contrast (standard deviation of pixel values)
            std_dev = float(np.std(gray))
            contrast_normalized = min(std_dev / 80.0, 1.0)

            # Detect problematic exposure based on SUBJECT brightness
            # Dark subject: < 60/255 = 0.235 (main subject too dark to see clearly)
            is_too_dark = subject_brightness < 0.235

            # Overexposed subject: > 230/255 = 0.90 (washed out)
            is_too_bright = subject_brightness > 0.90

            return {
                "brightness": brightness_normalized,
                "subject_brightness": subject_brightness,
                "contrast": contrast_normalized,
                "is_too_dark": is_too_dark,
                "is_too_bright": is_too_bright,
                "mean_brightness": mean_brightness,
            }
        except Exception as e:
            print(f"[WARNING] Failed to analyze image quality for {image_path}: {e}")
            return {
                "brightness": 0.5,
                "subject_brightness": 0.5,
                "contrast": 0.5,
                "is_too_dark": False,
                "is_too_bright": False,
                "mean_brightness": 128,
            }

    def _extract_frame_features(
        self, frame: dict[str, Any], visual_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract features from frame for contextual scoring."""
        # Get actual image quality metrics
        image_path = frame.get("local_path")
        image_quality = self._analyze_image_quality(image_path) if image_path else {}

        # Extract emotion data
        emotion = visual_analysis.get("dominant_emotion", "neutral")
        expression_intensity = visual_analysis.get("expression_intensity", 0.5)

        # Build features based on actual image analysis
        features = {
            "lighting": [],
            "color_palette": ["natural"],
            "composition": ["centered_subject"] if expression_intensity > 0.6 else ["standard"],
        }

        # Use actual brightness data to determine lighting
        brightness = image_quality.get("brightness", 0.5)
        contrast = image_quality.get("contrast", 0.5)

        if image_quality.get("is_too_dark"):
            features["lighting"].extend(["dark", "dim", "underexposed"])
        elif image_quality.get("is_too_bright"):
            features["lighting"].extend(["overexposed", "washed_out"])
        elif brightness > 0.6:
            features["lighting"].extend(["bright", "well_lit"])
        elif brightness > 0.4:
            features["lighting"].extend(["even", "balanced"])
        else:
            features["lighting"].extend(["dim", "low_light"])

        # Contrast-based lighting
        if contrast > 0.6:
            features["lighting"].append("high_contrast")
            features["color_palette"].append("high_contrast")
        elif contrast < 0.3:
            features["lighting"].append("flat")

        # Adjust based on emotion (secondary to actual image data)
        if emotion in ["joy", "excitement"]:
            features["color_palette"].append("warm_tones")
        elif emotion in ["surprise", "shock"]:
            features["lighting"].append("dramatic")
        elif emotion in ["serious", "focused"]:
            features["polish_level"] = ["professional"]

        # Store image quality for later use
        features["image_quality"] = image_quality

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
        score = 0.0  # Start from 0, no base score

        # Target emotion match (strong signal)
        target_emotion = brief.get("target_emotion", "").lower()
        detected_emotion = visual_analysis.get("dominant_emotion", "").lower()

        if target_emotion and target_emotion in detected_emotion:
            score += 0.4
        elif visual_analysis.get("has_face", False):
            score += 0.2  # Has face but not target emotion

        # Tone alignment
        tone = brief.get("tone", "").lower()
        expression_intensity = visual_analysis.get("expression_intensity", 0.0)

        if tone == "energetic" and expression_intensity > 0.7:
            score += 0.3
        elif tone == "calm" and expression_intensity < 0.4:
            score += 0.3
        elif tone == "professional" and 0.4 <= expression_intensity <= 0.7:
            score += 0.3
        else:
            # Partial match
            score += 0.1

        # Expression clarity bonus
        if expression_intensity > 0.5:
            score += 0.3

        return min(score, 1.0)

    def _compute_composition_score(self, visual_analysis: dict[str, Any]) -> float:
        """Compute composition quality from head pose and face presence."""
        if not visual_analysis.get("has_face", False):
            return 0.3  # Low score if no face detected

        score = 0.7  # Base score for having a face

        # Check head pose - penalize extreme angles
        head_pose = visual_analysis.get("head_pose", {})
        pitch = abs(head_pose.get("pitch", 0.0))
        yaw = abs(head_pose.get("yaw", 0.0))
        roll = abs(head_pose.get("roll", 0.0))

        # Penalize if head is too tilted (good thumbnails have centered faces)
        if pitch < 15 and yaw < 20 and roll < 15:
            score += 0.2  # Well-centered face
        elif pitch > 30 or yaw > 40 or roll > 30:
            score -= 0.3  # Badly angled face

        return max(min(score, 1.0), 0.0)

    def _compute_technical_quality(self, visual_analysis: dict[str, Any]) -> float:
        """Compute technical quality from face detection and feature clarity."""
        if not visual_analysis.get("has_face", False):
            return 0.2  # Very low if no face

        score = 0.6  # Base score for face detected

        # Eye openness (avoid mid-blink)
        eye_openness = visual_analysis.get("eye_openness", 0.5)
        if eye_openness > 0.7:
            score += 0.2  # Eyes wide open - clear shot
        elif eye_openness < 0.3:
            score -= 0.2  # Eyes closed/mid-blink

        # Mouth position (avoid mid-speech awkwardness)
        mouth_openness = visual_analysis.get("mouth_openness", 0.0)
        if mouth_openness < 0.3:
            score += 0.2  # Mouth closed or neutral - clean look
        elif mouth_openness > 0.6:
            score -= 0.1  # Wide open mouth - might be mid-speech

        return max(min(score, 1.0), 0.0)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _dedupe_frames_by_timestamp(
        self, frames: list[dict[str, Any]], bucket_seconds: float = 0.1
    ) -> list[dict[str, Any]]:
        """
        Deduplicate frames that land in the same timestamp bucket.

        Keeps the highest total_score per bucket to avoid duplicate-looking
        entries in Phase-1 results and debug.
        """
        buckets: dict[float, dict[str, Any]] = {}

        for frame in frames:
            ts = float(frame.get("timestamp", 0.0))
            bucket = round(ts / bucket_seconds) * bucket_seconds
            current = buckets.get(bucket)
            if current is None or frame.get("total_score", 0.0) > current.get("total_score", 0.0):
                buckets[bucket] = frame

        # Return sorted by total_score descending (consistent with upstream sort expectations)
        deduped = sorted(buckets.values(), key=lambda x: x.get("total_score", 0.0), reverse=True)
        return deduped

    def _compute_ahash(self, image_path: str, hash_size: int = 8) -> int | None:
        """Compute a simple average hash for perceptual similarity."""
        if not image_path or not Path(image_path).exists():
            return None
        try:
            with Image.open(image_path) as img:
                img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
                pixels = list(img.getdata())
                mean = sum(pixels) / len(pixels)
                bits = "".join("1" if p > mean else "0" for p in pixels)
                return int(bits, 2)
        except Exception:
            return None

    def _hamming(self, a: int, b: int) -> int:
        """Hamming distance between two integer hashes."""
        return (a ^ b).bit_count()

    def _dedupe_by_perceptual_hash(
        self, frames: list[dict[str, Any]], max_distance: int = 5
    ) -> list[dict[str, Any]]:
        """
        Deduplicate visually similar frames using average hash.

        Assumes frames are pre-sorted by total_score desc; keeps first in each cluster.
        """
        if not frames:
            return frames

        kept: list[dict[str, Any]] = []
        hashes: list[int] = []

        for frame in frames:
            h = self._compute_ahash(frame.get("local_path"))
            if h is None:
                kept.append(frame)
                continue

            is_dup = False
            for existing_hash in hashes:
                if self._hamming(h, existing_hash) <= max_distance:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(frame)
                hashes.append(h)

        return kept

    def _diversify_by_time(
        self, frames: list[dict[str, Any]], min_gap_seconds: float, max_frames: int
    ) -> list[dict[str, Any]]:
        """
        Enforce temporal diversity: walk sorted frames and keep those spaced apart in time.
        """
        kept: list[dict[str, Any]] = []
        last_ts = None
        for frame in frames:
            ts = float(frame.get("timestamp", 0.0))
            if last_ts is None or abs(ts - last_ts) >= min_gap_seconds:
                kept.append(frame)
                last_ts = ts
            if len(kept) >= max_frames:
                break
        return kept

    def _diversify_by_time_balanced(
        self, frames: list[dict[str, Any]], target_count: int, min_gap_seconds: float = 2.0
    ) -> list[dict[str, Any]]:
        """
        Select frames with balanced temporal coverage across the video.

        This ensures we don't miss good moments from the later parts of the video.
        Uses a hybrid approach:
        1. Divide video into equal time segments
        2. Take best frame(s) from each segment
        3. Fill remaining slots with highest-scored frames (respecting min_gap)

        Args:
            frames: All frames sorted by score (descending)
            target_count: Number of frames to select
            min_gap_seconds: Minimum gap between selected frames

        Returns:
            Diversified frame selection with temporal balance
        """
        if not frames:
            return []

        if len(frames) <= target_count:
            return frames

        # Get time range
        timestamps = [f.get("timestamp", 0.0) for f in frames]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        duration = max_ts - min_ts

        if duration < min_gap_seconds:
            # Video too short for meaningful diversity
            return frames[:target_count]

        # STRATEGY: If frames are already pre-selected from importance segments (close to target),
        # just apply min_gap filter instead of creating uniform segments
        # This avoids empty segments when frames are clustered in important regions
        if (
            len(frames) <= target_count * 1.6
        ):  # Within 60% of target (e.g., 16 frames for target=10)
            print(
                f"\n[TEMPORAL DIVERSITY] Using pre-selected importance frames ({len(frames)} frames):"
            )
            print(f"  Applying min_gap filter ({min_gap_seconds}s) to select top {target_count}...")

            selected = []
            for frame in frames:  # Already sorted by score (descending)
                ts = frame.get("timestamp", 0.0)

                # Check minimum gap from all selected frames
                too_close = any(
                    abs(ts - sel_frame.get("timestamp", 0.0)) < min_gap_seconds
                    for sel_frame in selected
                )

                if not too_close:
                    selected.append(frame)
                    print(f"  ✓ Frame @ {ts:.1f}s (score: {frame.get('total_score', 0):.3f})")

                if len(selected) >= target_count:
                    break

            print(f"[TEMPORAL DIVERSITY] Selected {len(selected)} frames from importance segments")
            return selected[:target_count]

        # FALLBACK: If many frames (not pre-selected), use uniform segmentation
        # This guarantees one frame from each portion of the video
        num_segments = target_count
        segment_duration = duration / num_segments

        selected = []

        print(
            f"\n[TEMPORAL DIVERSITY] Dividing {duration:.1f}s video into {num_segments} segments:"
        )

        # Select EXACTLY ONE best frame from each segment
        for seg_idx in range(num_segments):
            seg_start = min_ts + (seg_idx * segment_duration)
            seg_end = seg_start + segment_duration

            # Find frames in this segment
            segment_frames = [f for f in frames if seg_start <= f.get("timestamp", 0.0) < seg_end]

            if segment_frames:
                # Take best scored frame from this segment
                best_in_segment = segment_frames[0]  # Already sorted by score
                selected.append(best_in_segment)
                print(
                    f"  Segment {seg_idx+1} ({seg_start:.1f}s-{seg_end:.1f}s): frame @ {best_in_segment.get('timestamp'):.1f}s (score: {best_in_segment.get('total_score', 0):.3f})"
                )
            else:
                print(f"  Segment {seg_idx+1} ({seg_start:.1f}s-{seg_end:.1f}s): NO FRAMES")

        # If we didn't get enough frames (some segments were empty),
        # fill remaining slots from other segments while respecting min_gap
        if len(selected) < target_count:
            print(
                f"\n[TEMPORAL DIVERSITY] Only found {len(selected)}/{target_count} frames, filling remaining..."
            )

            # Get frames not yet selected, sorted by score
            remaining = [f for f in frames if f not in selected]

            for frame in remaining:
                ts = frame.get("timestamp", 0.0)

                # Check minimum gap from all selected frames
                too_close = any(
                    abs(ts - sel_frame.get("timestamp", 0.0)) < min_gap_seconds
                    for sel_frame in selected
                )

                if not too_close:
                    selected.append(frame)
                    print(f"  Added frame @ {ts:.1f}s")

                if len(selected) >= target_count:
                    break

        # Sort by score for final ranking
        selected.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)

        print(
            f"[TEMPORAL DIVERSITY] Final: {len(selected)} frames, range: {min([f.get('timestamp', 0) for f in selected]):.1f}s - {max([f.get('timestamp', 0) for f in selected]):.1f}s"
        )

        return selected[:target_count]

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
