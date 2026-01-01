"""Context-aware scoring that adapts aesthetic and psychology evaluation to channel niche and creative brief.

This module ensures that:
1. Aesthetic quality is judged by niche-specific criteria (beauty vs gaming vs news)
2. Psychology triggers are prioritized based on channel goals and audience
3. Scoring aligns with creator's brand, tone, and target metrics
"""

from __future__ import annotations

from typing import Any


class ContextualScoringCriteria:
    """Define niche-specific aesthetic and psychology criteria."""

    # ========================================================================
    # AESTHETIC CRITERIA BY NICHE
    # ========================================================================
    # What "high aesthetic quality" means varies by niche
    AESTHETIC_CRITERIA = {
        "beauty_lifestyle": {
            "lighting": {
                "preferred": ["soft", "warm", "golden_hour", "ring_light"],
                "avoid": ["harsh", "overhead", "fluorescent"],
                "weight": 0.25,
            },
            "color_palette": {
                "preferred": ["pastel", "warm_tones", "complementary", "monochromatic"],
                "avoid": ["oversaturated", "clashing", "muddy"],
                "weight": 0.25,
            },
            "composition": {
                "preferred": ["centered_subject", "shallow_depth", "clean_background"],
                "avoid": ["cluttered", "off_center", "busy_background"],
                "weight": 0.20,
            },
            "polish_level": {
                "preferred": ["professional", "edited", "polished"],
                "avoid": ["raw", "unedited", "amateur"],
                "weight": 0.20,
            },
            "visual_style": {
                "preferred": ["instagram_worthy", "magazine_quality", "aspirational"],
                "avoid": ["casual", "documentary", "gritty"],
                "weight": 0.10,
            },
        },
        "tech_educational": {
            "lighting": {
                "preferred": ["bright", "even", "studio_lighting"],
                "avoid": ["dim", "uneven", "overly_artistic"],
                "weight": 0.15,
            },
            "color_palette": {
                "preferred": ["vibrant", "high_contrast", "screen_visible"],
                "avoid": ["washed_out", "low_contrast"],
                "weight": 0.20,
            },
            "composition": {
                "preferred": ["rule_of_thirds", "clear_subject", "tech_visible"],
                "avoid": ["obscured_product", "unclear_focus"],
                "weight": 0.25,
            },
            "clarity": {
                "preferred": ["sharp", "readable_text", "clear_details"],
                "avoid": ["blurry", "unreadable", "soft_focus"],
                "weight": 0.30,  # ⬆️ Clarity is critical for tech
            },
            "visual_style": {
                "preferred": ["modern", "clean", "minimalist"],
                "avoid": ["overly_busy", "dated", "cluttered"],
                "weight": 0.10,
            },
        },
        "gaming": {
            "lighting": {
                "preferred": ["dramatic", "RGB", "neon", "high_contrast"],
                "avoid": ["flat", "neutral", "dull"],
                "weight": 0.15,
            },
            "color_palette": {
                "preferred": ["saturated", "bold", "complementary", "neon"],
                "avoid": ["desaturated", "muted", "monotone"],
                "weight": 0.25,  # ⬆️ Bold colors critical
            },
            "composition": {
                "preferred": ["dynamic", "action_focused", "energetic"],
                "avoid": ["static", "boring", "passive"],
                "weight": 0.25,
            },
            "energy_level": {
                "preferred": ["high_energy", "intense", "exciting"],
                "avoid": ["calm", "subdued", "low_energy"],
                "weight": 0.25,  # ⬆️ Energy is key
            },
            "visual_style": {
                "preferred": ["cinematic", "epic", "over_the_top"],
                "avoid": ["realistic", "mundane", "understated"],
                "weight": 0.10,
            },
        },
        "commentary": {
            "lighting": {
                "preferred": ["natural", "soft", "authentic"],
                "avoid": ["overly_polished", "studio", "artificial"],
                "weight": 0.15,
            },
            "color_palette": {
                "preferred": ["natural", "warm", "relatable"],
                "avoid": ["overly_saturated", "artificial"],
                "weight": 0.15,
            },
            "composition": {
                "preferred": ["face_focused", "intimate", "direct_address"],
                "avoid": ["distant", "impersonal"],
                "weight": 0.25,
            },
            "authenticity": {
                "preferred": ["genuine", "raw", "unpolished_ok"],
                "avoid": ["overly_produced", "fake", "staged"],
                "weight": 0.30,  # ⬆️ Authenticity critical
            },
            "visual_style": {
                "preferred": ["conversational", "relatable", "personal"],
                "avoid": ["corporate", "overly_professional"],
                "weight": 0.15,
            },
        },
        "cooking_food": {
            "lighting": {
                "preferred": ["natural", "warm", "appetizing"],
                "avoid": ["cool_tones", "fluorescent", "dim"],
                "weight": 0.25,
            },
            "color_palette": {
                "preferred": ["appetizing", "warm", "rich_colors"],
                "avoid": ["unappetizing", "gray_tones", "dull"],
                "weight": 0.25,
            },
            "composition": {
                "preferred": ["plating_visible", "overhead", "close_up"],
                "avoid": ["food_obscured", "poor_angle"],
                "weight": 0.25,  # ⬆️ Plating matters
            },
            "food_appeal": {
                "preferred": ["mouth_watering", "fresh", "vibrant"],
                "avoid": ["unappetizing", "wilted", "dull"],
                "weight": 0.20,
            },
            "visual_style": {
                "preferred": ["rustic", "elegant", "homestyle"],
                "avoid": ["clinical", "institutional"],
                "weight": 0.05,
            },
        },
        "fitness_health": {
            "lighting": {
                "preferred": ["bright", "energetic", "motivational"],
                "avoid": ["dim", "gloomy", "low_energy"],
                "weight": 0.20,
            },
            "color_palette": {
                "preferred": ["vibrant", "energetic", "inspiring"],
                "avoid": ["dull", "washed_out"],
                "weight": 0.20,
            },
            "composition": {
                "preferred": ["body_visible", "form_clear", "transformation_evident"],
                "avoid": ["obscured", "unclear"],
                "weight": 0.25,
            },
            "aspirational_quality": {
                "preferred": ["inspiring", "achievable", "motivating"],
                "avoid": ["demotivating", "unrealistic", "intimidating"],
                "weight": 0.25,  # ⬆️ Aspiration critical
            },
            "visual_style": {
                "preferred": ["dynamic", "active", "energetic"],
                "avoid": ["static", "passive", "boring"],
                "weight": 0.10,
            },
        },
    }

    # ========================================================================
    # PSYCHOLOGY TRIGGERS BY NICHE + BRIEF GOALS
    # ========================================================================
    # Different niches respond to different psychological triggers
    PSYCHOLOGY_TRIGGERS = {
        "beauty_lifestyle": {
            "primary_triggers": [
                {
                    "trigger": "aspiration",
                    "description": "Desire to achieve similar look/lifestyle",
                    "detection": ["transformation", "before_after", "glam_shot"],
                    "weight": 0.30,
                },
                {
                    "trigger": "emotional_contagion",
                    "description": "Positive emotion spreads (joy, confidence)",
                    "detection": ["genuine_smile", "confident_expression", "joy"],
                    "weight": 0.25,
                },
                {
                    "trigger": "curiosity_gap",
                    "description": "How did they achieve that look?",
                    "detection": ["dramatic_result", "unexpected", "unique_style"],
                    "weight": 0.20,
                },
                {
                    "trigger": "social_proof",
                    "description": "Others are doing it/loving it",
                    "detection": ["popular_trend", "viral_moment", "crowd_approved"],
                    "weight": 0.15,
                },
                {
                    "trigger": "scarcity",
                    "description": "Limited time/exclusive offer",
                    "detection": ["urgency", "exclusive", "limited"],
                    "weight": 0.10,
                },
            ],
        },
        "tech_educational": {
            "primary_triggers": [
                {
                    "trigger": "curiosity_gap",
                    "description": "Knowledge gap that needs filling",
                    "detection": ["question_pose", "surprising_fact", "unknown_revealed"],
                    "weight": 0.35,  # ⬆️ Highest for educational
                },
                {
                    "trigger": "authority",
                    "description": "Expert credibility signals",
                    "detection": ["confident_posture", "tech_visible", "professional_setting"],
                    "weight": 0.25,
                },
                {
                    "trigger": "clarity",
                    "description": "Clear, understandable explanation promised",
                    "detection": ["simple_framing", "clear_subject", "organized"],
                    "weight": 0.20,
                },
                {
                    "trigger": "pattern_interrupt",
                    "description": "Unexpected result/comparison",
                    "detection": ["surprising_comparison", "myth_busted", "unexpected"],
                    "weight": 0.15,
                },
                {
                    "trigger": "utility",
                    "description": "Practical value/usefulness",
                    "detection": ["problem_solution", "how_to", "practical"],
                    "weight": 0.05,
                },
            ],
        },
        "gaming": {
            "primary_triggers": [
                {
                    "trigger": "excitement",
                    "description": "High energy, hype, epic moment",
                    "detection": ["intense_expression", "action_moment", "epic_scene"],
                    "weight": 0.30,
                },
                {
                    "trigger": "surprise",
                    "description": "Unexpected gameplay moment",
                    "detection": ["shocked_expression", "rare_event", "unexpected"],
                    "weight": 0.25,
                },
                {
                    "trigger": "triumph",
                    "description": "Victory, achievement, win",
                    "detection": ["celebration", "victory_pose", "success"],
                    "weight": 0.20,
                },
                {
                    "trigger": "curiosity",
                    "description": "What happened? How?",
                    "detection": ["mysterious", "unclear_outcome", "question"],
                    "weight": 0.15,
                },
                {
                    "trigger": "fomo",
                    "description": "Fear of missing out on content",
                    "detection": ["exclusive", "new_content", "limited_time"],
                    "weight": 0.10,
                },
            ],
        },
        "commentary": {
            "primary_triggers": [
                {
                    "trigger": "authenticity",
                    "description": "Genuine, real reaction",
                    "detection": ["genuine_expression", "raw_emotion", "unfiltered"],
                    "weight": 0.30,
                },
                {
                    "trigger": "relatability",
                    "description": "Shared experience/feeling",
                    "detection": ["empathetic_expression", "knowing_look", "shared_frustration"],
                    "weight": 0.25,
                },
                {
                    "trigger": "controversy",
                    "description": "Hot take, strong opinion",
                    "detection": ["strong_expression", "disagreement", "debate"],
                    "weight": 0.20,
                },
                {
                    "trigger": "curiosity",
                    "description": "What's the take?",
                    "detection": ["questioning_look", "surprised", "intrigued"],
                    "weight": 0.15,
                },
                {
                    "trigger": "emotional_contagion",
                    "description": "Emotion spreads (frustration, joy)",
                    "detection": ["strong_emotion", "passionate", "animated"],
                    "weight": 0.10,
                },
            ],
        },
    }

    # ========================================================================
    # BRIEF-GUIDED PSYCHOLOGY PRIORITIZATION
    # ========================================================================
    # Map creator goals to psychology trigger priorities
    GOAL_TO_TRIGGERS = {
        "maximize_ctr": [
            "curiosity_gap",  # Primary CTR driver
            "pattern_interrupt",
            "surprise",
            "fomo",
        ],
        "maximize_watch_time": [
            "curiosity_gap",
            "authority",  # Trust = longer watch
            "utility",  # Value = retention
            "clarity",
        ],
        "grow_subscribers": [
            "authority",  # Credibility builds loyalty
            "authenticity",
            "relatability",
            "social_proof",
        ],
        "brand_building": [
            "authenticity",
            "aspiration",
            "authority",
            "emotional_contagion",
        ],
        "engagement": [
            "controversy",  # Comments
            "curiosity_gap",
            "surprise",
            "relatability",
        ],
    }

    # ========================================================================
    # CONTENT TONE → AESTHETIC/PSYCHOLOGY ADJUSTMENTS
    # ========================================================================
    TONE_ADJUSTMENTS = {
        "professional": {
            "aesthetic_boost": ["polish_level", "clarity", "clean_background"],
            "psychology_boost": ["authority", "clarity", "utility"],
            "aesthetic_penalty": ["raw", "unpolished", "casual"],
            "psychology_penalty": ["controversy", "surprise"],
        },
        "casual": {
            "aesthetic_boost": ["natural", "authentic", "relatable"],
            "psychology_boost": ["authenticity", "relatability", "emotional_contagion"],
            "aesthetic_penalty": ["overly_polished", "corporate"],
            "psychology_penalty": ["authority", "formality"],
        },
        "energetic": {
            "aesthetic_boost": ["vibrant", "high_contrast", "dynamic"],
            "psychology_boost": ["excitement", "surprise", "fomo"],
            "aesthetic_penalty": ["subdued", "calm", "muted"],
            "psychology_penalty": ["contemplative", "serious"],
        },
        "calm": {
            "aesthetic_boost": ["soft", "warm", "clean"],
            "psychology_boost": ["clarity", "utility", "aspiration"],
            "aesthetic_penalty": ["harsh", "chaotic", "busy"],
            "psychology_penalty": ["excitement", "controversy"],
        },
    }

    @classmethod
    def get_aesthetic_criteria(
        cls, channel_profile: dict[str, Any], creative_brief: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get niche-specific aesthetic criteria adjusted for creator's brief.

        Args:
            channel_profile: Channel niche, style, consistency preferences
            creative_brief: Creator's tone, brand, visual preferences

        Returns:
            Aesthetic evaluation criteria with weighted preferences

        Example:
            >>> profile = {"niche": "tech reviews"}
            >>> brief = {"tone": "professional", "visual_style": "modern"}
            >>> criteria = get_aesthetic_criteria(profile, brief)
            >>> criteria["clarity"]["weight"]  # Higher for tech
            0.30
        """
        # Get base niche criteria
        niche = channel_profile.get("niche", "").lower()
        niche_key = cls._map_niche_to_criteria_key(niche)
        base_criteria = cls.AESTHETIC_CRITERIA.get(
            niche_key, cls.AESTHETIC_CRITERIA["tech_educational"]
        ).copy()

        # Adjust for creative brief tone
        tone = creative_brief.get("tone", "").lower()
        if tone in cls.TONE_ADJUSTMENTS:
            adjustments = cls.TONE_ADJUSTMENTS[tone]

            # Boost weights for preferred attributes
            for category in base_criteria.values():
                if isinstance(category, dict) and "preferred" in category:
                    for boost_attr in adjustments.get("aesthetic_boost", []):
                        if boost_attr in category["preferred"]:
                            category["weight"] = min(category["weight"] * 1.2, 1.0)

                    # Reduce weights for penalized attributes
                    for penalty_attr in adjustments.get("aesthetic_penalty", []):
                        if penalty_attr in category["avoid"]:
                            category["weight"] = max(category["weight"] * 0.8, 0.0)

        # Add custom visual preferences from brief
        visual_style = creative_brief.get("visual_style")
        if visual_style:
            # Boost matching style preferences
            for category in base_criteria.values():
                if isinstance(category, dict) and "preferred" in category:
                    if visual_style in category["preferred"]:
                        category["weight"] = min(category["weight"] * 1.15, 1.0)

        return base_criteria

    @classmethod
    def get_psychology_priorities(
        cls, channel_profile: dict[str, Any], creative_brief: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Get prioritized psychology triggers based on niche and creator goals.

        Args:
            channel_profile: Channel niche
            creative_brief: Creator's goals, target metrics

        Returns:
            Ordered list of psychology triggers with adjusted weights

        Example:
            >>> profile = {"niche": "tech reviews"}
            >>> brief = {"primary_goal": "maximize_ctr"}
            >>> triggers = get_psychology_priorities(profile, brief)
            >>> triggers[0]["trigger"]  # Top priority
            'curiosity_gap'
        """
        # Get base niche triggers
        niche = channel_profile.get("niche", "").lower()
        niche_key = cls._map_niche_to_criteria_key(niche)
        base_triggers = cls.PSYCHOLOGY_TRIGGERS.get(
            niche_key, cls.PSYCHOLOGY_TRIGGERS["tech_educational"]
        )["primary_triggers"].copy()

        # Adjust weights based on creator goals
        primary_goal = creative_brief.get("primary_goal", "maximize_ctr")
        goal_triggers = cls.GOAL_TO_TRIGGERS.get(primary_goal, [])

        for trigger in base_triggers:
            if trigger["trigger"] in goal_triggers:
                # Boost weight for goal-aligned triggers
                trigger["weight"] = min(trigger["weight"] * 1.3, 1.0)

        # Re-normalize weights
        total_weight = sum(t["weight"] for t in base_triggers)
        for trigger in base_triggers:
            trigger["weight"] = trigger["weight"] / total_weight

        # Sort by weight (highest first)
        base_triggers.sort(key=lambda x: x["weight"], reverse=True)

        return base_triggers

    @classmethod
    def evaluate_aesthetic_alignment(
        cls,
        frame_features: dict[str, Any],
        aesthetic_criteria: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate how well a frame's aesthetics align with channel criteria.

        Args:
            frame_features: Detected aesthetic features (lighting, colors, etc.)
            aesthetic_criteria: Niche + brief-specific criteria

        Returns:
            Alignment scores per category and overall score

        Example:
            >>> features = {"lighting": "soft", "colors": "warm_tones"}
            >>> criteria = get_aesthetic_criteria(profile, brief)
            >>> scores = evaluate_aesthetic_alignment(features, criteria)
            >>> scores["overall_score"]
            0.87  # High alignment
        """
        category_scores = {}

        for category_name, category_criteria in aesthetic_criteria.items():
            if not isinstance(category_criteria, dict):
                continue

            preferred = category_criteria.get("preferred", [])
            avoid = category_criteria.get("avoid", [])
            weight = category_criteria.get("weight", 0.0)

            # Check matches
            feature_value = frame_features.get(category_name, "")
            if isinstance(feature_value, str):
                feature_value = [feature_value]

            matches_preferred = sum(1 for pref in preferred if pref in feature_value)
            matches_avoid = sum(1 for avoid_item in avoid if avoid_item in feature_value)

            # Score: +1 per preferred match, -1 per avoid match
            raw_score = matches_preferred - matches_avoid
            normalized_score = max(
                0.0, min(1.0, (raw_score + len(avoid)) / (len(preferred) + len(avoid)))
            )

            category_scores[category_name] = {
                "score": normalized_score,
                "weight": weight,
                "matches_preferred": matches_preferred,
                "matches_avoid": matches_avoid,
            }

        # Overall weighted score
        overall_score = sum(cat["score"] * cat["weight"] for cat in category_scores.values())

        # CRITICAL: Apply brightness penalty for dark/overexposed frames
        # Dark frames are unusable for thumbnails regardless of other qualities
        image_quality = frame_features.get("image_quality", {})
        if image_quality.get("is_too_dark"):
            # Severe penalty for dark frames
            brightness_penalty = 0.4  # Reduce score by 40%
            overall_score *= 1.0 - brightness_penalty
        elif image_quality.get("is_too_bright"):
            # Moderate penalty for overexposed frames
            overall_score *= 0.8

        return {
            "overall_score": overall_score,
            "category_scores": category_scores,
            "brightness_penalty_applied": image_quality.get("is_too_dark")
            or image_quality.get("is_too_bright"),
        }

    @classmethod
    def evaluate_psychology_alignment(
        cls,
        detected_triggers: list[str],
        psychology_priorities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate how well detected triggers align with channel priorities.

        Args:
            detected_triggers: Psychology triggers detected in frame
            psychology_priorities: Prioritized triggers for this channel

        Returns:
            Alignment scores and detected priority triggers

        Example:
            >>> detected = ["curiosity_gap", "authority", "clarity"]
            >>> priorities = get_psychology_priorities(profile, brief)
            >>> scores = evaluate_psychology_alignment(detected, priorities)
            >>> scores["overall_score"]
            0.92  # High alignment with tech channel goals
        """
        # Map trigger names to weights
        priority_weights = {t["trigger"]: t["weight"] for t in psychology_priorities}

        # Score each detected trigger
        trigger_scores = {}
        for trigger in detected_triggers:
            weight = priority_weights.get(trigger, 0.0)
            trigger_scores[trigger] = {
                "detected": True,
                "priority_weight": weight,
                "aligned": weight > 0.1,  # Significant priority
            }

        # Overall score: sum of priority weights for detected triggers
        overall_score = sum(ts["priority_weight"] for ts in trigger_scores.values())

        # Bonus for detecting top 3 priority triggers
        top_3_priorities = [t["trigger"] for t in psychology_priorities[:3]]
        top_3_detected = sum(1 for t in detected_triggers if t in top_3_priorities)
        bonus = 0.1 * top_3_detected

        overall_score = min(overall_score + bonus, 1.0)

        return {
            "overall_score": overall_score,
            "detected_triggers": trigger_scores,
            "top_priorities_matched": top_3_detected,
        }

    @classmethod
    def evaluate_editability(
        cls,
        frame_features: dict[str, Any],
        visual_analysis: dict[str, Any],
        niche: str = "general",
    ) -> dict[str, Any]:
        """
        Evaluate frame's editability - can it survive cropping, zooming, text overlays?

        This answers the critical question: "Can this frame survive being cropped,
        zoomed, simplified, and still communicate emotion instantly?"

        Args:
            frame_features: Detected features (composition, margins, etc.)
            visual_analysis: Face analysis data (expression, landmarks, etc.)
            niche: Content niche (affects text overlay importance)

        Returns:
            {
                "overall_editability": float [0,1],
                "crop_resilience": float [0,1],
                "zoom_potential": float [0,1],
                "text_overlay_space": float [0,1],
                "emotion_resilience": float [0,1],
                "composition_flexibility": float [0,1]
            }
        """
        # ================================================================
        # 1. CROP RESILIENCE
        # Can we crop 30-40% and still have a good frame?
        # ================================================================
        crop_score = 0.7  # Base score

        # Check if subject is centered (good for cropping)
        composition = frame_features.get("composition", [])
        if "centered_subject" in composition or "rule_of_thirds" in composition:
            crop_score += 0.15

        # Check face positioning (not too close to edges)
        # TODO: In production, analyze actual landmark positions
        # For now, assume centered faces are good
        crop_score += 0.15

        crop_score = min(crop_score, 1.0)

        # ================================================================
        # 2. ZOOM POTENTIAL
        # Can we zoom in 1.5-2x without losing emotion/context?
        # ================================================================
        zoom_score = 0.6  # Base

        # Strong facial expressions survive zooming better
        expression_intensity = visual_analysis.get("expression_intensity", 0.5)
        zoom_score += expression_intensity * 0.3

        # Clear, uncluttered backgrounds help zooming
        if "clean_background" in composition:
            zoom_score += 0.1

        zoom_score = min(zoom_score, 1.0)

        # ================================================================
        # 3. TEXT OVERLAY SPACE
        # Is there clear space for title text without obscuring subject?
        # ================================================================
        text_space_score = 0.5  # Base

        # Top/bottom margins good for text
        # TODO: In production, analyze actual pixel regions
        # For now, estimate based on composition
        if "clean_background" in composition:
            text_space_score += 0.2

        # Face not filling entire frame = space for text
        if expression_intensity < 0.8:  # Not an extreme close-up
            text_space_score += 0.15

        # Niche-specific: Gaming/tech need more text space
        if niche in ["gaming", "tech", "educational", "tech_educational"]:
            # These niches heavily rely on text overlays
            text_space_score *= 1.15

        text_space_score = min(text_space_score, 1.0)

        # ================================================================
        # 4. EMOTION RESILIENCE
        # Will emotion survive simplification, filters, compression?
        # ================================================================
        emotion_resilience = 0.6  # Base

        # Strong, clear emotions survive editing better
        emotion = visual_analysis.get("dominant_emotion", "neutral")
        if emotion != "neutral":
            emotion_resilience += 0.15

        # High expression intensity = clear emotion that survives
        if expression_intensity > 0.7:
            emotion_resilience += 0.2
        elif expression_intensity > 0.5:
            emotion_resilience += 0.1

        # Extreme emotions (surprise, joy) survive better than subtle ones
        if emotion in ["surprise", "joy", "shock", "excitement"]:
            emotion_resilience += 0.1

        emotion_resilience = min(emotion_resilience, 1.0)

        # ================================================================
        # 5. COMPOSITION FLEXIBILITY
        # Multiple cropping options? Good rule of thirds adherence?
        # ================================================================
        composition_flex = 0.6  # Base

        # Rule of thirds = flexible composition
        if "rule_of_thirds" in composition:
            composition_flex += 0.2

        # Not too centered = more crop options
        if "centered_subject" not in composition:
            composition_flex += 0.1

        # Dynamic composition = more flexibility
        if "dynamic" in composition:
            composition_flex += 0.1

        composition_flex = min(composition_flex, 1.0)

        # ================================================================
        # OVERALL EDITABILITY SCORE
        # ================================================================
        # Weighted average - text space and emotion resilience most critical
        overall = (
            0.15 * crop_score
            + 0.15 * zoom_score
            + 0.30 * text_space_score  # Critical for thumbnails
            + 0.30 * emotion_resilience  # Critical for impact
            + 0.10 * composition_flex
        )

        return {
            "overall_editability": overall,
            "crop_resilience": crop_score,
            "zoom_potential": zoom_score,
            "text_overlay_space": text_space_score,
            "emotion_resilience": emotion_resilience,
            "composition_flexibility": composition_flex,
        }

    @classmethod
    def _map_niche_to_criteria_key(cls, niche: str) -> str:
        """Map niche string to criteria dictionary key."""
        niche_mapping = {
            "beauty": "beauty_lifestyle",
            "lifestyle": "beauty_lifestyle",
            "fashion": "beauty_lifestyle",
            "tech": "tech_educational",
            "educational": "tech_educational",
            "tutorial": "tech_educational",
            "gaming": "gaming",
            "games": "gaming",
            "commentary": "commentary",
            "reaction": "commentary",
            "cooking": "cooking_food",
            "food": "cooking_food",
            "fitness": "fitness_health",
            "health": "fitness_health",
        }

        for keyword, key in niche_mapping.items():
            if keyword in niche.lower():
                return key

        return "tech_educational"  # Default


# ========================================================================
# INTEGRATION WITH SCORING WEIGHTS
# ========================================================================


def get_contextual_scores(
    frame_features: dict[str, Any],
    detected_triggers: list[str],
    channel_profile: dict[str, Any],
    creative_brief: dict[str, Any],
) -> dict[str, Any]:
    """
    Get context-aware aesthetic and psychology scores.

    This is the main function that integrates contextual scoring with
    the overall scoring system.

    Args:
        frame_features: Detected aesthetic features
        detected_triggers: Detected psychology triggers
        channel_profile: Channel niche, style
        creative_brief: Creator goals, tone, preferences

    Returns:
        Context-aware scores for aesthetic and psychology components

    Example:
        >>> # Beauty channel frame with soft lighting
        >>> features = {"lighting": "soft", "colors": "warm_tones"}
        >>> triggers = ["aspiration", "emotional_contagion"]
        >>> profile = {"niche": "beauty"}
        >>> brief = {"primary_goal": "brand_building", "tone": "professional"}
        >>>
        >>> scores = get_contextual_scores(features, triggers, profile, brief)
        >>> scores["aesthetic_score"]  # High for beauty + soft lighting
        0.91
        >>> scores["psychology_score"]  # High for brand + aspiration
        0.88
    """
    # Get niche + brief specific criteria
    aesthetic_criteria = ContextualScoringCriteria.get_aesthetic_criteria(
        channel_profile, creative_brief
    )
    psychology_priorities = ContextualScoringCriteria.get_psychology_priorities(
        channel_profile, creative_brief
    )

    # Evaluate alignment
    aesthetic_eval = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        frame_features, aesthetic_criteria
    )
    psychology_eval = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, psychology_priorities
    )

    return {
        "aesthetic_score": aesthetic_eval["overall_score"],
        "aesthetic_details": aesthetic_eval,
        "psychology_score": psychology_eval["overall_score"],
        "psychology_details": psychology_eval,
    }
