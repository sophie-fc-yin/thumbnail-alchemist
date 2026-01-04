"""Vision feature analysis for frame scoring.

This module provides comprehensive vision feature computation including:
- Image quality analysis (brightness, contrast, sharpness, noise)
- Face quality metrics (expression strength, eye contact, facial clarity)
- Composition analysis (rule of thirds, color harmony, visual balance)
- Technical quality (sharpness, noise, exposure)
- Aesthetic scoring
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.thumbnail_agent.contextual_scoring import ContextualScoringCriteria
from app.vision.face_analysis import FaceExpressionAnalyzer

logger = logging.getLogger(__name__)


def compute_vision_features(
    frame_path: str | Path,
    niche: str = "general",
    face_analyzer: Any = None,
) -> dict[str, Any]:
    """
    Compute comprehensive vision features for a frame.

    Calculates all vision features that feed into the final score:
    - Facial analysis (performed internally)
    - Aesthetics (image quality, lighting, color, sharpness, noise)
    - Editability (crop resilience, zoom potential, text overlay space)
    - Composition (rule of thirds, contrast, color harmony, visual balance, background)
    - Technical quality (sharpness, noise level, exposure quality)
    - Face quality (expression strength, eye contact, facial clarity, emotion authenticity)

    Args:
        frame_path: Path to the frame image file
        niche: Content niche for editability scoring (default: "general")
        face_analyzer: Optional FaceExpressionAnalyzer instance (creates one if not provided)

    Returns:
        Dictionary containing all vision features:
        {
            "face_analysis": {...},  # Face analysis results
            "aesthetics": {
                "score": float [0, 1],
                "image_quality": {
                    "brightness": float,
                    "subject_brightness": float,
                    "contrast": float,
                    "sharpness": float,
                    "noise_level": float,
                    "is_too_dark": bool,
                    "is_too_bright": bool
                },
                "frame_features": {...}
            },
            "editability": {
                "overall_editability": float [0, 1],
                "crop_resilience": float [0, 1],
                "zoom_potential": float [0, 1],
                "text_overlay_space": float [0, 1],
                "emotion_resilience": float [0, 1],
                "composition_flexibility": float [0, 1]
            },
            "composition": {
                "overall_score": float [0, 1],
                "rule_of_thirds": float [0, 1],
                "contrast_strength": float [0, 1],
                "color_harmony": float [0, 1],
                "visual_balance": float [0, 1],
                "background_cleanliness": float [0, 1]
            },
            "technical_quality": {
                "overall_score": float [0, 1],
                "sharpness": float [0, 1],
                "noise_level": float [0, 1],  # Inverted: higher = less noise
                "exposure_quality": float [0, 1]
            },
            "face_quality": {
                "overall_score": float [0, 1],
                "expression_strength": float [0, 1],
                "eye_contact_quality": float [0, 1],
                "facial_clarity": float [0, 1],
                "emotion_authenticity": float [0, 1],
                "composition_quality": float [0, 1]
            }
        }
    """
    frame_path = Path(frame_path)

    # Perform face analysis if analyzer is provided or create one
    if face_analyzer is None:
        face_analyzer = FaceExpressionAnalyzer()

    # Analyze face
    face_analysis = {
        "has_face": False,
        "dominant_emotion": "unknown",
        "expression_intensity": 0.0,
        "eye_openness": 0.0,
        "mouth_openness": 0.0,
        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    }

    if frame_path.exists():
        try:
            face_analysis = face_analyzer.analyze_frame(str(frame_path))
        except Exception as e:
            logger.warning("Face analysis for frame %s failed: %s", frame_path, e, exc_info=True)

    # Analyze image quality
    image_quality = _analyze_image_quality(str(frame_path))

    # Extract frame features
    frame_features = _extract_frame_features(image_quality, face_analysis)

    # Compute editability (requires ContextualScoringCriteria)
    editability = ContextualScoringCriteria.evaluate_editability(
        frame_features=frame_features,
        visual_analysis=face_analysis,
        niche=niche,
    )

    # Compute other scores (now return detailed dicts)
    face_quality = _compute_face_quality(face_analysis)
    composition = _compute_composition_score(face_analysis, image_quality)
    technical = _compute_technical_quality(face_analysis, image_quality)

    # Compute aesthetics score (simplified - based on image quality)
    aesthetic_score = _compute_aesthetic_score(image_quality, frame_features)

    return {
        "face_analysis": face_analysis,
        "aesthetics": {
            "score": aesthetic_score,
            "image_quality": image_quality,
            "frame_features": frame_features,
        },
        "editability": editability,
        "composition": composition,  # Now a dict with components
        "technical_quality": technical,  # Now a dict with components
        "face_quality": face_quality,  # Now a dict with components
    }


def compute_vision_features_batch(
    frame_paths: list[str | Path],
    niche: str = "general",
) -> list[dict[str, Any]]:
    """
    Compute vision features for multiple frames efficiently.

    Creates a single FaceExpressionAnalyzer and reuses it for all frames.

    Args:
        frame_paths: List of paths to frame image files
        niche: Content niche for editability scoring (default: "general")

    Returns:
        List of vision feature dictionaries (one per frame, same format as compute_vision_features)
    """
    # Create analyzer once and reuse for all frames (more efficient)
    face_analyzer = FaceExpressionAnalyzer()

    results = []
    for frame_path in frame_paths:
        try:
            features = compute_vision_features(
                frame_path=frame_path,
                niche=niche,
                face_analyzer=face_analyzer,  # Reuse analyzer
            )
            results.append(features)
        except Exception as e:
            logger.warning("Vision analysis for frame %s failed: %s", frame_path, e, exc_info=True)
            results.append({})  # Empty dict on failure

    return results


def _analyze_image_quality(image_path: str) -> dict[str, Any]:
    """Analyze comprehensive image quality metrics from pixel data."""
    try:
        # Check if file exists before attempting to read
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            logger.debug("Image file does not exist: %s", image_path)
            return {
                "brightness": 0.5,
                "subject_brightness": 0.5,
                "contrast": 0.5,
                "sharpness": 0.0,
                "noise_level": 1.0,
            }

        img = cv2.imread(image_path)
        if img is None:
            logger.debug("Failed to load image (file may be corrupted): %s", image_path)
            return {
                "brightness": 0.5,
                "subject_brightness": 0.5,
                "contrast": 0.5,
                "is_too_dark": False,
                "is_too_bright": False,
                "mean_brightness": 128,
                "sharpness": 0.5,
                "noise_level": 0.5,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Overall brightness
        mean_brightness = float(np.mean(gray))
        brightness_normalized = mean_brightness / 255.0

        # Center region (subject) brightness
        center_y_start = int(h * 0.3)
        center_y_end = int(h * 0.7)
        center_x_start = int(w * 0.3)
        center_x_end = int(w * 0.7)
        center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
        subject_brightness_raw = float(np.mean(center_region))
        subject_brightness = subject_brightness_raw / 255.0

        # Contrast
        std_dev = float(np.std(gray))
        contrast_normalized = min(std_dev / 80.0, 1.0)

        # Sharpness (Laplacian variance - higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_variance = float(np.var(laplacian))
        # Normalize: typical range is 0-1000, normalize to [0, 1]
        sharpness = min(sharpness_variance / 1000.0, 1.0)

        # Noise level (estimate from high-frequency content)
        # Use standard deviation of Laplacian as noise proxy
        noise_estimate = float(np.std(laplacian))
        # Normalize: typical range is 0-50, normalize to [0, 1]
        noise_level = min(noise_estimate / 50.0, 1.0)

        # Exposure detection
        is_too_dark = subject_brightness < 0.235
        is_too_bright = subject_brightness > 0.90

        return {
            "brightness": brightness_normalized,
            "subject_brightness": subject_brightness,
            "contrast": contrast_normalized,
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright,
            "mean_brightness": mean_brightness,
            "sharpness": sharpness,
            "noise_level": noise_level,
        }
    except Exception as e:
        logger.warning("Failed to analyze image quality for %s: %s", image_path, e)
        return {
            "brightness": 0.5,
            "subject_brightness": 0.5,
            "contrast": 0.5,
            "is_too_dark": False,
            "is_too_bright": False,
            "mean_brightness": 128,
            "sharpness": 0.5,
            "noise_level": 0.5,
        }


def _extract_frame_features(
    image_quality: dict[str, Any],
    face_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Extract frame features for scoring."""
    emotion = face_analysis.get("dominant_emotion", "neutral")
    expression_intensity = face_analysis.get("expression_intensity", 0.5)

    features = {
        "lighting": [],
        "color_palette": ["natural"],
        "composition": ["centered_subject"] if expression_intensity > 0.6 else ["standard"],
    }

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

    if contrast > 0.6:
        features["lighting"].append("high_contrast")
        features["color_palette"].append("high_contrast")
    elif contrast < 0.3:
        features["lighting"].append("flat")

    if emotion in ["joy", "excitement"]:
        features["color_palette"].append("warm_tones")
    elif emotion in ["surprise", "shock"]:
        features["lighting"].append("dramatic")
    elif emotion in ["serious", "focused"]:
        features["polish_level"] = ["professional"]

    features["image_quality"] = image_quality
    return features


def _compute_face_quality(face_analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Compute comprehensive face quality metrics.

    Returns:
        {
            "overall_score": float [0, 1],
            "expression_strength": float [0, 1],
            "eye_contact_quality": float [0, 1],
            "facial_clarity": float [0, 1],
            "emotion_authenticity": float [0, 1],
            "composition_quality": float [0, 1]
        }
    """
    if not face_analysis or not face_analysis.get("has_face", False):
        return {
            "overall_score": 0.3,
            "expression_strength": 0.0,
            "eye_contact_quality": 0.0,
            "facial_clarity": 0.0,
            "emotion_authenticity": 0.0,
            "composition_quality": 0.0,
        }

    # Expression strength (0.30 weight)
    expression_intensity = face_analysis.get("expression_intensity", 0.5)
    emotion = face_analysis.get("dominant_emotion", "neutral")
    # Strong emotions get higher score
    if emotion in ["surprise", "happiness", "shock", "excitement"]:
        expression_strength = min(expression_intensity * 1.2, 1.0)
    elif emotion != "neutral":
        expression_strength = expression_intensity
    else:
        expression_strength = expression_intensity * 0.7

    # Eye contact quality (0.25 weight) - based on head pose
    head_pose = face_analysis.get("head_pose", {})
    yaw = abs(head_pose.get("yaw", 0.0))
    pitch = abs(head_pose.get("pitch", 0.0))
    # Direct camera gaze: yaw < 15, pitch < 10
    if yaw < 15 and pitch < 10:
        eye_contact_quality = 1.0
    elif yaw < 30 and pitch < 20:
        eye_contact_quality = 0.7
    elif yaw < 45 and pitch < 30:
        eye_contact_quality = 0.4
    else:
        eye_contact_quality = 0.1

    # Facial clarity (0.20 weight) - based on eye/mouth openness and sharpness
    eye_openness = face_analysis.get("eye_openness", 0.5)
    mouth_openness = face_analysis.get("mouth_openness", 0.0)
    # Good clarity: eyes open, mouth not too open
    facial_clarity = eye_openness * 0.6 + (1.0 - min(mouth_openness, 0.5)) * 0.4

    # Emotion authenticity (0.15 weight) - based on expression intensity
    # Higher intensity = more authentic
    emotion_authenticity = expression_intensity

    # Composition quality (0.10 weight) - based on head pose alignment
    roll = abs(head_pose.get("roll", 0.0))
    if yaw < 20 and pitch < 15 and roll < 15:
        composition_quality = 1.0
    elif yaw < 40 and pitch < 30 and roll < 30:
        composition_quality = 0.6
    else:
        composition_quality = 0.3

    # Weighted overall score
    overall_score = (
        0.30 * expression_strength
        + 0.25 * eye_contact_quality
        + 0.20 * facial_clarity
        + 0.15 * emotion_authenticity
        + 0.10 * composition_quality
    )

    return {
        "overall_score": min(overall_score, 1.0),
        "expression_strength": expression_strength,
        "eye_contact_quality": eye_contact_quality,
        "facial_clarity": facial_clarity,
        "emotion_authenticity": emotion_authenticity,
        "composition_quality": composition_quality,
    }


def _compute_composition_score(
    face_analysis: dict[str, Any],
    image_quality: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute comprehensive composition quality.

    Returns:
        {
            "overall_score": float [0, 1],
            "rule_of_thirds": float [0, 1],
            "contrast_strength": float [0, 1],
            "color_harmony": float [0, 1],
            "visual_balance": float [0, 1],
            "background_cleanliness": float [0, 1]
        }
    """
    if not face_analysis.get("has_face", False):
        return {
            "overall_score": 0.3,
            "rule_of_thirds": 0.0,
            "contrast_strength": image_quality.get("contrast", 0.5),
            "color_harmony": 0.5,
            "visual_balance": 0.5,
            "background_cleanliness": 0.5,
        }

    # Rule of thirds (0.25 weight) - simplified: centered face = good
    head_pose = face_analysis.get("head_pose", {})
    yaw = abs(head_pose.get("yaw", 0.0))
    pitch = abs(head_pose.get("pitch", 0.0))
    # Well-centered face approximates rule of thirds
    if yaw < 20 and pitch < 15:
        rule_of_thirds = 0.8
    elif yaw < 40 and pitch < 30:
        rule_of_thirds = 0.5
    else:
        rule_of_thirds = 0.2

    # Contrast strength (0.25 weight)
    contrast = image_quality.get("contrast", 0.5)
    contrast_strength = contrast

    # Color harmony (0.20 weight) - simplified based on contrast
    # High contrast often indicates good color separation
    color_harmony = min(contrast * 1.2, 1.0)

    # Visual balance (0.15 weight) - based on head pose symmetry
    roll = abs(head_pose.get("roll", 0.0))
    if roll < 10:
        visual_balance = 1.0
    elif roll < 20:
        visual_balance = 0.7
    else:
        visual_balance = 0.4

    # Background cleanliness (0.15 weight) - simplified
    # Assume good if subject is well-lit (good contrast between subject and background)
    subject_brightness = image_quality.get("subject_brightness", 0.5)
    overall_brightness = image_quality.get("brightness", 0.5)
    # High contrast between subject and overall = clean background
    brightness_diff = abs(subject_brightness - overall_brightness)
    background_cleanliness = min(brightness_diff * 2.0, 1.0)

    # Weighted overall score
    overall_score = (
        0.25 * rule_of_thirds
        + 0.25 * contrast_strength
        + 0.20 * color_harmony
        + 0.15 * visual_balance
        + 0.15 * background_cleanliness
    )

    return {
        "overall_score": min(overall_score, 1.0),
        "rule_of_thirds": rule_of_thirds,
        "contrast_strength": contrast_strength,
        "color_harmony": color_harmony,
        "visual_balance": visual_balance,
        "background_cleanliness": background_cleanliness,
    }


def _compute_technical_quality(
    face_analysis: dict[str, Any],
    image_quality: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute comprehensive technical quality.

    Returns:
        {
            "overall_score": float [0, 1],
            "sharpness": float [0, 1],
            "noise_level": float [0, 1],  # Lower is better, so we'll invert
            "exposure_quality": float [0, 1]
        }
    """
    if not face_analysis.get("has_face", False):
        return {
            "overall_score": 0.2,
            "sharpness": image_quality.get("sharpness", 0.5),
            "noise_level": image_quality.get("noise_level", 0.5),
            "exposure_quality": 0.5,
        }

    # Sharpness (0.40 weight)
    sharpness = image_quality.get("sharpness", 0.5)

    # Noise level (0.30 weight) - invert so lower noise = higher score
    noise_level_raw = image_quality.get("noise_level", 0.5)
    noise_level = 1.0 - noise_level_raw  # Invert: lower noise = higher score

    # Exposure quality (0.30 weight)
    subject_brightness = image_quality.get("subject_brightness", 0.5)
    is_too_dark = image_quality.get("is_too_dark", False)
    is_too_bright = image_quality.get("is_too_bright", False)

    if is_too_dark or is_too_bright:
        exposure_quality = 0.3
    elif 0.4 <= subject_brightness <= 0.8:
        exposure_quality = 1.0  # Optimal exposure
    elif 0.3 <= subject_brightness < 0.4 or 0.8 < subject_brightness <= 0.9:
        exposure_quality = 0.7
    else:
        exposure_quality = 0.5

    # Weighted overall score
    overall_score = 0.40 * sharpness + 0.30 * noise_level + 0.30 * exposure_quality

    return {
        "overall_score": min(overall_score, 1.0),
        "sharpness": sharpness,
        "noise_level": noise_level,  # Already inverted
        "exposure_quality": exposure_quality,
    }


def _compute_aesthetic_score(
    image_quality: dict[str, Any],
    frame_features: dict[str, Any],
) -> float:
    """Compute aesthetic score based on image quality and features."""
    score = 0.7  # Base score

    # Brightness penalty
    if image_quality.get("is_too_dark"):
        score *= 0.6  # Severe penalty for dark frames
    elif image_quality.get("is_too_bright"):
        score *= 0.8  # Moderate penalty for overexposed

    # Contrast bonus
    contrast = image_quality.get("contrast", 0.5)
    if contrast > 0.6:
        score += 0.1
    elif contrast < 0.3:
        score -= 0.1

    # Lighting quality
    lighting = frame_features.get("lighting", [])
    if "well_lit" in lighting or "bright" in lighting:
        score += 0.1
    elif "dark" in lighting or "dim" in lighting:
        score -= 0.15

    return max(min(score, 1.0), 0.0)
