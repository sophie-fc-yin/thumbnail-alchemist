"""Lightweight vision pipeline for scoring candidate frames.

This module is intentionally standalone and not wired into main.py yet.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class FrameSignals:
    path: Path
    timestamp: float | None
    brightness: float | None
    sharpness: float | None
    motion: float | None
    face: float | None
    expression: float | None
    highlight_score: float | None


@dataclass(frozen=True)
class VisionPipelineConfig:
    weight_brightness: float = 0.2
    weight_sharpness: float = 0.3
    weight_motion: float = 0.2
    weight_face: float = 0.3
    weight_expression: float = 0.0
    min_face_area_ratio: float = 0.02


def analyze_frame_quality(
    frame_paths: Iterable[Path],
    config: VisionPipelineConfig | None = None,
) -> list[FrameSignals]:
    """Score frames using a small CPU-friendly vision stack."""
    cfg = config or VisionPipelineConfig()
    frames = [Path(p) for p in frame_paths]
    signals: list[FrameSignals] = []

    prev_gray = None
    face_detector = _create_face_detector()

    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            signals.append(
                FrameSignals(
                    path=path,
                    timestamp=None,
                    brightness=None,
                    sharpness=None,
                    motion=None,
                    face=None,
                    expression=None,
                    highlight_score=None,
                )
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion = None
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff) / 255.0)
        prev_gray = gray

        face_score = _detect_face_score(face_detector, image, cfg.min_face_area_ratio)
        expression_score = None

        highlight = _score_highlight(
            brightness=brightness,
            sharpness=sharpness,
            motion=motion,
            face=face_score,
            expression=expression_score,
            config=cfg,
        )

        signals.append(
            FrameSignals(
                path=path,
                timestamp=None,
                brightness=brightness,
                sharpness=sharpness,
                motion=motion,
                face=face_score,
                expression=expression_score,
                highlight_score=highlight,
            )
        )

    return signals


def rank_frames(signals: Iterable[FrameSignals]) -> list[FrameSignals]:
    """Return frames sorted by highlight score, highest first."""
    return sorted(
        signals,
        key=lambda s: (s.highlight_score is not None, s.highlight_score or 0.0),
        reverse=True,
    )


def _get_model_path() -> str:
    """Download and cache the MediaPipe face detection model."""
    model_dir = Path.home() / ".cache" / "mediapipe" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "blaze_face_short_range.tflite"

    # Download model if not already cached
    if not model_path.exists():
        model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(model_url, model_path)

    return str(model_path)


def _create_face_detector():
    """Create MediaPipe face detector using the new 0.10+ API."""
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Get model path (downloads if needed)
    model_path = _get_model_path()

    # Configure face detector options
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.3,  # Lower threshold to catch more faces
    )

    return FaceDetector.create_from_options(options)


def _detect_face_score(detector, image, min_area_ratio: float) -> float | None:
    """Detect faces using MediaPipe and return normalized area score."""
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure contiguous array for MediaPipe
    rgb_image = np.ascontiguousarray(rgb_image)

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect faces
    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return 0.0

    # Find largest face by bounding box area
    height, width = image.shape[:2]
    total_pixels = height * width
    max_area = 0.0

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        # Bounding box is in pixels, normalize to image size
        box_area_pixels = bbox.width * bbox.height
        box_area_normalized = float(box_area_pixels) / total_pixels

        if box_area_normalized > max_area:
            max_area = box_area_normalized

    if max_area < min_area_ratio:
        return 0.0

    return max_area


def _score_highlight(
    *,
    brightness: float | None,
    sharpness: float | None,
    motion: float | None,
    face: float | None,
    expression: float | None,
    config: VisionPipelineConfig,
) -> float | None:
    if brightness is None or sharpness is None:
        return None

    motion_value = motion or 0.0
    face_value = face or 0.0
    expression_value = expression or 0.0

    normalized_sharpness = min(sharpness / 1000.0, 1.0)
    normalized_brightness = 1.0 - abs(brightness - 0.5) * 2.0

    score = (
        config.weight_brightness * normalized_brightness
        + config.weight_sharpness * normalized_sharpness
        + config.weight_motion * motion_value
        + config.weight_face * face_value
        + config.weight_expression * expression_value
    )
    return max(score, 0.0)
