"""Visual change detection for shot/layout changes and motion spikes."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for HSV histogram analysis."""
    frame = cv2.resize(frame, (224, 224))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv


def preprocess_gray(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for motion detection (grayscale)."""
    frame = cv2.resize(frame, (224, 224))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def hsv_histogram(hsv: np.ndarray) -> np.ndarray:
    """Calculate normalized HSV histogram (H and S channels)."""
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1],  # H, S
        mask=None,
        histSize=[32, 32],
        ranges=[0, 180, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist


def hist_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Calculate Bhattacharyya distance between two histograms."""
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))


def detect_shot_change(frame_t: np.ndarray, frame_t1: np.ndarray) -> float:
    """
    Detect shot/layout change between two frames using HSV histogram comparison.

    Args:
        frame_t: Current frame (BGR)
        frame_t1: Previous frame (BGR)

    Returns:
        Visual difference score [0, 1] where higher = more change (shot cut)
    """
    hsv1 = preprocess(frame_t)
    hsv2 = preprocess(frame_t1)

    hist1 = hsv_histogram(hsv1)
    hist2 = hsv_histogram(hsv2)

    diff = hist_distance(hist1, hist2)
    return float(diff)


def detect_motion(frame_t: np.ndarray, frame_t1: np.ndarray) -> float:
    """
    Detect motion/visual change between two frames using frame differencing.

    Args:
        frame_t: Current frame (BGR)
        frame_t1: Previous frame (BGR)

    Returns:
        Motion score [0, 1] where higher = more motion
    """
    g1 = preprocess_gray(frame_t)
    g2 = preprocess_gray(frame_t1)

    diff = cv2.absdiff(g1, g2)
    motion = diff.mean() / 255.0  # normalize to 0-1
    return float(motion)


def is_motion_spike(motion_values: list[float], i: int, spike_factor: float = 2.0) -> bool:
    """
    Detect if motion value at index i is a spike (sudden increase).

    Args:
        motion_values: List of motion scores
        i: Current index
        spike_factor: Multiplier threshold for spike detection

    Returns:
        True if motion at i is a spike
    """
    if i < 3:
        return False

    baseline = sum(motion_values[i - 3 : i]) / 3
    return motion_values[i] > baseline * spike_factor


def analyze_visual_changes(frame_paths: list[Path]) -> list[dict[str, Any]]:
    """
    Analyze visual changes across a sequence of frames.

    Args:
        frame_paths: List of frame file paths (sorted by timestamp, e.g., sample_{timestamp_ms}ms.jpg)

    Returns:
        List of frame dictionaries, each containing:
            - time: Timestamp in seconds (extracted from filename)
            - filename: Frame filename/path
            - shot_change: Shot/layout change score [0, 1]
            - motion_score: Motion score [0, 1]
            - motion_spike: Boolean indicating if motion is a spike
    """
    frames = []
    motion_scores_list = []

    prev_frame = None

    for _i, frame_path in enumerate(frame_paths):
        # Extract timestamp from filename (sample_{timestamp_ms}ms.jpg)
        frame_name = frame_path.stem  # e.g., "sample_3000ms"
        try:
            timestamp_ms = int(frame_name.split("_")[1].replace("ms", ""))
            timestamp = timestamp_ms / 1000.0
        except (ValueError, IndexError):
            logger.warning("Failed to extract timestamp from frame name: %s", frame_name)
            timestamp = 0.0

        try:
            # Check if file exists before attempting to read
            if not frame_path.exists():
                logger.debug("Frame file does not exist, skipping: %s", frame_path)
                frames.append(
                    {
                        "time": timestamp,
                        "filename": str(frame_path),
                        "shot_change": 0.0,
                        "motion_score": 0.0,
                        "motion_spike": False,
                    }
                )
                motion_scores_list.append(0.0)
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning("Failed to load frame (file may be corrupted): %s", frame_path)
                frames.append(
                    {
                        "time": timestamp,
                        "filename": str(frame_path),
                        "shot_change": 0.0,
                        "motion_score": 0.0,
                        "motion_spike": False,
                    }
                )
                motion_scores_list.append(0.0)
                continue

            if prev_frame is not None:
                shot_change = detect_shot_change(frame, prev_frame)
                motion = detect_motion(frame, prev_frame)
                motion_scores_list.append(motion)
            else:
                shot_change = 0.0
                motion = 0.0
                motion_scores_list.append(0.0)

            prev_frame = frame

            # Add frame to list (motion_spike will be calculated after all scores are known)
            frames.append(
                {
                    "time": timestamp,
                    "filename": str(frame_path),
                    "shot_change": float(shot_change),
                    "motion_score": float(motion),
                    "motion_spike": False,  # Will be updated below
                }
            )

        except Exception as e:
            logger.warning("Error processing frame %s: %s", frame_path, e)
            frames.append(
                {
                    "time": timestamp,
                    "filename": str(frame_path),
                    "shot_change": 0.0,
                    "motion_score": 0.0,
                    "motion_spike": False,
                }
            )
            motion_scores_list.append(0.0)

    # Detect motion spikes and update frames
    for i in range(len(frames)):
        is_spike = is_motion_spike(motion_scores_list, i)
        frames[i]["motion_spike"] = is_spike

    return frames
