"""Face expression analysis using MediaPipe + FER+ ONNX model.

This module provides facial expression intensity detection for adaptive frame sampling.
Uses MediaPipe for face structure and FER+ for expression intensity.
"""

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from mediapipe import Image as MPImage
from mediapipe import ImageFormat, tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions


class FaceExpressionAnalyzer:
    """Analyze facial expressions using MediaPipe + FER+ emotion model."""

    def __init__(self, model_path: Path | None = None):
        """
        Initialize face expression analyzer.

        Args:
            model_path: Path to FER+ ONNX model. If None, downloads from HuggingFace.
        """
        # MediaPipe FaceLandmarker (new API in MediaPipe 0.10+)
        # Download face landmarker model
        model_dir = Path.home() / ".cache" / "mediapipe"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_asset_path = model_dir / "face_landmarker.task"

        if not model_asset_path.exists():
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, str(model_asset_path))
            print(f"Downloaded to {model_asset_path}")

        # Configure FaceLandmarker
        options = FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Load FER+ model (download if not cached)
        if model_path is None:
            model_dir_fer = Path.home() / ".cache" / "ferplus"
            model_dir_fer.mkdir(parents=True, exist_ok=True)
            model_path = model_dir_fer / "emotion-ferplus-8.onnx"

            if not model_path.exists():
                # Download from ONNX Model Zoo
                import urllib.request

                print("Downloading FER+ ONNX model (~50MB)...")
                url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
                urllib.request.urlretrieve(url, str(model_path))
                print(f"✓ Downloaded FER+ model to {model_path}")
            else:
                print(f"✓ Using cached FER+ model from {model_path}")

        # Load FER+ emotion model
        self.emotion_model = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        # FER+ emotion labels (index → label)
        self.emotion_labels = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]

    def analyze_frame(self, frame_path: Path | str) -> dict[str, Any]:
        """
        Analyze facial expression in a single frame.

        Args:
            frame_path: Path to image file

        Returns:
            Dictionary with:
                - has_face: bool - whether a face was detected
                - expression_intensity: float [0,1] - how far from neutral (1 - P(neutral))
                - eye_openness: float [0,1] - average eye openness
                - mouth_openness: float [0,1] - mouth openness
                - head_pose: dict - head orientation (pitch, yaw, roll)
                - emotion_probs: dict - probability for each emotion
                - landmarks: list - facial landmarks (for motion calculation)
        """
        # Check if file exists before attempting to read
        frame_path_obj = Path(frame_path)
        if not frame_path_obj.exists():
            return self._empty_result()

        # Read and convert image
        image = cv2.imread(str(frame_path))
        if image is None:
            return self._empty_result()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Create MediaPipe Image object
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)

        # MediaPipe face detection and landmarks (new API)
        detection_result = self.face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return self._empty_result()

        landmarks = detection_result.face_landmarks[0]

        # Extract geometric features
        eye_openness = self._calculate_eye_openness(landmarks, h, w)
        mouth_openness = self._calculate_mouth_openness(landmarks, h, w)
        head_pose = self._estimate_head_pose(landmarks, h, w)

        # Crop face for emotion analysis
        face_crop = self._crop_face_for_emotion(image, landmarks, h, w)

        if face_crop is None:
            return self._empty_result()

        # Get emotion probabilities from FER+
        emotion_probs = self._get_emotion_probs(face_crop)

        # Calculate expression intensity (key metric!)
        # This is 1 - P(neutral), measuring "how much is happening"
        expression_intensity = 1.0 - emotion_probs.get("neutral", 0.5)

        # Determine dominant emotion (highest probability)
        if emotion_probs:
            dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "neutral"

        # Landmarks removed - not used anywhere in the codebase
        # If needed in future, can be re-enabled for facial motion tracking

        return {
            "has_face": True,
            "expression_intensity": float(expression_intensity),
            "dominant_emotion": dominant_emotion,  # Add missing field!
            "eye_openness": float(eye_openness),
            "mouth_openness": float(mouth_openness),
            "head_pose": head_pose,
            "emotion_probs": emotion_probs,
        }

    def _empty_result(self) -> dict[str, Any]:
        """Return empty result when no face is detected."""
        return {
            "has_face": False,
            "expression_intensity": 0.0,
            "dominant_emotion": "unknown",  # Add missing field!
            "eye_openness": 0.0,
            "mouth_openness": 0.0,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "emotion_probs": {},
        }

    def _calculate_eye_openness(self, landmarks, img_h: int, img_w: int) -> float:
        """
        Calculate average eye openness using MediaPipe landmarks.

        Uses eye aspect ratio (EAR) - ratio of eye height to width.
        """
        # Left eye landmarks (indices from MediaPipe Face Mesh)
        LEFT_EYE_UPPER = [159, 160]
        LEFT_EYE_LOWER = [144, 145]
        LEFT_EYE_LEFT = [33]
        LEFT_EYE_RIGHT = [133]

        # Right eye landmarks
        RIGHT_EYE_UPPER = [386, 387]
        RIGHT_EYE_LOWER = [373, 374]
        RIGHT_EYE_LEFT = [362]
        RIGHT_EYE_RIGHT = [263]

        def eye_aspect_ratio(upper, lower, left, right):
            # Vertical distance (average of two measurements)
            v1 = np.linalg.norm(
                np.array([landmarks[upper[0]].x, landmarks[upper[0]].y])
                - np.array([landmarks[lower[0]].x, landmarks[lower[0]].y])
            )
            v2 = np.linalg.norm(
                np.array([landmarks[upper[1]].x, landmarks[upper[1]].y])
                - np.array([landmarks[lower[1]].x, landmarks[lower[1]].y])
            )

            # Horizontal distance
            h = np.linalg.norm(
                np.array([landmarks[left[0]].x, landmarks[left[0]].y])
                - np.array([landmarks[right[0]].x, landmarks[right[0]].y])
            )

            # Eye aspect ratio
            return (v1 + v2) / (2.0 * h) if h > 0 else 0

        left_ear = eye_aspect_ratio(LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_ear = eye_aspect_ratio(
            RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT
        )

        # Average and normalize to [0, 1] (typical EAR range is 0.1-0.3)
        avg_ear = (left_ear + right_ear) / 2.0
        normalized = min(max((avg_ear - 0.1) / 0.2, 0.0), 1.0)

        return normalized

    def _calculate_mouth_openness(self, landmarks, img_h: int, img_w: int) -> float:
        """
        Calculate mouth openness using MediaPipe landmarks.

        Uses mouth aspect ratio (MAR) - ratio of mouth height to width.
        """
        # Mouth landmarks
        MOUTH_UPPER = [13, 14]
        MOUTH_LOWER = [312, 311]
        MOUTH_LEFT = [61]
        MOUTH_RIGHT = [291]

        # Vertical distance (average)
        v1 = np.linalg.norm(
            np.array([landmarks[MOUTH_UPPER[0]].x, landmarks[MOUTH_UPPER[0]].y])
            - np.array([landmarks[MOUTH_LOWER[0]].x, landmarks[MOUTH_LOWER[0]].y])
        )
        v2 = np.linalg.norm(
            np.array([landmarks[MOUTH_UPPER[1]].x, landmarks[MOUTH_UPPER[1]].y])
            - np.array([landmarks[MOUTH_LOWER[1]].x, landmarks[MOUTH_LOWER[1]].y])
        )

        # Horizontal distance
        h = np.linalg.norm(
            np.array([landmarks[MOUTH_LEFT[0]].x, landmarks[MOUTH_LEFT[0]].y])
            - np.array([landmarks[MOUTH_RIGHT[0]].x, landmarks[MOUTH_RIGHT[0]].y])
        )

        # Mouth aspect ratio
        mar = (v1 + v2) / (2.0 * h) if h > 0 else 0

        # Normalize to [0, 1] (typical MAR range is 0.1-0.6)
        normalized = min(max((mar - 0.1) / 0.5, 0.0), 1.0)

        return normalized

    def _estimate_head_pose(self, landmarks, img_h: int, img_w: int) -> dict[str, float]:
        """
        Estimate head pose (pitch, yaw, roll) using facial landmarks.

        This is a simplified estimation using nose and eye positions.
        """
        # Key landmarks for head pose
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        mouth_center = landmarks[13]

        # Yaw (left-right rotation) - based on nose position
        yaw = np.arctan2(nose_tip.x - 0.5, 0.5) * 180 / np.pi

        # Pitch (up-down tilt) - based on nose-mouth distance
        pitch = (nose_tip.y - mouth_center.y) * 100

        # Roll (head tilt) - based on eye horizontal alignment
        roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / np.pi

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
        }

    def _crop_face_for_emotion(
        self, image: np.ndarray, landmarks, img_h: int, img_w: int
    ) -> np.ndarray | None:
        """
        Crop face region for emotion analysis.

        FER+ expects 64x64 grayscale face images.
        """
        # Get bounding box from landmarks
        x_coords = [lm.x * img_w for lm in landmarks]
        y_coords = [lm.y * img_h for lm in landmarks]

        x_min = max(0, int(min(x_coords)))
        x_max = min(img_w, int(max(x_coords)))
        y_min = max(0, int(min(y_coords)))
        y_max = min(img_h, int(max(y_coords)))

        # Add padding (20%)
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)

        x_min = max(0, x_min - padding_x)
        x_max = min(img_w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(img_h, y_max + padding_y)

        # Crop face
        face = image[y_min:y_max, x_min:x_max]

        if face.size == 0:
            return None

        # Convert to grayscale and resize to 64x64 (FER+ input size)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))

        return resized

    def _get_emotion_probs(self, face_crop: np.ndarray) -> dict[str, float]:
        """
        Get emotion probabilities from FER+ model.

        Args:
            face_crop: 64x64 grayscale face image

        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        # Prepare input for ONNX model
        # FER+ expects shape (1, 1, 64, 64) and pixel values [0, 255]
        input_tensor = face_crop.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add channel dim

        # Run inference
        input_name = self.emotion_model.get_inputs()[0].name
        output_name = self.emotion_model.get_outputs()[0].name

        outputs = self.emotion_model.run([output_name], {input_name: input_tensor})

        # Get probabilities (softmax already applied in model)
        probs = outputs[0][0]

        # Map to emotion labels
        emotion_probs = {label: float(prob) for label, prob in zip(self.emotion_labels, probs)}

        return emotion_probs


def calculate_landmark_motion(
    prev_landmarks: list[tuple[float, float, float]],
    curr_landmarks: list[tuple[float, float, float]],
) -> float:
    """
    Calculate facial landmark motion between two frames.

    Args:
        prev_landmarks: Previous frame landmarks [(x, y, z), ...]
        curr_landmarks: Current frame landmarks [(x, y, z), ...]

    Returns:
        Motion score [0, 1] - normalized average displacement
    """
    if not prev_landmarks or not curr_landmarks:
        return 0.0

    if len(prev_landmarks) != len(curr_landmarks):
        return 0.0

    # Calculate average Euclidean distance between corresponding landmarks
    distances = []
    for prev, curr in zip(prev_landmarks, curr_landmarks):
        dist = np.sqrt(
            (curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2 + (curr[2] - prev[2]) ** 2
        )
        distances.append(dist)

    avg_dist = np.mean(distances)

    # Normalize to [0, 1] (typical motion range is 0-0.1)
    normalized = min(max(avg_dist / 0.1, 0.0), 1.0)

    return float(normalized)
