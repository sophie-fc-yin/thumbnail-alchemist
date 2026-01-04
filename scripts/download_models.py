#!/usr/bin/env python3
"""Pre-download ML models to avoid runtime downloads."""

import sys
import urllib.request
from pathlib import Path

try:
    from transformers import AutoModelForAudioClassification, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def download_fer_model():
    """Download FER+ emotion model from ONNX model zoo."""
    model_dir = Path.home() / ".cache" / "ferplus"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "emotion-ferplus-8.onnx"

    if model_path.exists():
        print("✓ FER+ emotion model already cached")
        return True

    try:
        print("Downloading FER+ ONNX model (~50MB)...", flush=True)
        url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

        urllib.request.urlretrieve(url, str(model_path))
        print("✓ Downloaded FER+ model")
        return True
    except Exception as e:
        print(f"❌ Failed to download FER+ model: {e}", file=sys.stderr)
        return False


def download_silero_vad():
    """Download Silero VAD model via torch.hub."""
    try:
        print("Downloading Silero VAD model (~2MB)...", flush=True)
        import torch

        # This will download to ~/.cache/torch/hub
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        print("✓ Downloaded Silero VAD model")
        return True
    except Exception as e:
        print(f"❌ Failed to download Silero VAD: {e}", file=sys.stderr)
        return False


def download_mediapipe_models():
    """Download MediaPipe face landmarker model."""
    model_dir = Path.home() / ".cache" / "mediapipe"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_landmarker.task"

    if model_path.exists():
        print("✓ MediaPipe face landmarker already cached")
        return True

    try:
        print("Downloading MediaPipe face landmarker (~30MB)...", flush=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

        urllib.request.urlretrieve(url, str(model_path))
        print("✓ Downloaded MediaPipe face landmarker")
        return True
    except Exception as e:
        print(f"⚠️  MediaPipe will download on first use: {e}", file=sys.stderr)
        return True  # Don't fail setup


def download_emotion_model():
    """Download audeering emotion recognition model from HuggingFace."""
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  transformers not available, emotion model will download on first use")
        return True  # Don't fail setup

    try:
        print("Downloading emotion recognition model (~1.5GB)...", flush=True)
        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

        # Enable progress bars and resume downloads
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_info()  # Show download progress

        print("  Downloading model weights...", flush=True)
        # This will download to ~/.cache/huggingface/hub
        AutoModelForAudioClassification.from_pretrained(
            model_name,
            resume_download=True,  # Resume if interrupted
            local_files_only=False,
            trust_remote_code=True,
        )

        print("  Downloading feature extractor...", flush=True)
        AutoProcessor.from_pretrained(
            model_name,
            resume_download=True,
            local_files_only=False,
            trust_remote_code=True,
        )

        print("✓ Downloaded emotion recognition model")
        return True
    except Exception as e:
        print(f"⚠️  Emotion model will download on first use: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return True  # Don't fail setup


if __name__ == "__main__":
    print("Pre-downloading ML models...\n", flush=True)

    results = []
    results.append(download_fer_model())
    results.append(download_silero_vad())
    results.append(download_mediapipe_models())
    results.append(download_emotion_model())

    if all(results):
        print("\n✅ All models ready!")
        sys.exit(0)
    else:
        print("\n⚠️  Some models will download on first use")
        sys.exit(0)  # Don't fail setup, models will download later
