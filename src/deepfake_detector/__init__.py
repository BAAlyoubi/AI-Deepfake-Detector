"""Attention-guided MobileNetV2-LSTM deepfake detector."""

from .model import DeepfakeDetector, build_model

__all__ = ["DeepfakeDetector", "build_model"]
