"""MobileNetV2-CBAM-LSTM architecture for image-level deepfake detection."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .attention import CBAM


class DeepfakeDetector(nn.Module):
    """Attention-guided spatial-sequential detector.

    MobileNetV2 extracts a compact spatial feature map. CBAM reweights channels
    and locations that are likely to contain manipulation artifacts. The feature
    map is then reshaped into a left-to-right spatial sequence and modeled by an
    LSTM before binary classification.
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.35,
        bidirectional: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        feature_channels = backbone.last_channel

        self.attention = CBAM(feature_channels)
        self.sequence_norm = nn.LayerNorm(feature_channels)
        self.lstm = nn.LSTM(
            input_size=feature_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = self.attention(features)

        batch, channels, height, width = features.shape
        sequence = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        sequence = self.sequence_norm(sequence)

        lstm_out, _ = self.lstm(sequence)
        pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)


def build_model(**kwargs: object) -> DeepfakeDetector:
    """Factory used by training, evaluation, and prediction scripts."""

    return DeepfakeDetector(**kwargs)
