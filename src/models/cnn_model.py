"""
CNN Model for Satellite Image Feature Extraction
=================================================
Extracts spatial features from multi-spectral satellite images
using a deep convolutional neural network backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class SatelliteCNN(nn.Module):
    """
    CNN-based feature extractor for satellite imagery.
    
    Uses a ResNet50 backbone pre-trained on ImageNet,
    adapted for multi-spectral satellite images.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB, 
                           use 13 for Sentinel-2 multi-spectral).
        feature_dim (int): Output feature vector dimension.
        pretrained (bool): Use ImageNet pre-trained weights.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 512, pretrained: bool = True):
        super(SatelliteCNN, self).__init__()
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Load ResNet50 backbone
        backbone = models.resnet50(pretrained=pretrained)
        
        # Adapt first conv layer for multi-spectral input
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final FC layer — use as feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, H, W)
        
        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        # Backbone feature extraction
        features = self.backbone(x)          # (B, 2048, H', W')
        
        # Global average pooling
        pooled = self.gap(features)          # (B, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 2048)
        
        # Project to feature space
        output = self.projection(pooled)     # (B, feature_dim)
        
        return output


class DisasterCNNClassifier(nn.Module):
    """
    Standalone CNN classifier for single-modality disaster prediction.
    Used as a baseline and for ablation studies.
    
    Args:
        in_channels (int): Input image channels.
        num_classes (int): Number of disaster categories.
        feature_dim (int): Intermediate feature dimension.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, feature_dim: int = 512):
        super(DisasterCNNClassifier, self).__init__()
        
        self.feature_extractor = SatelliteCNN(
            in_channels=in_channels,
            feature_dim=feature_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Satellite image tensor (B, C, H, W)
        
        Returns:
            logits: Class logits (B, num_classes)
            features: Feature embeddings (B, feature_dim)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


if __name__ == "__main__":
    # Quick test
    model = SatelliteCNN(in_channels=13, feature_dim=512)
    dummy_input = torch.randn(4, 13, 224, 224)
    output = model(dummy_input)
    print(f"CNN Output shape: {output.shape}")   # Expected: (4, 512)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
