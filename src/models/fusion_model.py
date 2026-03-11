"""
Multimodal Fusion Network for Disaster Prediction
==================================================
Fuses spatial features (CNN) with temporal features (LSTM)
via cross-modal attention and late fusion to predict disasters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .cnn_model import SatelliteCNN
from .lstm_model import SensorLSTM


# Disaster categories
DISASTER_CLASSES = {
    0: "No Disaster",
    1: "Flood",
    2: "Earthquake",
    3: "Wildfire"
}


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention to allow image features to attend
    to time-series features and vice versa.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, 1, D) — query modality
            key_value: (B, 1, D) — key/value modality
        
        Returns:
            attended: (B, D) cross-attended features
        """
        attended, _ = self.attention(query, key_value, key_value)
        return self.norm(attended.squeeze(1) + query.squeeze(1))


class MultiModalFusionNet(nn.Module):
    """
    Full multimodal fusion network for disaster prediction.
    
    Architecture:
        1. CNN extracts spatial features from satellite images
        2. LSTM extracts temporal features from sensor/weather data
        3. Cross-modal attention fuses the two modalities
        4. Classification head produces disaster predictions
    
    Args:
        image_channels (int): Satellite image input channels.
        sensor_input_size (int): Number of sensor/weather features per timestep.
        feature_dim (int): Shared feature dimension across modalities.
        num_classes (int): Number of disaster categories.
        num_attention_heads (int): Heads in cross-modal attention.
    """

    def __init__(
        self,
        image_channels: int = 13,
        sensor_input_size: int = 32,
        feature_dim: int = 512,
        num_classes: int = 4,
        num_attention_heads: int = 8
    ):
        super(MultiModalFusionNet, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # --- Modality Encoders ---
        self.image_encoder = SatelliteCNN(
            in_channels=image_channels,
            feature_dim=feature_dim
        )

        self.sensor_encoder = SensorLSTM(
            input_size=sensor_input_size,
            feature_dim=feature_dim
        )

        # --- Cross-Modal Attention ---
        self.image_to_sensor_attn = CrossModalAttention(feature_dim, num_attention_heads)
        self.sensor_to_image_attn = CrossModalAttention(feature_dim, num_attention_heads)

        # --- Fusion Layer ---
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.LayerNorm(feature_dim)
        )

        # --- Disaster Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # --- Risk Level Regression Head (0.0 - 1.0) ---
        self.risk_regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        satellite_img: torch.Tensor,
        sensor_data: torch.Tensor,
        sensor_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal fusion network.
        
        Args:
            satellite_img: Satellite image (B, C, H, W)
            sensor_data: Time-series sensor data (B, T, F)
            sensor_lengths: Actual sequence lengths (optional)
        
        Returns:
            dict with keys:
                'logits': Class logits (B, num_classes)
                'probs': Class probabilities (B, num_classes)
                'risk_score': Continuous risk level 0-1 (B, 1)
                'image_features': CNN feature vector (B, D)
                'sensor_features': LSTM feature vector (B, D)
                'fused_features': Fused representation (B, D)
        """
        # --- Encode each modality ---
        image_features = self.image_encoder(satellite_img)                     # (B, D)
        sensor_features, _ = self.sensor_encoder(sensor_data, sensor_lengths)  # (B, D)

        # --- Cross-modal attention ---
        img_q = image_features.unsqueeze(1)      # (B, 1, D)
        sen_q = sensor_features.unsqueeze(1)     # (B, 1, D)

        img_attended = self.image_to_sensor_attn(img_q, sen_q)   # (B, D)
        sen_attended = self.sensor_to_image_attn(sen_q, img_q)   # (B, D)

        # --- Fusion ---
        fused = torch.cat([img_attended, sen_attended], dim=-1)  # (B, 2D)
        fused_features = self.fusion(fused)                       # (B, D)

        # --- Predictions ---
        logits = self.classifier(fused_features)                  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)                         # (B, num_classes)
        risk_score = self.risk_regressor(fused_features)          # (B, 1)

        return {
            "logits": logits,
            "probs": probs,
            "risk_score": risk_score,
            "image_features": image_features,
            "sensor_features": sensor_features,
            "fused_features": fused_features
        }


if __name__ == "__main__":
    # Quick test
    model = MultiModalFusionNet(
        image_channels=13,
        sensor_input_size=32,
        feature_dim=512,
        num_classes=4
    )

    img = torch.randn(4, 13, 224, 224)
    sensors = torch.randn(4, 72, 32)

    output = model(img, sensors)
    print("Fusion Model Output:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
