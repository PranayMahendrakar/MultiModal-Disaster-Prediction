"""
Inference Module — Early Warning System
=========================================
Loads a trained MultiModalFusionNet and runs disaster risk
predictions on new satellite/sensor/weather data inputs.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional

from src.models.fusion_model import MultiModalFusionNet, DISASTER_CLASSES


RISK_LEVELS = {
    (0.0, 0.3):  "LOW",
    (0.3, 0.6):  "MODERATE",
    (0.6, 0.8):  "HIGH",
    (0.8, 1.01): "CRITICAL"
}

def get_risk_level(score: float) -> str:
    for (low, high), level in RISK_LEVELS.items():
        if low <= score < high:
            return level
    return "CRITICAL"


class DisasterPredictor:
    """
    High-level inference interface for the MultiModal Disaster Prediction system.
    
    Provides an early warning API that accepts raw data paths or
    pre-processed tensors and returns structured prediction results.
    
    Args:
        model_path (str): Path to trained model checkpoint (.pth file).
        device (str): Inference device ('cuda', 'cpu', or 'auto').
        model_config (dict): Model architecture config (optional).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        model_config: Optional[dict] = None
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default model config
        config = model_config or {
            "image_channels": 13,
            "sensor_input_size": 32,
            "feature_dim": 512,
            "num_classes": 4
        }

        # Load model
        self.model = MultiModalFusionNet(**config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"Model loaded from {model_path} on {self.device}")

    @torch.no_grad()
    def predict_tensors(
        self,
        satellite_img: torch.Tensor,
        sensor_data: torch.Tensor
    ) -> Dict:
        """
        Run inference on pre-processed tensors.
        
        Args:
            satellite_img: (1, C, H, W) satellite image tensor
            sensor_data: (1, T, F) time-series sensor tensor
        
        Returns:
            Prediction dictionary with type, risk level, confidence, and scores.
        """
        satellite_img = satellite_img.to(self.device)
        sensor_data = sensor_data.to(self.device)

        output = self.model(satellite_img, sensor_data)

        probs = output["probs"].squeeze(0).cpu().numpy()
        risk_score = output["risk_score"].squeeze().cpu().item()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            "type": DISASTER_CLASSES[pred_class],
            "class_id": pred_class,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
            "class_probabilities": {
                DISASTER_CLASSES[i]: float(probs[i])
                for i in range(len(probs))
            }
        }

    def predict(
        self,
        satellite_image: Union[str, np.ndarray],
        sensor_data: Union[str, np.ndarray],
        weather_data: Optional[Union[str, dict]] = None
    ) -> Dict:
        """
        High-level prediction from raw data paths or arrays.
        
        Args:
            satellite_image: Path to GeoTIFF or numpy array (C, H, W)
            sensor_data: Path to CSV or numpy array (T, F)
            weather_data: Path to JSON or dict with weather features (optional)
        
        Returns:
            Prediction result dict.
        """
        # --- Load satellite image ---
        if isinstance(satellite_image, str):
            try:
                import rasterio
                with rasterio.open(satellite_image) as src:
                    img_array = src.read().astype(np.float32)  # (C, H, W)
            except ImportError:
                raise ImportError("Install rasterio: pip install rasterio")
        else:
            img_array = satellite_image.astype(np.float32)

        # Normalize to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, C, H, W)

        # Resize to 224x224 if needed
        if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
            import torch.nn.functional as F
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

        # --- Load sensor data ---
        if isinstance(sensor_data, str):
            import pandas as pd
            df = pd.read_csv(sensor_data)
            sensor_array = df.values.astype(np.float32)
        else:
            sensor_array = sensor_data.astype(np.float32)

        sensor_tensor = torch.from_numpy(sensor_array).unsqueeze(0)  # (1, T, F)

        return self.predict_tensors(img_tensor, sensor_tensor)


if __name__ == "__main__":
    # Demo with random tensors (no model file needed for shape testing)
    print("DisasterPredictor — Demo Mode")
    print("-" * 40)
    img = torch.randn(1, 13, 224, 224)
    sensors = torch.randn(1, 72, 32)
    print(f"Input satellite image: {img.shape}")
    print(f"Input sensor data:     {sensors.shape}")
    print("\nExpected output keys: type, class_id, confidence, risk_score, risk_level, class_probabilities")
