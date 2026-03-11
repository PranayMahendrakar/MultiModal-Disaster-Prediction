"""
LSTM Model for Time-Series Sensor & Weather Data
=================================================
Captures temporal patterns and sequential anomalies
from IoT sensor readings and meteorological data.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SensorLSTM(nn.Module):
    """
    Bidirectional LSTM for time-series sensor and weather data.
    
    Processes sequential data from IoT sensors (temperature, humidity,
    ground vibration, water levels) and weather stations to extract
    temporal patterns that precede disaster events.
    
    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): LSTM hidden state dimension.
        num_layers (int): Number of stacked LSTM layers.
        feature_dim (int): Output feature vector dimension.
        dropout (float): Dropout probability between LSTM layers.
        bidirectional (bool): Use bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 256,
        num_layers: int = 3,
        feature_dim: int = 512,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(SensorLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism over time steps
        self.attention = TemporalAttention(hidden_size * self.num_directions)

        # Feature projection
        lstm_out_dim = hidden_size * self.num_directions
        self.projection = nn.Sequential(
            nn.Linear(lstm_out_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Time-series tensor of shape (batch, seq_len, input_size)
            lengths: Actual sequence lengths for packed padding (optional)
        
        Returns:
            features: Feature vector (batch, feature_dim)
            attention_weights: Attention map (batch, seq_len)
        """
        B, T, _ = x.shape

        # Normalize input
        x = self.input_norm(x)

        # Pack padded sequences for efficiency
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        lstm_out, (h_n, _) = self.lstm(x)

        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply temporal attention
        context, attention_weights = self.attention(lstm_out)

        # Project to feature space
        features = self.projection(context)

        return features, attention_weights


class TemporalAttention(nn.Module):
    """
    Soft attention mechanism over LSTM timesteps.
    
    Learns to weight which time steps are most relevant
    for disaster prediction.
    """

    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            context: Weighted context vector (batch, hidden_dim)
            weights: Attention weights (batch, seq_len)
        """
        scores = self.attention(lstm_output).squeeze(-1)   # (B, T)
        weights = torch.softmax(scores, dim=-1)             # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (B, H)
        return context, weights


if __name__ == "__main__":
    # Quick test
    model = SensorLSTM(input_size=32, hidden_size=256, num_layers=3, feature_dim=512)
    dummy_input = torch.randn(4, 72, 32)  # batch=4, 72 timesteps, 32 features
    features, attn = model(dummy_input)
    print(f"LSTM Feature shape: {features.shape}")   # Expected: (4, 512)
    print(f"Attention shape:    {attn.shape}")        # Expected: (4, 72)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
