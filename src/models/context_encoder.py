import torch
import torch.nn as nn


class FlightContextEncoder(nn.Module):
    """Kinematics and flight dynamics encoder capturing aircraft acceleration and turbulence."""

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting input shape [B, 8, temporal_samples]."""
        return self.net(x)
