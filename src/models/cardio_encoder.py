import torch
import torch.nn as nn


class CardioResidualBlock1D(nn.Module):
    """1D residual block with strided convolution and shortcut projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out


class CardioEncoder(nn.Module):
    """1D residual convolutional encoder for cardio and autonomic waveform channels."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_filters: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
        )

        self.block1 = CardioResidualBlock1D(base_filters, base_filters * 2, stride=2)
        self.block2 = CardioResidualBlock1D(base_filters * 2, base_filters * 4, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting input shape [B, 3, temporal_samples]."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        embed = self.projection(x)
        return embed
