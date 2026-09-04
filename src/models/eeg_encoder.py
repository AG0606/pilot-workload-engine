import torch
import torch.nn as nn


class EEGNetEncoder(nn.Module):
    """Spatial-temporal convolutional encoder for multi-channel EEG time series."""

    def __init__(
        self,
        num_channels: int = 17,
        temporal_samples: int = 80,
        f1_filters: int = 8,
        depth_multiplier: int = 2,
        embed_dim: int = 128,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.temporal_samples = temporal_samples
        f2_filters = f1_filters * depth_multiplier

        # Block 1: Temporal convolution across time samples
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=f1_filters,
            kernel_size=(1, 15),
            padding=(0, 7),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1_filters)

        # Depthwise spatial convolution across electrode leads
        self.spatial_conv = nn.Conv2d(
            in_channels=f1_filters,
            out_channels=f2_filters,
            kernel_size=(num_channels, 1),
            groups=f1_filters,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f2_filters)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution (depthwise + pointwise)
        self.depthwise_conv = nn.Conv2d(
            in_channels=f2_filters,
            out_channels=f2_filters,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=f2_filters,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=f2_filters,
            out_channels=f2_filters,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(f2_filters)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute output feature map dimensions
        test_input = torch.zeros(1, 1, num_channels, temporal_samples)
        with torch.no_grad():
            out = self.temporal_conv(test_input)
            out = self.bn1(out)
            out = self.spatial_conv(out)
            out = self.bn2(out)
            out = self.act1(out)
            out = self.pool1(out)
            out = self.depthwise_conv(out)
            out = self.pointwise_conv(out)
            out = self.bn3(out)
            out = self.act2(out)
            out = self.pool2(out)
            flattened_dim = out.view(1, -1).size(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting input shape [B, num_channels, temporal_samples]."""
        if x.dim() == 3:
            # Reshape [B, channels, time] -> [B, 1, channels, time]
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        embed = self.projection(x)
        return embed
