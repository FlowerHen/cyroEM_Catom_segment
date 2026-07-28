from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet3D(nn.Module):
    """Compact 3D U-Net that returns uncalibrated heatmap logits."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        channels = [base_channels * 2**level for level in range(depth)]
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for output_channels in channels:
            self.encoders.append(ConvBlock(current_channels, output_channels, dropout))
            current_channels = output_channels
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = channels[-1] * 2
        for skip_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose3d(current_channels, skip_channels, 2, stride=2))
            self.decoders.append(ConvBlock(skip_channels * 2, skip_channels, dropout))
            current_channels = skip_channels
        self.output = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("UNet3D inputs must have shape [B, C, D, H, W]")
        skips: list[torch.Tensor] = []
        features = inputs
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            features = upconv(features)
            if features.shape[2:] != skip.shape[2:]:
                features = F.interpolate(
                    features, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            features = decoder(torch.cat((features, skip), dim=1))
        return self.output(features)
