import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .conv_block import ConvBlock

class AttentionGate(nn.Module):
    def __init__(self, down_features: int, up_features: int, out_features: int,
                 attention_features: int = 64, attention_heads: int = 8):
        super(AttentionGate, self).__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads

        self.conv_q = nn.Sequential(
            nn.Conv3d(
                in_channels=self.ufz,
                out_channels=self.afz,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(self.afz, affine=True)
        )
        self.conv_k = nn.Sequential(
            nn.Conv3d(
                in_channels=self.dfz,
                out_channels=self.afz,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(self.afz, affine=True)
        )
        self.conv_v = nn.Sequential(
            nn.Conv3d(
                in_channels=self.dfz,
                out_channels=self.ufz,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(self.ufz, affine=True)
        )
        # Gate module generates multiple attention heads, which are averaged
        self.gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=self.afz,
                out_channels=self.ahz,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_back = ConvBlock(self.ufz, self.ofz, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip):
        # Interpolate x to match the size of skip
        x_upsampled = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        
        # Use gradient checkpointing for memory efficiency
        def checkpointed_forward(inputs):
            x_upsampled, skip = inputs
            q = self.conv_q(x_upsampled)      # shape: (B, afz, D, H, W)
            k = self.conv_k(skip)             # shape: (B, afz, D, H, W)
            s = self.relu(q + k)
            psi = self.gate(s)                # shape: (B, ahz, D, H, W)
            psi_avg = psi.mean(dim=1, keepdim=True)  # shape: (B, 1, D, H, W)
            v = self.conv_v(skip)             # shape: (B, up_features, D, H, W)
            attn_feature = self.conv_back(v * psi_avg)
            return attn_feature
        
        return checkpoint(checkpointed_forward, (x_upsampled, skip), use_reentrant=False)
