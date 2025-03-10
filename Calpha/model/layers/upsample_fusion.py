# ModelAngelo: Algorithm 7 Upsample then add
import torch
import torch.nn as nn

class UpsampleAdd(nn.Module):
    """Feature fusion with element-wise addition"""
    def __init__(self, learnable=False):
        super().__init__()
        self.learnable = learnable
        if self.learnable:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, skip):
        x_ups = nn.functional.interpolate(
            x, size=skip.shape[2:], mode='trilinear', align_corners=True
        )
        if self.learnable:
            return self.alpha * x_ups + self.beta * skip
        else:
            return x_ups + skip

class UpsampleCat(nn.Module):
    """Feature fusion with channel-wise concatenation"""
    def __init__(self):
        super().__init__()

    def forward(self, x, skip):
        x_ups = nn.functional.interpolate(
            x, size=skip.shape[2:], mode='trilinear', align_corners=True
        )
        return torch.cat([x_ups, skip], dim=1)