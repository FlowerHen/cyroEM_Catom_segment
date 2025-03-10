# Adapted from https://github.com/SBQ-1999/CryFold/blob/1ccf55a628440ef364879b02b6aeb23f6864cd22/CryFold/Unet/Unet.py#L99
import torch
import torch.nn as nn
from .conv_block import SimamModule

class Res2NetBlock(nn.Module):
    """
    Res2Net block with multi-scale convolutional groups
    Architecture: Split -> [Conv3x3 groups] -> Concat -> Conv1x1
    """
    def __init__(self, in_channels, out_channels, stride=1, scale=4, 
                 activation=nn.ReLU, norm_layer=nn.InstanceNorm3d, affine=True,
                 use_sa=False, **kwargs):
        super().__init__()
        self.scale = scale
        hidden_channels = out_channels // scale
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, affine=affine)
        )
        self.convs = nn.ModuleList([
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride, 1, bias=False)
            for _ in range(scale-1)
        ])
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, affine=affine)
        )
        
        self.se = SimamModule(out_channels) if use_sa else nn.Identity()
        self.activation = activation(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                norm_layer(out_channels, affine=affine)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.activation(self.conv1(x))
        split = torch.chunk(out, self.scale, 1)
        
        feats = [split[0]]
        for i in range(1, self.scale):
            feats.append(self.convs[i-1](split[i] + feats[-1]))
        
        out = torch.cat(feats, dim=1)
        out = self.activation(self.conv2(out))
        out = self.se(out)
        
        return self.activation(out + identity)
