# ModelAngelo: Algorithm 3 Bottleneck
# Adapted from https://github.com/3dem/model-angelo/blob/main/model_angelo/models/bottleneck.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .conv_block import SimamModule

class Bottleneck(nn.Module):
    """
    3D Bottleneck block with optional checkpointing
    Architecture: Conv1x1 -> ReLU -> Conv3x3 -> ReLU -> Conv1x1 -> Residual
    Expansion: 4x
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, 
                 activation=nn.ReLU, norm_layer=nn.InstanceNorm3d, affine=True, 
                 checkpointing=True, use_sa = False,**kwargs):
        super().__init__()
        self.checkpointing = checkpointing
        hidden_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, 1, bias=False, groups=groups)
        self.norm1 = norm_layer(hidden_channels, affine=affine)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, 3, stride, 1, 
                              bias=False, groups=groups)
        self.norm2 = norm_layer(hidden_channels, affine=affine)
        self.conv3 = nn.Conv3d(hidden_channels, out_channels, 1, bias=False, groups=groups)
        self.norm3 = norm_layer(out_channels, affine=affine)
        
        self.se = SimamModule(out_channels) if use_sa else nn.Identity()
        self.activation = activation(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride,
                    bias=False, 
                    groups=groups
                ),
                norm_layer(out_channels, affine=affine)
            )

    def _forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)
        
        out += identity
        return self.activation(out)

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False) if self.checkpointing else self._forward(x)