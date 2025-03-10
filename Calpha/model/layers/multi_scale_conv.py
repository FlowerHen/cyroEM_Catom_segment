# ModelAngelo: Algorithm 8 Multi Scale Convolution
# Add residual connection as a option
import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    """
    Multi-scale convolution with residual connections
    Architecture: Parallel conv3x3,5x5,7x7 -> concat -> conv1x1
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, mid_channels, k, padding=k//2)
            for k in [3,5,7]
        ])
        self.norm = nn.InstanceNorm3d(mid_channels*3)
        self.final_conv = nn.Conv3d(mid_channels*3, out_channels, 3, padding=1)
        
    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.norm(out)
        out = nn.functional.relu(out, inplace=True)
        return self.final_conv(out)
