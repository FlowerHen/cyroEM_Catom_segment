# ModelAngelo: Algorithm 4 Downsample Layer. Factor-two downsampling with strided convolution
import torch
import torch.nn as nn

class DownsampleLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.InstanceNorm3d(output_channels, affine=True) # output = a × normalized_input + b for each channel
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
