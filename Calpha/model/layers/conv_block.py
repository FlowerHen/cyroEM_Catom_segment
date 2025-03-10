# ModelAngelo: Algorithm 2 Convolutional Building Block
# Add convlution attention module
import torch
import torch.nn as nn

# SIMAM module for 3D, adapted from https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py
class SimamModule(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimamModule, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, d, h, w = x.size()
        n = d * h * w - 1
        x_mean = x.mean(dim=[2,3,4], keepdim=True)
        x_minus_mu_square = (x - x_mean).pow(2)
        y = x_minus_mu_square / (4*(x_minus_mu_square.sum(dim=[2,3,4], keepdim=True)/n + self.e_lambda)) + 0.5
        return x * self.activation(y)

# Revised ConvBlock. The use_sa flag toggles SIMAM attention.
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, use_sa=False):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_c, affine=True)
        self.act = nn.ReLU(inplace=True)
        # Use SIMAM if use_sa is True; otherwise, simply pass through.
        self.attention = SimamModule(e_lambda=1e-4) if use_sa else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.attention(x)
        return self.act(x)
