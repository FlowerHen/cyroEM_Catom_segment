# ModelAngelo: Algorithm 6 Main Layer
import torch
import torch.nn as nn
from .bottleneck import Bottleneck
from .res2net import Res2NetBlock

class MainLayer(nn.Module):
    """
    Stack of repeated building blocks
    """
    def __init__(self, channels, num_layers, 
                 block_type='bottleneck', expansion=4, use_sa=False):
        super().__init__()
        block_class = {
            'bottleneck': Bottleneck,
            'res2net': Res2NetBlock
        }[block_type]
        
        layers = []
        for _ in range(num_layers):
            layers.append(block_class(
                in_channels=channels,
                out_channels=channels,
                expansion=expansion,
                use_sa=use_sa
            ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
