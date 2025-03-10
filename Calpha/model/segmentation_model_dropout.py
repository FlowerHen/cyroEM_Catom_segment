import torch
import torch.nn as nn
from .layers import (
    ConvBlock, Bottleneck, Res2NetBlock,
    DownsampleLayer, MainLayer, UpsampleAdd, UpsampleCat, MultiScaleConv, AttentionGate
)

class SegmentationModel(nn.Module):
    """Algorithm 1: ModelAngelo model"""
    def __init__(self, use_sa=False, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Downsampling layers
        self.down_conv = nn.Sequential(
            ConvBlock(1, 64, kernel_size=5, padding=2),
            DownsampleLayer(64, 64),
            DownsampleLayer(64, 128),
            DownsampleLayer(128, 256),
            DownsampleLayer(256, 512)
        )
        
        # Intermediate processing layers
        self.tl4 = ConvBlock(512, 256)  # 512 -> 256
        self.ll4 = ConvBlock(256, 256)  # Keep 256
        self.c4 = MainLayer(256, num_layers=2, expansion=4, block_type='bottleneck', use_sa=use_sa)
        
        self.tl3 = ConvBlock(256, 128)  # 256 -> 128
        self.ll3 = ConvBlock(128, 128)  # Keep 128
        self.c3 = MainLayer(128, num_layers=10, expansion=4, block_type='bottleneck', use_sa=use_sa)
        
        self.tl2 = ConvBlock(128, 64)   # 128 -> 64
        self.ll2 = ConvBlock(64, 64)    # Keep 64
        self.c2 = MainLayer(64, num_layers=20, expansion=4, block_type='bottleneck')
        
        self.tl1 = ConvBlock(64, 64)    # Keep 64
        self.ll1 = ConvBlock(64, 64)    # Keep 64
        self.c1 = MainLayer(64, num_layers=5, expansion=4, block_type='bottleneck')
        
        # Upsampling and feature fusion
        self.upsample_add = UpsampleAdd()
        self.final_conv = MultiScaleConv(64, 1, 1)  # Output channels adjusted to 1
        
    def forward(self, V):
        # Normalization
        V = (V - V.mean()) / (V.std() + 1e-6)
        
        # Downsampling
        ds0 = self.down_conv[0](V)
        ds1 = self.down_conv[1](ds0)
        ds2 = self.down_conv[2](ds1)
        ds3 = self.down_conv[3](ds2)
        ds4 = self.down_conv[4](ds3)
        
        # Apply dropout at bottleneck (highest-level features)
        ds4 = self.dropout(ds4)
        
        # Decoder path
        tl4 = self.tl4(ds4)
        c4 = self.c4(self.upsample_add(tl4, self.ll4(ds3)))
        c4 = self.dropout(c4)  # Apply dropout after deep feature processing
        
        tl3 = self.tl3(c4)
        c3 = self.c3(self.upsample_add(tl3, self.ll3(ds2)))
        c3 = self.dropout(c3)  # Apply dropout
        
        tl2 = self.tl2(c3)
        c2 = self.c2(self.upsample_add(tl2, self.ll2(ds1)))
        
        tl1 = self.tl1(c2)
        c1 = self.c1(self.upsample_add(tl1, self.ll1(ds0)))
        
        # Final output
        return self.final_conv(c1)

class SegmentationModelResnet(nn.Module):
    """Algorithm 2: Res2Net-based U-Net variant by Cryfold V1.2"""
    def __init__(self, use_sa=False, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Normalization
        self.norm = nn.InstanceNorm3d(1, affine=True)
        
        # Initial convolution
        self.init_conv = ConvBlock(1, 256, kernel_size=3, padding=1)
        
        # Downsampling
        self.down_layers = nn.ModuleList([
            Bottleneck(256, 256, stride=2),
            Bottleneck(256, 256, stride=2),
            Bottleneck(256, 256, stride=2),
            Bottleneck(256, 256, stride=2)
        ])
        
        # Intermediate layers
        self.mid_layers = nn.ModuleList([
            MainLayer(256, num_layers=2, expansion=4, block_type='res2net', use_sa=use_sa),
            MainLayer(256, num_layers=3, expansion=4, block_type='res2net', use_sa=use_sa),
            MainLayer(128, num_layers=4, expansion=4, block_type='res2net', use_sa=use_sa),
            MainLayer(64, num_layers=4, expansion=4, block_type='res2net'),
            MainLayer(64, num_layers=4, expansion=4, block_type='res2net')
        ])
        
        # Upsampling path
        self.upsample_cat = UpsampleCat()
        self.up_convs = nn.ModuleList([
            ConvBlock(512, 256),  # 256 + 256 = 512
            ConvBlock(512, 128),  # 256 + 256 = 512
            ConvBlock(384, 64),   # 128 + 256 = 384
            ConvBlock(320, 64)    # 64 + 256 = 320
        ])
        
        # Final output
        self.final_conv = MultiScaleConv(64, 1, 1)  # Output channels adjusted to 1
        
    def forward(self, x):
        # Normalization
        x = self.norm(x)
        
        # Encoder
        ds0 = self.init_conv(x)
        ds1 = self.down_layers[0](ds0)
        ds2 = self.down_layers[1](ds1)
        ds3 = self.down_layers[2](ds2)
        ds4 = self.down_layers[3](ds3)
        
        # Apply dropout at bottleneck
        ds4 = self.dropout(ds4)
        
        # Intermediate processing
        c4 = self.mid_layers[0](ds4)
        
        # Decoder
        c3 = self._upsample_block(c4, ds3, 0)
        c3 = self.mid_layers[1](c3)
        c3 = self.dropout(c3)  # Apply dropout after feature fusion and processing
        
        c2 = self._upsample_block(c3, ds2, 1)
        c2 = self.mid_layers[2](c2)
        c2 = self.dropout(c2)  # Apply dropout
        
        c1 = self._upsample_block(c2, ds1, 2)
        c1 = self.mid_layers[3](c1)
        
        c0 = self._upsample_block(c1, ds0, 3)
        c0 = self.mid_layers[4](c0)
        
        return self.final_conv(c0)
    
    def _upsample_block(self, x, skip, layer_idx):
        x = self.upsample_cat(x, skip)
        # Apply dropout after feature concatenation
        if layer_idx < 2:  # Only apply to deeper layers
            x = self.dropout(x)
        return self.up_convs[layer_idx](x)


class SegmentationModelAttn(SegmentationModelResnet):
    """Algorithm 3: U-Net variant with attention mechanism by CryFold V1.3"""
    def __init__(self, use_sa=True, dropout_rate=0.2):
        super().__init__(use_sa, dropout_rate)
        # Replace upsampling module with attention gates
        self.attn_gates = nn.ModuleList([
            AttentionGate(256, 256, 256),
            AttentionGate(256, 256, 128),
            AttentionGate(256, 128, 64),
            AttentionGate(256, 64, 64)
        ])
        
        # Adjust upsampling path for attention gates
        self.up_convs = nn.ModuleList([
            ConvBlock(256 + 256, 256),  # 256 (attn) + 256 (skip) = 512
            ConvBlock(128 + 256, 128),  # 128 (attn) + 256 (skip) = 384
            ConvBlock(64 + 128, 64),    # 64 (attn) + 128 (skip) = 192
            ConvBlock(64 + 64, 64)      # 64 (attn) + 64 (skip) = 128
        ])
        
    def _upsample_block(self, x, skip, layer_idx):
        attn = self.attn_gates[layer_idx](x, skip)
        # Apply dropout to attended features for deeper layers
        if layer_idx < 2:
            attn = self.dropout(attn)
            
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, attn], dim=1)
        
        # Apply dropout after feature concatenation for deeper layers
        if layer_idx < 2:
            x = self.dropout(x)
            
        return self.up_convs[layer_idx](x)

class SegmentationModelMini(nn.Module):
    def __init__(self, use_sa=False, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.norm = nn.InstanceNorm3d(1, affine=True)
        self.conv0 = ConvBlock(1, 32, kernel_size=3, padding=1)
        self.down1 = DownsampleLayer(32, 64)
        self.down2 = DownsampleLayer(64, 128)
        self.mid = MainLayer(128, num_layers=1, expansion=4, block_type='bottleneck', use_sa=use_sa)
        self.proj_mid = ConvBlock(128, 64, kernel_size=1, padding=0)
        self.skip1 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.skip0 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.upsample_add = UpsampleAdd()
        self.up_conv1 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.up_conv2 = ConvBlock(64, 32, kernel_size=3, padding=1)
        self.final_conv = MultiScaleConv(32, 1, 1)

    def forward(self, x):
        x = self.norm(x)
        ds0 = self.conv0(x)
        ds1 = self.down1(ds0)
        ds2 = self.down2(ds1)
        
        # Apply dropout at bottleneck
        ds2 = self.dropout(ds2)
        
        mid = self.mid(ds2)
        mid_proj = self.proj_mid(mid)
        
        # Apply dropout after bottleneck processing
        mid_proj = self.dropout(mid_proj)
        
        up1 = self.upsample_add(mid_proj, self.skip1(ds1))
        up1 = self.up_conv1(up1)
        
        up2 = self.upsample_add(up1, self.skip0(ds0))
        up2 = self.up_conv2(up2)
        
        return self.final_conv(up2)

class SegmentationModelResMini(nn.Module):
    def __init__(self, use_sa=False, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.norm = nn.InstanceNorm3d(1, affine=True)
        self.init_conv = ConvBlock(1, 64, kernel_size=3, padding=1)
        self.down_layers = nn.ModuleList([
            Bottleneck(64, 64, stride=2),
            Bottleneck(64, 64, stride=2)
        ])
        self.mid = MainLayer(64, num_layers=1, expansion=4, block_type='res2net', use_sa=use_sa)
        self.upsample_cat = UpsampleCat()
        self.up_convs = nn.ModuleList([
            ConvBlock(64 + 64, 64, kernel_size=3, padding=1),
            ConvBlock(64 + 64, 64, kernel_size=3, padding=1)
        ])
        self.final_conv = MultiScaleConv(64, 1, 1)

    def forward(self, x):
        x = self.norm(x)
        ds0 = self.init_conv(x)
        ds1 = self.down_layers[0](ds0)
        ds2 = self.down_layers[1](ds1)
        
        # Apply dropout at bottleneck
        ds2 = self.dropout(ds2)
        
        mid = self.mid(ds2)
        
        # Apply dropout after bottleneck processing
        mid = self.dropout(mid)
        
        up1 = self._upsample_block(mid, ds1, 0)
        up2 = self._upsample_block(up1, ds0, 1)
        
        return self.final_conv(up2)

    def _upsample_block(self, x, skip, layer_idx):
        x = self.upsample_cat(x, skip)
        
        # Apply dropout after concatenation for the first upsampling layer
        if layer_idx == 0:
            x = self.dropout(x)
            
        return self.up_convs[layer_idx](x)

class SegmentationAttnMini(SegmentationModelResMini):
    """SegmentationModelResMini with attention mechanism"""
    def __init__(self, use_sa=False, dropout_rate=0.2):
        super().__init__(use_sa=use_sa, dropout_rate=dropout_rate)
        
        # Attention gates for the upsampling path
        self.attn_gates = nn.ModuleList([
            AttentionGate(64, 64, 64),
            AttentionGate(64, 64, 64)
        ])
        
        # Adjust upsampling path for attention gates
        self.up_convs = nn.ModuleList([
            ConvBlock(64 + 64, 64, kernel_size=3, padding=1),  # 64 (x) + 64 (attn)
            ConvBlock(64 + 64, 64, kernel_size=3, padding=1)
        ])

    def _upsample_block(self, x, skip, layer_idx):
        attn = self.attn_gates[layer_idx](x, skip)
        
        # Apply dropout to attention features in the deeper layers
        if layer_idx == 0:
            attn = self.dropout(attn)
            
        x_upsampled = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x_concat = torch.cat([x_upsampled, attn], dim=1)
        
        # Apply dropout after concatenation for deeper layers
        if layer_idx == 0:
            x_concat = self.dropout(x_concat)
            
        return self.up_convs[layer_idx](x_concat)
