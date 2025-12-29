

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """Conv block with residual connection and time conditioning"""
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        
        return h + self.residual(x)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False, num_heads=4):
        super().__init__()
        
        self.conv1 = ResidualConvBlock(in_channels, out_channels, time_dim)
        self.conv2 = ResidualConvBlock(out_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads)
    
    def forward(self, x, t_emb):
        h = self.conv1(x, t_emb)
        h = self.conv2(h, t_emb)
        
        if self.use_attention:
            h = self.attn(h)
        
        return self.downsample(h), h


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False, num_heads=4):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = ResidualConvBlock(in_channels + out_channels, out_channels, time_dim)
        self.conv2 = ResidualConvBlock(out_channels, out_channels, time_dim)
        
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionBlock(out_channels, num_heads)
    
    def forward(self, x, skip, t_emb):
        h = self.upsample(x)
        
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        h = torch.cat([h, skip], dim=1)
        h = self.conv1(h, t_emb)
        h = self.conv2(h, t_emb)
        
        if self.use_attention:
            h = self.attn(h)
        
        return h


