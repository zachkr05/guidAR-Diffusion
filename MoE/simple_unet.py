import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings
from .attention import AttentionBlock


class TimeAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SimpleTrajectoryUNet(nn.Module):
    """
    Direct conditional UNet: concatenate all per-class features + noisy
    trajectory as input channels.

    No encoder, no tokenizer, no cross-attention.
    Just prove the trajectory target works.
    """

    def __init__(self, in_channels, base_channels=64, time_dim=256):
        super().__init__()
        bc = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(bc),
            nn.Linear(bc, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = TimeAwareBlock(in_channels, bc, time_dim)
        self.enc2 = TimeAwareBlock(bc, bc * 2, time_dim)
        self.enc3 = TimeAwareBlock(bc * 2, bc * 4, time_dim)

        # Bottleneck
        self.mid1 = TimeAwareBlock(bc * 4, bc * 4, time_dim)
        self.mid_attn = AttentionBlock(bc * 4, num_heads=4)
        self.mid2 = TimeAwareBlock(bc * 4, bc * 4, time_dim)

        # Decoder (with skip connections, so input channels double)
        self.dec3 = TimeAwareBlock(bc * 8, bc * 2, time_dim)
        self.dec2 = TimeAwareBlock(bc * 4, bc, time_dim)
        self.dec1 = TimeAwareBlock(bc * 2, bc, time_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            nn.Conv2d(bc, 1, kernel_size=1),
        )

    def forward(self, x, t, conditioning=None):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)                          # 128
        e2 = self.enc2(F.max_pool2d(e1, 2), t_emb)        # 64
        e3 = self.enc3(F.max_pool2d(e2, 2), t_emb)        # 32

        # Bottleneck
        m = self.mid1(F.max_pool2d(e3, 2), t_emb)         # 16
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        # Decoder
        d3 = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)  # 32

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)  # 64

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)  # 128

        return self.final(d1)
