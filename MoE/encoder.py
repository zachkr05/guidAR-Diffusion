import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(1, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ObstacleEncoder(nn.Module):
    """Per-class encoder: (B, 9, 128, 128) -> (B, feat_dim, 32, 32)"""

    def __init__(self, in_channels=9, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),       # 128 -> 64
            ConvBlock(32, 64, stride=2),                 # 64  -> 32
            ConvBlock(64, feat_dim, stride=1),           # 32  -> 32, channels -> feat_dim
        )

    def forward(self, x):
        return self.net(x)


class ObstacleTokenizer(nn.Module):
    """
    Takes a list of per-class feature maps, flattens spatial dims,
    adds positional + class embeddings, and concatenates across classes.

    Input:  list of (B, D, 32, 32) tensors, one per class
    Output: (B, num_classes * 1024, D) obstacle tokens
    """

    def __init__(self, feat_dim=128, num_classes=3, spatial_size=32):
        super().__init__()
        num_positions = spatial_size * spatial_size  # 1024
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, feat_dim) * 0.02)
        self.class_embed = nn.Parameter(torch.randn(1, num_classes, feat_dim) * 0.02)

    def forward(self, feature_maps):
        tokens_list = []
        for i, feat in enumerate(feature_maps):
            B, D, H, W = feat.shape
            t = feat.flatten(2).transpose(1, 2)             # (B, H*W, D)
            t = t + self.pos_embed[:, : H * W, :]           # positional
            t = t + self.class_embed[:, i : i + 1, :]       # class (broadcast over spatial)
            tokens_list.append(t)
        return torch.cat(tokens_list, dim=1)                 # (B, N_total, D)
