


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        if channels % num_heads != 0:
            raise ValueError(f"Channels {channels} must be divisible by heads {num_heads}")

        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(1, channels)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2,3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2,3)

        attn = torch.matmul(q,k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn,v)

        out = out.transpose(2,3).contiguous().view(B,C,H,W)
        out=self.proj(out)

        return x + out
