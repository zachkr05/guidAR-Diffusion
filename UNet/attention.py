
import math
import torch
import toch.nn as nn
import torch.nn.functional as F
from time_emb import SinusoidalEmbeddings


class AttentionBlock(nn.Module):
    """Self-attention for spatial feature maps"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
       

        #dot of q and k
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        #attn dotted with value tensor
        h = torch.einsum('bhij,bhcj->bhci', attn, v)
        h = h.view(B, C, H, W) #reshape it
        h = self.proj(h) #translate bcus of the  
        
        return x + h


