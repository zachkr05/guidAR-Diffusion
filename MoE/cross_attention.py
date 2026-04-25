import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Q = bottleneck spatial features (flattened)
    K, V = obstacle tokens
    LayerNorm -> attention + residual -> FFN + residual
    """

    def __init__(self, query_dim, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.SiLU(),
            nn.Linear(query_dim * 4, query_dim),
        )

    def forward(self, x, context):
        """
        x:       (B, N_q, D)   bottleneck pixels flattened
        context: (B, N_kv, D)  obstacle tokens
        returns: (B, N_q, D)
        """
        # Cross-attention + residual
        h = self.norm_q(x)
        ctx = self.norm_kv(context)

        B, N_q, D = h.shape
        N_kv = ctx.shape[1]

        q = self.to_q(h).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.proj(out)
        x = x + out

        # FFN + residual
        x = x + self.ffn(self.norm_ffn(x))
        return x
