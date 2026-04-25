import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings
from .attention import AttentionBlock
from .cross_attention import CrossAttentionBlock
from .LoRA import LoRA
from .FiLM import FiLMLayer


class TimeAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return h


class TrajectoryUNet(nn.Module):
    """
    Denoiser for trajectory heatmaps.

    Input:  noisy heatmap (B, 1, 128, 128) + timestep + obstacle tokens (B, N, D)
    Output: predicted noise (B, 1, 128, 128)

    Spatial flow:
        enc1 (stride 2):  128² → 64²   [skip]
        enc2 (stride 2):   64² → 32²   [skip]
        pool + center:     32² → 16²
        self-attention at 16²
        cross-attention at 16²  (Q=bottleneck, K/V=obstacle tokens)
        dec2:  up 16² → 32²  + enc2 skip,  LoRA + FiLM
        dec1:  up 32² → 64²  + enc1 skip,  LoRA + FiLM
        up 64² → 128² → 1×1 conv
    """

    def __init__(
        self,
        base_channels=32,
        time_dim=128,
        context_dim=128,
        lora_rank=4,
        lora_scale=1.0,
    ):
        super().__init__()
        bc = base_channels

        # --- time embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(bc),
            nn.Linear(bc, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- encoder (noisy trajectory only, 1 channel) ---
        self.enc1 = TimeAwareBlock(2, bc, time_dim, stride=2)        # 128 → 64
        self.enc2 = TimeAwareBlock(bc, bc * 2, time_dim, stride=2)   #  64 → 32

        # --- center at 16×16 ---
        self.center = TimeAwareBlock(bc * 2, bc * 4, time_dim)       #  16 → 16
        self.mid_attn = AttentionBlock(bc * 4, num_heads=4)

        # --- cross-attention at bottleneck ---
        self.cross_attn = CrossAttentionBlock(
            query_dim=bc * 4,
            context_dim=context_dim,
            num_heads=4,
        )

        # --- decoder ---
        # up 16→32, cat enc2 skip (bc*2): in = bc*4 + bc*2
        self.dec2 = TimeAwareBlock(bc * 4 + bc * 2, bc * 2, time_dim)
        self.lora_dec2 = LoRA(bc * 2, rank=lora_rank, scale=lora_scale)
        self.film_dec2 = FiLMLayer(bc * 2, cond_dim=context_dim)

        # up 32→64, cat enc1 skip (bc): in = bc*2 + bc
        self.dec1 = TimeAwareBlock(bc * 2 + bc, bc, time_dim)
        self.lora_dec1 = LoRA(bc, rank=lora_rank, scale=lora_scale)
        self.film_dec1 = FiLMLayer(bc, cond_dim=context_dim)

        # up 64→128, 1×1 conv
        self.final = nn.Conv2d(bc, 1, kernel_size=1)

    def forward(self, x_t, t, obstacle_tokens):
        """
        x_t:             (B, 1, H, W)  noisy trajectory heatmap
        t:               (B,)          diffusion timestep
        obstacle_tokens: (B, N, D)     from ObstacleTokenizer
        """
        t_emb = self.time_mlp(t)
        token_summary = obstacle_tokens.mean(dim=1)  # (B, D) for FiLM

        # ---- encoder ----
        e1 = self.enc1(x_t, t_emb)                             # (B, bc,   64, 64)
        e2 = self.enc2(e1, t_emb)                               # (B, bc*2, 32, 32)

        # ---- center ----
        c = self.center(F.max_pool2d(e2, 2), t_emb)            # (B, bc*4, 16, 16)
        c = self.mid_attn(c)

        # ---- cross-attention ----
        B, C, H, W = c.shape
        c_flat = c.flatten(2).transpose(1, 2)                   # (B, 256, bc*4)
        c_flat = self.cross_attn(c_flat, obstacle_tokens)
        c = c_flat.transpose(1, 2).view(B, C, H, W)            # (B, bc*4, 16, 16)

        # ---- decoder ----
        c_up = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([c_up, e2], dim=1), t_emb)    # (B, bc*2, 32, 32)
        d2 = self.lora_dec2(d2)
        d2 = self.film_dec2(d2, token_summary)

        d2_up = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1), t_emb)   # (B, bc, 64, 64)
        d1 = self.lora_dec1(d1)
        d1 = self.film_dec1(d1, token_summary)

        d1_up = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        return self.final(d1_up)                                 # (B, 1, 128, 128)

    def set_finetune(self, active=True):
        """Freeze everything except LoRA and FiLM for IRL finetuning."""
        for p in self.parameters():
            p.requires_grad = not active
        if active:
            for module in [
                self.lora_dec2, self.lora_dec1,
                self.film_dec2, self.film_dec1,
            ]:
                for p in module.parameters():
                    p.requires_grad = True
