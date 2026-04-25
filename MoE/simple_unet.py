import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings
from .attention import AttentionBlock
from .LoRA import LoRA
from .FiLM import FiLMLayer


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
    Direct conditional UNet with LoRA after every encoder/decoder block
    and FiLM conditioning on every decoder block.

    During base training: train everything end-to-end.
    During IRL finetuning: freeze encoder + bottleneck, only train LoRA + FiLM.
    """

    def __init__(self, in_channels, base_channels=64, time_dim=256,
                 film_cond_dim=128, lora_rank=4, lora_scale=1.0):
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
        self.lora_enc1 = LoRA(bc, rank=lora_rank, scale=lora_scale)

        self.enc2 = TimeAwareBlock(bc, bc * 2, time_dim)
        self.lora_enc2 = LoRA(bc * 2, rank=lora_rank, scale=lora_scale)

        self.enc3 = TimeAwareBlock(bc * 2, bc * 4, time_dim)
        self.lora_enc3 = LoRA(bc * 4, rank=lora_rank, scale=lora_scale)

        # Bottleneck
        self.mid1 = TimeAwareBlock(bc * 4, bc * 4, time_dim)
        self.mid_attn = AttentionBlock(bc * 4, num_heads=4)
        self.mid2 = TimeAwareBlock(bc * 4, bc * 4, time_dim)

        # Decoder (skip connections double input channels)
        self.dec3 = TimeAwareBlock(bc * 8, bc * 2, time_dim)
        self.lora_dec3 = LoRA(bc * 2, rank=lora_rank, scale=lora_scale)
        self.film_dec3 = FiLMLayer(bc * 2, cond_dim=film_cond_dim)

        self.dec2 = TimeAwareBlock(bc * 4, bc, time_dim)
        self.lora_dec2 = LoRA(bc, rank=lora_rank, scale=lora_scale)
        self.film_dec2 = FiLMLayer(bc, cond_dim=film_cond_dim)

        self.dec1 = TimeAwareBlock(bc * 2, bc, time_dim)
        self.lora_dec1 = LoRA(bc, rank=lora_rank, scale=lora_scale)
        self.film_dec1 = FiLMLayer(bc, cond_dim=film_cond_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            nn.Conv2d(bc, 1, kernel_size=1),
        )

    def forward(self, x, t, conditioning=None, film_cond=None):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e1 = self.lora_enc1(e1)

        e2 = self.enc2(F.max_pool2d(e1, 2), t_emb)
        e2 = self.lora_enc2(e2)

        e3 = self.enc3(F.max_pool2d(e2, 2), t_emb)
        e3 = self.lora_enc3(e3)

        # Bottleneck
        m = self.mid1(F.max_pool2d(e3, 2), t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)

        # Decoder
        d3 = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        d3 = self.lora_dec3(d3)
        if film_cond is not None:
            d3 = self.film_dec3(d3, film_cond)

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        d2 = self.lora_dec2(d2)
        if film_cond is not None:
            d2 = self.film_dec2(d2, film_cond)

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        d1 = self.lora_dec1(d1)
        if film_cond is not None:
            d1 = self.film_dec1(d1, film_cond)

        return self.final(d1)

    def set_finetune(self, active=True):
        """Freeze everything except LoRA and FiLM for IRL finetuning."""
        for p in self.parameters():
            p.requires_grad = not active

        if active:
            # Unfreeze all LoRA (encoder + decoder)
            for module in [
                self.lora_enc1, self.lora_enc2, self.lora_enc3,
                self.lora_dec3, self.lora_dec2, self.lora_dec1,
            ]:
                for p in module.parameters():
                    p.requires_grad = True

            # Unfreeze all FiLM (decoder only)
            for module in [
                self.film_dec3, self.film_dec2, self.film_dec1,
            ]:
                for p in module.parameters():
                    p.requires_grad = True
