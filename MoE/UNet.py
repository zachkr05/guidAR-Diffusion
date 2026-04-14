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
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

class LightweightUNet(nn.Module):
    def __init__(self, in_channels, context_channels, out_channels=1, base_channels=32, time_dim=128, lora_rank = 4, lora_scale = 1.0):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.enc1 = self.conv_block(1+in_channels, base_channels, time_dim)
        self.enc2 = self.conv_block(base_channels, base_channels*2, time_dim)

        self.center = self.conv_block(base_channels*2, base_channels*4, time_dim) 
        self.mid_attn = AttentionBlock(base_channels*4, num_heads=4)

        self.dec2 = self.conv_block(base_channels*6, base_channels*2, time_dim) 
        self.dec1 = self.conv_block(base_channels*3, base_channels, time_dim)

        self.lora_pre_dec2 = LoRA(base_channels * 6, rank =lora_rank, scale = lora_scale)
        self.lora_between = LoRA(base_channels * 2, rank =lora_rank, scale = lora_scale)

        self.film = FiLMLayer(base_channels * 2, cond_channels = in_channels) #This module should the learn the inc/dec cost
        
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c, time_dim):
        return TimeAwareBlock(in_c, out_c, time_dim)

    def forward(self, x_t, t, conditioning):
        
        x=torch.cat([x_t, conditioning], dim=1)
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(F.max_pool2d(e1, 2), t_emb)

        c=self.center(F.max_pool2d(e2,2), t_emb)
        c = self.mid_attn(c)

        c_up = F.interpolate(c, scale_factor=2, mode='bilinear', align_corners=False)
        
        d2_in = torch.cat([c_up,e2], dim=1)
        d2_in = self.lora_pre_dec2(d2_in)
        d2 = self.dec2(d2_in, t_emb)
        d2 = self.lora_between(d2)
        d2 = self.film(d2, conditioning[:, :-1])

        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False) 
        d1_in = torch.cat([d2_up, e1], dim=1)
        d1 = self.dec1(d1_in, t_emb)

        return self.final(d1)


    def set_finetune(self, active=True):
        for p in self.parameters():
            p.requires_grad = not active

        if active:
            #for p in self.lora_pre_dec2.parameters():
            #    p.requires_grad = True
            #for p in self.lora_between.parameters():
            #    p.requires_grad = True
            for p in self.film.parameters():
                p.requires_grad = True

