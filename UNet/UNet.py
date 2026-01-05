# UNet_class_lora.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings
from .lora import LoRA
from .attention import AttentionBlock
from .unet_parts import *


NUM_CLASSES=4
class UNet(nn.Module):
    """
    Each obstacle class has its own LoRA, allowing independent fine-tuning.
    """
    def __init__(
        self, 
        in_channels=6,  # NUM_CLASSES + 1 (goal) + 1 (noisy costmap)
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),
        attention_resolutions=(2, 3),
        time_dim=128,
        num_heads=4,
        dropout=0.1,
        num_classes=NUM_CLASSES,
        lora_rank=8,
        lora_alpha=16.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.num_classes = num_classes
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = i in attention_resolutions
            self.downs.append(DownBlock(ch, out_ch, time_dim, use_attention=use_attn, num_heads=num_heads))
            ch = out_ch
            channels.append(ch)
        
        # Bottleneck
        self.mid_block1 = ResidualConvBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResidualConvBlock(ch, ch, time_dim, dropout)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = i in attention_resolutions
            self.ups.append(UpBlock(ch, out_ch, time_dim, use_attention=use_attn, num_heads=num_heads))
            ch = out_ch
        
        # Final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
        
        # Class-specific LoRA at bottleneck and decoder
        self.lora_mid = LoRA(
            channels[-1], num_classes=num_classes, rank=lora_rank, alpha=lora_alpha
        )
        self.lora_ups = nn.ModuleList([
            LoRA(
                base_channels * mult, num_classes=num_classes, rank=lora_rank, alpha=lora_alpha
            )
            for mult in reversed(channel_mults)
        ])
    
    def forward(self, x, t):
        """
        Args:
            x: [B, in_channels, H, W] 
               Channels: [class_0, class_1, ..., class_N, goal, noisy_costmap]
        """
        # Extract class activations from input (first NUM_CLASSES channels)
        class_activations = x[:, :self.num_classes, :, :]  # [B, NUM_CLASSES, H, W]
        
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Initial conv
        h = self.init_conv(x)
        
        # Store class activations at each resolution for LoRA
        class_acts_downsampled = [class_activations]
        
        # Downsampling with skip connections
        skips = [h]
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
            # Downsample class activations to match
            class_acts_downsampled.append(
                F.interpolate(class_acts_downsampled[-1], size=h.shape[2:], mode='bilinear', align_corners=True)
            )
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
    
        class_acts_at_res = F.interpolate(
        class_activations, size=h.shape[2:], mode='bilinear', align_corners=True
        )

        # Class-specific LoRA at bottleneck
        h = h + self.lora_mid(h, class_acts_at_res)
        
        # Upsampling with skip connections
        for i, up in enumerate(self.ups):
            skip = skips.pop()
            #class_acts_downsampled.pop()
            h = up(h, skip, t_emb)
            
            # Class-specific LoRA at each decoder level
            # Use class activations at this resolution
            current_class_acts = F.interpolate(
                class_activations, size=h.shape[2:], mode='bilinear', align_corners=True
            )
            h = h + self.lora_ups[i](h, current_class_acts)
        
        return self.final_conv(h)
    
    def freeze_base_model(self):
        """Freeze everything except LoRA."""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_all_lora_except_class(self, class_id):
        """
        Freeze base model AND all LoRA except for one class.
        Use this when fine-tuning for a specific class.
        
        Args:
            class_id: 0=chair, 1=table, 2=person, 3=wall
        """
        # Freeze base model
        self.freeze_base_model()
        
        # Freeze all LoRA except target class
        self.lora_mid.freeze_all_except(class_id)
        for lora in self.lora_ups:
            lora.freeze_all_except(class_id)
    
    def get_lora_parameters(self):
        """Return all LoRA parameters."""
        params = []
        params.extend(self.lora_mid.parameters())
        for lora in self.lora_ups:
            params.extend(lora.parameters())
        return params
    
    def get_class_lora_parameters(self, class_id):
        """Return LoRA parameters for a specific class."""
        params = []
        params.extend(self.lora_mid.get_class_parameters(class_id))
        for lora in self.lora_ups:
            params.extend(lora.get_class_parameters(class_id))
        return params
    
    def count_parameters(self):
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

