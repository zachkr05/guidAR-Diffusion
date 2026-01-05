 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings



# Updated lora.py

class LoRA(nn.Module):
    """LoRA with optional FiLM modulation."""
    
    def __init__(self, channels, num_classes=4, rank=4, alpha=1.0):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.ModuleList([
            nn.Conv2d(channels, rank, kernel_size=3, padding=1, bias=False)
            for _ in range(num_classes)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(num_classes)
        ])
        
        for i in range(num_classes):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[i].weight)
    
    def forward(self, x, class_activations=None):
        """
        Args:
            x: [B, C, H, W]
            class_activations: [B, num_classes, H, W]
            gammas: [B, num_classes] - FiLM scale (optional)
            betas: [B, num_classes] - FiLM shift (optional)
        """
        B, C, H, W = x.shape
        output = torch.zeros_like(x)
        
        for class_id in range(self.num_classes):
            lora_out = self.lora_B[class_id](self.lora_A[class_id](x))
            lora_out = lora_out * self.scaling
            
            # Apply FiLM if provided
#            if gammas is not None and betas is not None:
 #               gamma = gammas[:, class_id].view(B, 1, 1, 1)
  #              beta = betas[:, class_id].view(B, 1, 1, 1)
   #             lora_out = gamma * lora_out + beta
            
            # Weight by class activation
            if class_activations is not None:
                weight = class_activations[:, class_id:class_id+1, :, :]
                lora_out = lora_out * weight
            
            output = output + lora_out
        
        return output
