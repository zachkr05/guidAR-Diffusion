import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_emb import SinusoidalEmbeddings

class LoRA(nn.Module):
    """LoRA with optional FiLM modulation and MC Dropout."""
    
    def __init__(self, channels, num_classes=4, rank=4, alpha=1.0, dropout_p=0.1):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout_p = dropout_p
        
        self.dropout = nn.Dropout(p=dropout_p)
        
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
            
        self.class_scales = nn.Parameter(torch.ones(num_classes))
    
    def forward(self, x, class_activations=None, spatial_mask=None):
        output = torch.zeros_like(x)
        
        for class_id in range(self.num_classes):
            h = self.lora_A[class_id](x)
            h = self.dropout(h)
            lora_out = self.lora_B[class_id](h)
            lora_out = lora_out * self.scaling * self.class_scales[class_id]
            
            if class_activations is not None:
                weight = class_activations[:, class_id:class_id+1, :, :]
                lora_out = lora_out * weight
            
            if spatial_mask is not None:
                lora_out = lora_out * spatial_mask
                
            output = output + lora_out
            
        return output

    def enable_mc_dropout(self):
        self.dropout.train()

    def freeze_all_except_classes(self, class_ids):
        """
        Freeze all LoRA paths EXCEPT the specified class IDs.
        Note: This does NOT freeze the base model (that is UNet's job).
        """
        if isinstance(class_ids, int):
            class_ids = [class_ids]
            
        for i in range(self.num_classes):
            is_active = i in class_ids
            
            for param in self.lora_A[i].parameters():
                param.requires_grad = is_active
            for param in self.lora_B[i].parameters():
                param.requires_grad = is_active
                
        self.class_scales.requires_grad = True

    def get_multi_class_parameters(self, class_ids):
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        params = []
        for i in class_ids:
            params.extend(self.lora_A[i].parameters())
            params.extend(self.lora_B[i].parameters())
        params.append(self.class_scales)
        return params
