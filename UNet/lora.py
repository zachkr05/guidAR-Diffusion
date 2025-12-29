#lora.py
 
import math
import torch
import toch.nn as nn
import torch.nn.functional as F
from time_emb import SinusoidalEmbeddings

class LoRA(nn.Module):
    """
    LoRA adapter that only affects specific input channels (obstacle classes).
    
    Each class gets its own low-rank adapter. The adapter output is masked
    to only affect the corresponding class's features.
    """
    def __init__(self, channels, num_classes=4, rank=4, alpha=1.0):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.rank = rank
        self.scaling = alpha / rank
        
        # Separate LoRA for each obstacle type
        self.lora_A = nn.ModuleList([
            nn.Conv2d(channels, rank, kernel_size=3, padding=1, bias=False)
            for _ in range(num_classes)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(num_classes)
        ])
        
        # Initialize
        for i in range(num_classes):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[i].weight)
        
        # Class-specific scaling factors (learnable, for fine-tuning strength)
        #self.class_scales = nn.Parameter(torch.ones(num_classes))
        #self.class_shifts = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, x, class_activations=None):
        """
        Args:
            x: feature map [B, C, H, W]
            class_activations: [B, NUM_CLASSES, H, W] - how much each class 
                              is present at each location (from conditioning)
        
        Returns:
            LoRA adjustment [B, C, H, W]
        """
        B, C, H, W = x.shape
        output = torch.zeros_like(x)
        
        for class_id in range(self.num_classes):
            # Compute LoRA output for this class
            lora_out = self.lora_B[class_id](self.lora_A[class_id](x))
            lora_out = lora_out * self.scaling # * self.class_scales[class_id] + self.class_shifts[class_id]
            
            # Weight by class activation if provided
            if class_activations is not None:
                weight = class_activations[:, class_id:class_id+1, :, :]  # [B, 1, H, W]
                lora_out = lora_out * weight
            
            output = output + lora_out #, instead of summation among all of the LoRA adapters, we should use TIES-MERGING
        
        return output
    
    def get_class_parameters(self, class_id):
        """Get parameters for a specific class (for targeted fine-tuning)."""
        return list(self.lora_A[class_id].parameters()) + \
               list(self.lora_B[class_id].parameters()) + \
               [self.class_scales]
    
    def freeze_all_except(self, class_id):
        """Freeze all classes except one."""
        for i in range(self.num_classes):
            for param in self.lora_A[i].parameters():
                param.requires_grad = (i == class_id)
            for param in self.lora_B[i].parameters():
                param.requires_grad = (i == class_id)
        self.class_scales.requires_grad = True  # Keep scales trainable



