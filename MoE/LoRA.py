

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LoRA(nn.Module):
    
    def __init__(self, channels, rank, scale):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size = 1, bias=False)
        self.scale = scale
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return h + self.scale*self.up(self.down(h))

