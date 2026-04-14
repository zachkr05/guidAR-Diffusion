
import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, feature_channels, cond_channels):
        """
        Feature-wise Linear Modulation.

        Args:
            feature_channels: number of channels in the feature map h
            cond_channels: number of channels in the conditioning stack
                           (before pooling — this layer handles the pooling)
        """
        super().__init__()
        # Pool conditioning spatially, then project to gamma/beta
        self.gamma_proj = nn.Sequential(
            nn.Linear(cond_channels, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(cond_channels, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels),
        )

        # Initialize to identity: gamma=1, beta=0
        nn.init.zeros_(self.gamma_proj[-1].weight)
        nn.init.ones_(self.gamma_proj[-1].bias)
        nn.init.zeros_(self.beta_proj[-1].weight)
        nn.init.zeros_(self.beta_proj[-1].bias)


    def forward(self, h, conditioning):
        # Channel 1 = curr_rad, channels 2,3 = curr_sin, curr_cos
        rad = conditioning[:, 1:2].abs()  # (B, 1, H, W)
        sin_map = conditioning[:, 2:3]
        cos_map = conditioning[:, 3:4]
        
        # Orientation magnitude as additional weight: 
        # nonzero sin/cos means an obstacle is there with a real orientation
        orient_mag = torch.sqrt(sin_map**2 + cos_map**2 + 1e-8)
        
        # Combined weight: radius * orientation presence
        weights = rad * orient_mag  # (B, 1, H, W)
        weights = weights / (weights.sum(dim=[2, 3], keepdim=True) + 1e-8)
        
        z = (conditioning * weights).sum(dim=[2, 3])  # (B, cond_channels)
        
        gamma = self.gamma_proj(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(z).unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta
