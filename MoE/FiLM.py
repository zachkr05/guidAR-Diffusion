
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
        """
        Args:
            h: (B, C, H, W) feature map from LoRA
            conditioning: (B, cond_channels, H, W) raw conditioning stack
        Returns:
            (B, C, H, W) modulated features
        """
        # Global average pool conditioning to (B, cond_channels)
        z = conditioning.mean(dim=[2, 3])

        gamma = self.gamma_proj(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_proj(z).unsqueeze(-1).unsqueeze(-1)

        return gamma * h + beta
