import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditioned on a pre-pooled vector (e.g. mean-pooled obstacle tokens).
    """

    def __init__(self, feature_channels, cond_dim):
        super().__init__()
        self.gamma_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels),
        )

        # Initialize to identity: gamma=1, beta=0
        nn.init.zeros_(self.gamma_proj[-1].weight)
        nn.init.ones_(self.gamma_proj[-1].bias)
        nn.init.zeros_(self.beta_proj[-1].weight)
        nn.init.zeros_(self.beta_proj[-1].bias)

    def forward(self, h, cond_vector):
        """
        h:           (B, C, H, W) feature map
        cond_vector: (B, cond_dim) already-pooled conditioning
        """
        gamma = self.gamma_proj(cond_vector).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(cond_vector).unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta
