import torch
import torch.nn as nn
from .encoder import ObstacleEncoder, ObstacleTokenizer
from .UNet import TrajectoryUNet


class TrajectoryDiffusionModel(nn.Module):
    """
    Full pipeline:
        per-class features → encoders → tokenizer → obstacle tokens
        noisy heatmap + timestep + tokens → TrajectoryUNet → predicted noise

    Usage:
        # Training (DDPM calls forward directly):
        loss = ddpm.compute_loss(model, x_0, class_features_dict)

        # Sampling (pre-encode once, reuse across all timesteps):
        tokens = model.encode_obstacles(class_features_dict)
        trajectory = ddpm.sample(model, tokens, shape)
    """

    def __init__(
        self,
        obstacle_classes,
        in_channels=9,
        feat_dim=128,
        base_channels=32,
        time_dim=128,
        lora_rank=4,
        lora_scale=1.0,
    ):
        super().__init__()

        self.class_order = list(obstacle_classes)

        # Per-class encoders (shared architecture, separate weights)
        self.encoders = nn.ModuleDict(
            {cls: ObstacleEncoder(in_channels, feat_dim) for cls in self.class_order}
        )

        # Tokenizer with positional + class embeddings
        self.tokenizer = ObstacleTokenizer(
            feat_dim=feat_dim,
            num_classes=len(self.class_order),
        )

        # Trajectory denoiser
        self.unet = TrajectoryUNet(
            base_channels=base_channels,
            time_dim=time_dim,
            context_dim=feat_dim,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
        )

    def encode_obstacles(self, class_features):
        """
        Encode per-class feature maps into obstacle tokens.

        Args:
            class_features: dict {class_name: (B, C, H, W)} or list of tensors
                            (in class_order)

        Returns:
            obstacle_tokens: (B, num_classes * spatial, feat_dim)
        """
        if isinstance(class_features, dict):
            encoded = [
                self.encoders[cls](class_features[cls]) for cls in self.class_order
            ]
        else:
            # Already a list in class_order
            encoded = [
                self.encoders[cls](feat)
                for cls, feat in zip(self.class_order, class_features)
            ]
        return self.tokenizer(encoded)

    def forward(self, x_t, t, conditioning):
        if isinstance(conditioning, dict):
            tokens = self.encode_obstacles(conditioning)
            # Build a spatial hint: sum binary occupancy across classes
            occ = torch.stack(
                [conditioning[cls][:, 0:1] for cls in self.class_order], dim=0
            ).sum(dim=0).clamp(0, 1)
            x_in = torch.cat([x_t, occ], dim=1)  # (B, 2, H, W)
        else:
            tokens = conditioning
            x_in = x_t
        return self.unet(x_in, t, tokens)

    def set_finetune(self, active=True):
        """Freeze encoders and most of UNet; keep LoRA + FiLM trainable."""
        # Freeze encoders entirely
        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad = not active

        # Freeze tokenizer
        for p in self.tokenizer.parameters():
            p.requires_grad = not active

        # Delegate to UNet's finetune logic
        self.unet.set_finetune(active)
