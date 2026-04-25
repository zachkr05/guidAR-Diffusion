import torch
import torch.nn as nn
from .simple_unet import SimpleTrajectoryUNet


class SimpleTrajectoryModel(nn.Module):
    """
    Concatenate all per-class features + noisy trajectory -> UNet -> noise.

    For IRL finetuning, pass a film_cond vector (B, film_cond_dim) derived
    from user edits / preference signal. During base training, film_cond
    is None and FiLM layers act as identity (initialized to gamma=1, beta=0).
    """

    def __init__(self, obstacle_classes, in_channels_per_class=13,
                 base_channels=64, time_dim=256, film_cond_dim=128,
                 lora_rank=4, lora_scale=1.0):
        super().__init__()
        self.class_order = list(obstacle_classes)
        num_classes = len(self.class_order)

        # 1 (noisy traj) + num_classes * feature_channels
        total_in = 1 + num_classes * in_channels_per_class

        self.unet = SimpleTrajectoryUNet(
            in_channels=total_in,
            base_channels=base_channels,
            time_dim=time_dim,
            film_cond_dim=film_cond_dim,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
        )

    def forward(self, x_t, t, conditioning, film_cond=None):
        if isinstance(conditioning, dict):
            all_features = torch.cat(
                [conditioning[cls] for cls in self.class_order], dim=1
            )
            x_in = torch.cat([x_t, all_features], dim=1)
        else:
            x_in = torch.cat([x_t, conditioning], dim=1)

        return self.unet(x_in, t, film_cond=film_cond)

    def set_finetune(self, active=True):
        """Freeze everything except LoRA and FiLM for IRL finetuning."""
        self.unet.set_finetune(active)
