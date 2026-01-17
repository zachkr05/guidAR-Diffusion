

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from DataGenerator.dataGenerator import CostmapDataset

from MoE.UNet import LightweightUNet
from MoE.ddpm import DDPM

class ExpertEnsemble(nn.Module):
    def __init__(self, obstacle_classes, conditioning_channels):
        super().__init__()
        self.obstacle_classes = obstacle_classes
        self.experts = nn.ModuleDict({
                        obs_class: LightweightUNet(
                            in_channels = conditioning_channels,
                            context_channels = 0,
                            out_channels = 1
                            ) for obs_class in obstacle_classes
            })


    def forward(self, obs_class, x_t, t, conditioning):
        return self.experts[obs_class](x_t, t, conditioning)


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    obstacle_classes = ["chair", "table", "bomb"]
    batch_size = 32
    epochs = 100
    lr=1e-4
    checkpoint_dir = "checkpoints"  
    save_every = 1
    dataset = CostmapDataset(n_samples=250000, H=64, W=64)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers =4)

    
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes-1) +1

    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    ddpm = DDPM(timesteps=1000, device = device)

    for epoch in range(epochs):
        model.train()
        epoch_losses = {obs_class: 0.0 for obs_class in obstacle_classes}

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} / {epochs}")
        for features, targets in pbar:

            optimizer.zero_grad()
            total_loss = 0.0

            for cls in obstacle_classes:
                conditioning = features[cls].to(device)
                x_0 = targets[cls].to(device)

                loss = ddpm.compute_loss(model.experts[cls], x_0, conditioning)
                total_loss = total_loss + loss
                epoch_losses[cls] += loss.item()


            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({cls: f"{epoch_losses[cls]/max(1,pbar.n):.4f}" for cls in obstacle_classes})
        
        print(f"Epoch {epoch+1} Summary: ")
        for cls in obstacle_classes:
            avg_loss = epoch_losses[cls] / len(loader)
            print(f"  {cls}: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch{epoch+1}.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")


if __name__ == "__main__":
    train()
