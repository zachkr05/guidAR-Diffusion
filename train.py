

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import your Data Generator
from DataGenerator.dataset_costmap import MultiClassCostmapDataset, get_cond_channels
from DataGenerator.sim import NUM_CLASSES

# Import your Architecture
from MoE.compositional_model import CompositionalModel


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better for structural learning than linear.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ds =  MultiClassCostmapDataset(
            n_samples = args.n_samples,
            H = args.img_size, W = args.img_size,
            n_obs_per_class=args.n_obs_per_class
            )
    dl = DataLoader(ds,batch_size=args.batch_size, shuffle=True, num_workers=4)

    expert_channels = 5 + ((NUM_CLASSES -1) *3) + 1 #base + context + density

    model = CompositionalModel(
            num_classes=NUM_CLASSES,
            expert_input_channels=expert_channels
            ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    criterion = nn.MSELoss()

    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas = 1.00 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epochj {epoch+1} / {args.epochs}")
        epoch_loss = 0.00

        for cond, x0 in pbar:
            cond = cond.to(device)
            x0 = x0.to(device)
            B = x0.shape[0]

            optimizer.zero_grad()

            t = torch.randint(0, args.timesteps, (B,), device=device).long()

            noise = torch.randn_like(x0)
            sqrt_alpha_t = sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            
            x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise

            pred_x0 = model(x_t, t, cond, training_phase='base')

            loss = criterion (pred_x0, x0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
            
        avg_loss = epoch_loss / len(dl)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
    
        save_path = os.path.join(args.checkpoint_dir, f"base_experts_ep{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--n_obs_per_class", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    train(args)
