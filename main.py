

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_utils import schedule_betas, q_sample
from DataGenerator.dataset_costmap import MultiClassCostmapDataset
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES
from UNet.UNet import UNet
from train import sample_ddpm_with_cond
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def visualize_comparison(ground_truth, prediction):

    fig = make_subplots(
            rows=1, cols = 2,
            subplot_titles=("Ground Truth", "Prediction"),
            horizontal_spacing = 0.05
            )

    fig.add_trace(go.Heatmap(
        z=ground_truth,
        colorscale='Viridis',
        zmin=0, zmax=1,
        showscale=True
        ), row=1,col=1)

    fig.add_trace(go.Heatmap(
        z=prediction,
        colorscale='Viridis',
        zmin=0, zmax=1,
        showscale=True
        ), row = 1, col=2)

    fig.update_layout(height=400,width=1200,  title_text = "Diffusion Costmap Comparison")
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange="reversed")

    fig.show()

def run_full_pipeline(model, dataset, args, device="cuda"):

    model.eval()
    
    betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)

    for batch_idx, (cond, x0) in enumerate(dataset):

        cond = cond.to(device) #[16, 4 + 1, 64,64]
        x0 = x0.to(device)

        with torch.no_grad():
            generated_maps = sample_ddpm_with_cond(
                    model,
                    cond,
                    betas,
                    alphas,
                    alpha_bar,
                    device=device)
        # process each scenario

        batch_size = cond.shape[0]
        for i in range(batch_size):
            map_pred = generated_maps[i,0].cpu().numpy()
            map_gt = (x0[i, 0].cpu().numpy() + 1.0) / 2.0
            print(f"Sample {batch_idx}:")
            print(f"  Prediction - min: {map_pred.min():.6f}, max: {map_pred.max():.6f}, mean: {map_pred.mean():.6f}")
            print(f"  Ground Truth - min: {map_gt.min():.6f}, max: {map_gt.max():.6f}, mean: {map_gt.mean():.6f}")
            print(f"  Any NaN in pred? {np.isnan(map_pred).any()}")
            print(f"  Any NaN in GT? {np.isnan(map_gt).any()}")
            mse = np.mean(np.square((map_pred - map_gt)))
            print(f"Sample {i}: MSE Loss = {mse:.5f}")
            visualize_comparison(map_pred, map_gt)
    return 

def seed_env(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)



def seed_env(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__": 
    # FIX 6: Initialize Argument Parser (was missing)
    parser = argparse.ArgumentParser()
    
    # Diffusion Hyperparameters
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Beta schedule start")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta schedule end")
    parser.add_argument("--lora-rank", type=int, default=8, help="Rank for LoRA adapters")
    
    # Evaluation settings
    parser.add_argument("--n-samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--img-size", type=int, default=64, help="Image resolution")
    parser.add_argument("--n-obs-per-class", type=int, default=5, help="Max obstacles per class")
    parser.add_argument("--min-obs-per-class", type=int, default=1, help="Min obstacles per class")
    parser.add_argument("--min-total-obs", type=int, default=3, help="Minimum total obstacles")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", help="Path to checkpoint")

    args = parser.parse_args()

    seed_env(42) 
    
    # FIX 7: Define device inside main
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset & Dataloader
    eval_ds = MultiClassCostmapDataset(
        n_samples=args.n_samples,
        H=args.img_size,
        W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    # Note: Keep batch_size small for visualization so Plotly doesn't open 100 tabs at once
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False)

    # Load Model
    # FIX 8: Pass num_classes explicitly if your config uses it
    model = UNet(in_channels=NUM_CLASSES+2, lora_rank=args.lora_rank, num_classes=NUM_CLASSES).to(device)
    
    # FIX 9: Correct checkpoint loading variable names
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found! Using random weights.")

    # Run
    # FIX 10: Pass required args parameter
    run_full_pipeline(model, eval_dl, args, device=device)
