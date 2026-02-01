import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import *
# Import local modules
from DataGenerator.dataGenerator import CostmapDataset
from MoE.ddpm import DDPM
from train import ExpertEnsemble
from torch.utils.data.dataloader import default_collate
from utils.planner import SoftGridPlanner
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


def make_expert_target(user_path, H, W, device):
    """
    Converts a user path (N, 2) [x, y] into a soft target image (1, 1, H, W).
    """
    target = torch.zeros((1, 1, H, W), device=device)
    
    # user_path comes in as (N, 2) numpy array
    if isinstance(user_path, np.ndarray):
        path_tensor = torch.from_numpy(user_path).float().to(device)
    else:
        path_tensor = user_path

    # Extract coordinates. Assuming user_path is [x, y] (col, row)
    # We clip to ensure we don't crash if the path touches the edge
    xs = path_tensor[:, 0].long().clamp(0, W-1)
    ys = path_tensor[:, 1].long().clamp(0, H-1)
    
    # "Draw" the path onto the grid
    target[0, 0, ys, xs] = 1.0
    
    # Smooth the target slightly (Blur) so the loss isn't too harsh
    # This helps the optimizer "find" the path if the prediction is slightly off
    target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    target = target / (target.sum() + 1e-8) # Normalize to sum to 1
    
    return target

def evaluate():
   
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    #Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"
    output_dir = "eval_results"
    obstacle_classes = ["chair", "table", "bomb"]
    H, W = 128, 128
    batch_size = 1
    
    # Calculate channels: 2 (curr) + 2*(n-1) (others) + 1 (goal)
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes - 1) + 1

    # --- Load Data ---
    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)

    #Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch5.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #Setup reverse diffusion process
    ddpm = DDPM(timesteps=1000, device=device)
    mse_results = {cls: [] for cls in obstacle_classes}
    diffused_cm = {cls: [] for cls in obstacle_classes}

    first_batch = None

    print("Starting evaluation...")
    with torch.no_grad():
        first_batch= next(iter(loader))
        features, targets, positions, radii, goal = first_batch

        for cls in obstacle_classes:
            print(f"Sampling for expert: {cls}...")
            
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            
            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            loss = F.mse_loss(generated, gt)
            mse_results[cls].append(loss.item())
            diffused_cm[cls].append(generated)

    fused_costmap = fuse_costmaps(diffused_cm)
    #print(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal)
    #print("original path: ", orig_path)
    
    from utils.finetune_focused import finetune_models_focused
    loss_history = finetune_models_focused(
        model=model,
        batch=first_batch,
        orig_path=orig_path,  # You already have this!
        user_path=user_path,
        device=device,
        lr=1e-4,
        target_class="chair",
        epochs=500,
        ddpm=ddpm,
        planner=SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device),
    )
    original_image_tensor = diffused_cm["chair"][0]
    
    visualize_improvement(
        model=model,
        target_class="chair",
        features=features,
        targets=targets,
        goal=goal,
        device=device,
        diffused_cm_old=diffused_cm,  # Pass the whole dict, not just chair
        positions=positions,
        radii=radii
    ) 

def visualize_improvement(model, target_class, features, targets, goal, device, diffused_cm_old, positions, radii):
    """
    1. Fuse OLD costmaps (before finetuning) -> compute trajectory
    2. Generate NEW costmap for target_class -> fuse with other classes -> compute trajectory
    3. Visualize side by side
    """
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"\nGenerating 'After' image for {target_class}...")
    
    # 1. Setup and generate new costmap for target class
    expert_model = model.experts[target_class]
    expert_model.eval()
    ddpm = DDPM(timesteps=1000, device=device)
    
    cond = features[target_class].to(device)
    shape = diffused_cm_old[target_class][0].shape  # Get shape from old generation
    
    with torch.no_grad():
        new_generated = ddpm.sample(expert_model, cond, shape=shape)
    
    # 2. Build NEW diffused_cm dict (replace only the target class)
    diffused_cm_new = {}
    for cls in model.obstacle_classes:
        if cls == target_class:
            diffused_cm_new[cls] = [new_generated]
        else:
            diffused_cm_new[cls] = diffused_cm_old[cls]  # Keep others the same
    
    # 3. Fuse costmaps using your existing function
    fused_old = fuse_costmaps(diffused_cm_old)
    fused_new = fuse_costmaps(diffused_cm_new)
    
    # 4. Compute trajectories using your existing function
    old_path, _ = get_user_adjustments(fused_old, positions, radii, goal)
    new_path, _ = get_user_adjustments(fused_new, positions, radii, goal)
    
    # 5. Convert fused maps to numpy for plotting
    # Adjust this based on what fuse_costmaps returns
    if isinstance(fused_old, torch.Tensor):
        fused_old_np = fused_old[0, 0].cpu().numpy()
        fused_new_np = fused_new[0, 0].cpu().numpy()
    elif isinstance(fused_old, dict):
        # If it returns a dict, extract the fused map
        fused_old_np = list(fused_old.values())[0][0, 0].cpu().numpy()
        fused_new_np = list(fused_new.values())[0][0, 0].cpu().numpy()
    else:
        fused_old_np = np.array(fused_old)
        fused_new_np = np.array(fused_new)
    
    goal_np = goal[0].cpu().numpy()
    
    # 6. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot: BEFORE finetuning
    im1 = axes[0].imshow(fused_old_np, cmap='viridis', origin='lower')
    axes[0].plot(old_path[:, 0], old_path[:, 1], 'r-', linewidth=2, label='Original Path')
    axes[0].plot(0, 0, 'go', markersize=10, label='Start')
    axes[0].plot(goal_np[1], goal_np[0], 'r*', markersize=15, label='Goal')
    axes[0].set_title("Before Fine-tuning")
    axes[0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot: AFTER finetuning
    im2 = axes[1].imshow(fused_new_np, cmap='viridis', origin='lower')
    axes[1].plot(new_path[:, 0], new_path[:, 1], 'b-', linewidth=2, label='New Path')
    axes[1].plot(0, 0, 'go', markersize=10, label='Start')
    axes[1].plot(goal_np[1], goal_np[0], 'r*', markersize=15, label='Goal')
    axes[1].set_title("After Fine-tuning")
    axes[1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot: Both paths overlaid on delta
    delta = fused_new_np - fused_old_np
    max_val = max(abs(np.min(delta)), abs(np.max(delta)), 1e-6)
    im3 = axes[2].imshow(delta, cmap='seismic', origin='lower', vmin=-max_val, vmax=max_val)
    axes[2].plot(old_path[:, 0], old_path[:, 1], 'r-', linewidth=2, label='Old Path')
    axes[2].plot(new_path[:, 0], new_path[:, 1], 'b-', linewidth=2, label='New Path')
    axes[2].plot(goal_np[1], goal_np[0], 'r*', markersize=15, label='Goal')
    axes[2].set_title("Costmap Delta + Both Paths\nRed=Before | Blue=After")
    axes[2].legend(loc='upper right')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"trajectory_comparison_{target_class}.png")
    plt.show()
    
    print(f"Saved to trajectory_comparison_{target_class}.png")
    
    return old_path, new_path

if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate() 
