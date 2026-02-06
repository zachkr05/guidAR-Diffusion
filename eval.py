
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataGenerator.dataGenerator import CostmapDataset
from MoE.ddpm import DDPM
from MoE.UNet import LightweightUNet
from train import ExpertEnsemble
from utils import *
from utils.finetune import finetune_models_focused
from utils.planner import SoftGridPlanner


def get_user_input(batch,model,device,ddpm, obstacle_classes):
    
    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal = batch 

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)

            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            diffused_cm[cls].append(generated)
    
    fused_costmap = fuse_costmaps(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal)
        
    return orig_path, user_path, diffused_cm

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
    replay_buffer = []



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
    train_batch = next(iter(loader))

    #Generate costmaps and get user input
    orig_path, user_path, diffused_cms = get_user_input(obstacle_classes= obstacle_classes,batch=train_batch,model=model,device=device,ddpm=ddpm,)

    if np.all(user_path == None):
        print("No modifications made to original path!")
        return None
    
    #Get classes to modify
    #target_classes, area_mask = identify_classes(obstacle_classes=obstacle_classes,batch=batch, orig_path=orig_path, user_path=user_path)
    
    # Get edit regions
    edit_regions = get_edit_regions(
        orig_path=orig_path,
        user_path=user_path,
        obstacle_classes=obstacle_classes,
        batch=train_batch,
    )

    print(f"Found {len(edit_regions)} edit region(s)")
    filtered_edit_regions = []
    affected_class_threshold = 0.83
    for i, (class_contributions, points, mask) in enumerate(edit_regions):
    #    dominant_class = max(class_contributions, key=class_contributions.get)
    #    print(f"  Region {i}: {len(points)} pixels, dominant class = {dominant_class} ({class_contributions[dominant_class]*100:.1f}%)")
        curr_prob = 0
        affected_classes = set()
        while(curr_prob<affected_class_threshold):
            remaining_classes = {k: v for k, v in class_contributions.items() if k not in affected_classes}
            best_class = max(remaining_classes, key = remaining_classes.get)
            curr_prob += remaining_classes[best_class]
            affected_classes.add(best_class)
        filtered_edit_regions.append([mask, points, affected_classes])
        print(f" Region {i}: {len(points)} pixels ; Affected Classes: {affected_classes}")

    #Different scene to eval

    eval_batch = next(iter(loader))
   
    
    print(filtered_edit_regions)
    
    #Eval new scene before finetuning the models
    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal = eval_batch 

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)

            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            diffused_cm[cls].append(generated)
    
    before_fused_costmap = fuse_costmaps(diffused_cm)
   
    #Finetune the models
    loss_history = finetune_models(
        model=model,
        batch=train_batch,
        orig_path=orig_path,
        user_path=user_path,
        device=device,
        lr=1e-4,
        edit_regions=filtered_edit_regions,
        epochs=500,
        ddpm=ddpm,
        planner=SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device),
    )


    # After finetune_models(...)
    visualize_results(
        model=model,
        train_batch=train_batch,
        eval_batch=eval_batch,
        orig_path=orig_path,
        user_path=user_path,
        obstacle_classes=obstacle_classes,
        device=device,
        ddpm=ddpm,
        before_fused_train=fuse_costmaps(diffused_cms),  # from get_user_input
        before_fused_eval=before_fused_costmap,            # already computed
        planner=SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device),
        save_path="finetuning_results.png",
    )

def visualize_results(
    model,
    train_batch,
    eval_batch,
    orig_path,
    user_path,
    obstacle_classes,
    device,
    ddpm,
    before_fused_train,
    before_fused_eval,
    planner,
    save_path="finetuning_results.png",
):
    """
    Visualize before/after finetuning on both train and eval scenes.
    Shows: costmap before, costmap after, delta, and planned paths.
    """
    model.eval()

    # --- Generate after-finetuning costmaps ---
    def generate_fused(batch):
        features, targets, positions, radii, goal = batch
        diffused_cm = {cls: [] for cls in obstacle_classes}
        with torch.no_grad():
            for cls in obstacle_classes:
                cond = features[cls].to(device)
                gt = targets[cls].to(device)
                generated = ddpm.sample(model.experts[cls], cond, shape=gt.shape)
                diffused_cm[cls].append(generated)
        return fuse_costmaps(diffused_cm)

    after_fused_train = generate_fused(train_batch)
    after_fused_eval = generate_fused(eval_batch)

    # --- Plan paths on after costmaps ---
    def plan_path(fused_costmap, batch):
        _, _, _, _, goal = batch
        cost_map = F.softplus(fused_costmap) + 0.1
        with torch.no_grad():
            visitation = planner(cost_map.to(device), goal.to(device))
        return visitation.cpu().numpy()[0, 0]

    after_vis_train = plan_path(after_fused_train, train_batch)
    after_vis_eval = plan_path(after_fused_eval, eval_batch)
    before_vis_train = plan_path(before_fused_train, train_batch)
    before_vis_eval = plan_path(before_fused_eval, eval_batch)

    # --- Convert to numpy ---
    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().squeeze()
        return np.array(t).squeeze()

    before_train = to_np(before_fused_train)
    after_train = to_np(after_fused_train)
    before_eval = to_np(before_fused_eval)
    after_eval = to_np(after_fused_eval)

    delta_train = after_train - before_train
    delta_eval = after_eval - before_eval

    # --- Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 0: Training scene
    # Row 1: Eval scene
    titles_top = ["Before (Train)", "After (Train)", "Delta (Train)", "Visitation (Train)"]
    titles_bot = ["Before (Eval)", "After (Eval)", "Delta (Eval)", "Visitation (Eval)"]

    vmin_cost = min(before_train.min(), after_train.min(), before_eval.min(), after_eval.min())
    vmax_cost = max(before_train.max(), after_train.max(), before_eval.max(), after_eval.max())
    delta_abs_max = max(abs(delta_train).max(), abs(delta_eval).max())

    # Training scene
    im0 = axes[0, 0].imshow(before_train, cmap="hot", vmin=vmin_cost, vmax=vmax_cost)
    im1 = axes[0, 1].imshow(after_train, cmap="hot", vmin=vmin_cost, vmax=vmax_cost)
    im2 = axes[0, 2].imshow(delta_train, cmap="RdBu_r", vmin=-delta_abs_max, vmax=delta_abs_max)
    im3 = axes[0, 3].imshow(before_vis_train, cmap="Blues", alpha=0.5)
    axes[0, 3].imshow(after_vis_train, cmap="Reds", alpha=0.5)

    # Overlay paths on training scene
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else np.array(orig_path)
    user_np = user_path if isinstance(user_path, np.ndarray) else np.array(user_path)

    for col in range(4):
        axes[0, col].plot(orig_np[:, 0], orig_np[:, 1], 'g--', linewidth=1.5, label='Original')
        axes[0, col].plot(user_np[:, 0], user_np[:, 1], 'c-', linewidth=1.5, label='User')
        axes[0, col].set_title(titles_top[col], fontsize=12)
        axes[0, col].axis('off')

    axes[0, 0].legend(loc='upper right', fontsize=8)

    # Eval scene
    axes[1, 0].imshow(before_eval, cmap="hot", vmin=vmin_cost, vmax=vmax_cost)
    axes[1, 1].imshow(after_eval, cmap="hot", vmin=vmin_cost, vmax=vmax_cost)
    axes[1, 2].imshow(delta_eval, cmap="RdBu_r", vmin=-delta_abs_max, vmax=delta_abs_max)
    axes[1, 3].imshow(before_vis_eval, cmap="Blues", alpha=0.5, label="Before")
    axes[1, 3].imshow(after_vis_eval, cmap="Reds", alpha=0.5, label="After")

    for col in range(4):
        axes[1, col].set_title(titles_bot[col], fontsize=12)
        axes[1, col].axis('off')

    # Colorbars
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Cost")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Cost")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Δ Cost")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-delta_abs_max, delta_abs_max)),
        ax=axes[1, 2], fraction=0.046, pad=0.04, label="Δ Cost"
    )

    # Add text annotation for visitation columns
    axes[0, 3].text(5, 10, "Blue=Before, Red=After", color='white', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    axes[1, 3].text(5, 10, "Blue=Before, Red=After", color='white', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    fig.suptitle("Finetuning Results: Training Scene vs Eval Scene", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate() 
