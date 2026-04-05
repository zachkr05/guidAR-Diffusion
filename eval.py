
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
from utils.planner import SoftGridPlanner
from skimage.graph import route_through_array

from matplotlib.patches import Circle
from matplotlib.lines import Line2D


def get_user_input(batch, model, device, ddpm, obstacle_classes):

    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal, orientations = batch

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            diffused_cm[cls].append(generated)

    fused_costmap = fuse_costmaps(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal, orientations)

    return orig_path, user_path, diffused_cm


def visualize_improvement(scene_train, scene_eval):

    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    train_fused_cm, train_orig_path, train_user_path, train_obstacle_classes, train_filtered_edit_regions, train_batch = scene_train

    fig, (train_ax, eval_ax, diff_ax) = plt.subplots(1,3, figsize=(128,128))
    
    map_np = train_fused_cm[0,0].detach().cpu().numpy()
    train_ax.imshow(map_np)

    _, _, positions, radii, goal, orientations = train_batch

    #Tech debt but obstacle positions is a list of dictionaries
    for cls, obs_list in positions[0].items():
        for i, pos in enumerate(obs_list):
            train_ax.plot(pos[1],pos[0], color=colors[cls], marker='o', label=f'{cls}')

            angle = orientations[cls][i].item()  # assuming orientations passed in same structure
            arrow_len = 5
            dx = arrow_len * np.cos(angle)
            dy = arrow_len * np.sin(angle)
            train_ax.arrow(pos[1], pos[0], dx, dy, 
                     head_width=1.5, head_length=1.0, 
                     fc=colors[cls], ec=colors[cls], alpha=0.8)
    
    goal = train_goal[0]
    train_ax.plot(goal[1], goal[0], color='olive', marker='*', label='goal')

    #Plot User and original curves

    train_ax.plot(train_orig_path[0], train_orig_path[1], linewidth=2, label = 'Original B-spline curve')

    train_ax.plot(train_user_path[0], train_user_path[1], linewidth=2, label = 'User B-spline curve')

    

def generate_fused(model, batch, obstacle_classes, device, ddpm):
    """Generate costmaps from each expert and fuse them."""
    features, targets, positions, radii, goal, _ = batch
    diffused_cm = {cls: [] for cls in obstacle_classes}
    per_class_maps = {}
    with torch.no_grad():
        for cls in obstacle_classes:
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            generated = ddpm.sample(model.experts[cls], cond, shape=gt.shape)
            diffused_cm[cls].append(generated)
            per_class_maps[cls] = generated.cpu().numpy().squeeze()
    fused = fuse_costmaps(diffused_cm)
    return fused, per_class_maps

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
    conditioning_channels = 4 + 4 * (n_classes - 1) + 1

    # --- Load Data ---
    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)

    #Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch8.pt", map_location=device)
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
    
    # After get_user_input returns diffused_cms:
    before_fused_train = fuse_costmaps(diffused_cms)
    before_per_class_train = {
        cls: diffused_cms[cls][0].cpu().numpy().squeeze() for cls in obstacle_classes
    }

    # Get edit regions
    edit_regions = get_edit_regions(
        orig_path=orig_path,
        user_path=user_path,
        obstacle_classes=obstacle_classes,
        batch=train_batch,
    )

    print(f"Found {len(edit_regions)} edit region(s)")
    filtered_edit_regions = []
    affected_class_threshold = 0.85
    for i, (class_contributions, points, mask) in enumerate(edit_regions):
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

    print(filtered_edit_regions)
    
    eval_batch = next(iter(loader))
   
    #Eval new scene before finetuning the models
    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal, _ = eval_batch 

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)

            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            diffused_cm[cls].append(generated)
    
    before_fused_costmap = fuse_costmaps(diffused_cm)
    
    #planner = SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device)

    #Finetune and rerun models

    after_fused_costmap = before_fused_costmap

    #Visualize the difference between scenes

    scene_train = (before_fused_train, orig_path, user_path, obstacle_classes,  filtered_edit_regions, train_batch)
    scene_eval = (before_fused_costmap, after_fused_costmap, eval_batch)

    visualize_improvement(scene_train, scene_eval)

if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate() 
