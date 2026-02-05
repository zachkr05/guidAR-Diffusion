
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
from utils.finetune_focused import finetune_models_focused
from utils.planner import SoftGridPlanner


def make_expert_target(user_path, H, W, device):
    """

    Turns user Demonstrated path into a gaussian probability distribution on the current costmap. 

    """
    target = torch.zeros((1, 1, H, W), device=device)

    
    #Turn user path into a tensor
    if isinstance(user_path, np.ndarray):
        path_tensor = torch.from_numpy(user_path).float().to(device)
    else:
        path_tensor = user_path

    xs = path_tensor[:, 0].long().clamp(0, W-1)
    ys = path_tensor[:, 1].long().clamp(0, H-1)
    
    target[0, 0, ys, xs] = 1.0
    
    #Make it a gaussian distribution
    target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    target = target / (target.sum() + 1e-8) # norm to 1 since its a probabilit dist.
    
    return target

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

        print(f" Region {i}: {len(points)} pixels ; Affected Classes: {affected_classes}")

    #Different scene to eval

    eval_batch = next(iter(loader))
    
    
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
    
    #Get difference between the models
    
     

    return 

    #Finetune the models
    #loss_history = finetune_models_focused(
    #    model=model,
    #    batch=batch,
    #    orig_path=orig_path,  # You already have this!
    #    user_path=user_path,
    #    device=device,
    #    lr=1e-4,
    #    target_class="chair",
    #    epochs=500,
    #    ddpm=ddpm,
    #    planner=SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device),
    #)

if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate() 
