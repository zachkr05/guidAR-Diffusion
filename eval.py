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

def collate_ignore_metadata(batch):
    """

    Fixes stacking bug cus some of the features vary in length 

    """

    features = default_collate([item[0] for item in batch])
    targets = default_collate([item[1] for item in batch])
    goals = default_collate([item[4] for item in batch])
    
    
    positions = [item[2] for item in batch]
    radii = [item[3] for item in batch]

    return features, targets, positions, radii, goals

def evaluate():
    
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

    get_user_adjustments(fused_costmap)

    #retrain()

    #visualize_costmap()
    

if __name__ == "__main__":
    evaluate() 
