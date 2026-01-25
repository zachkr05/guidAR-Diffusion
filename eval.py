# eval.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from DataGenerator.dataGenerator import CostmapDataset
from MoE.UNet import LightweightUNet
from MoE.ddpm import DDPM
from train import ExpertEnsemble
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from plotter import user_interactive_plot 
from pathlib import Path

import numpy as np
from scipy.special import logsumexp  # pip install scipy



def evaluate(checkpoint_path, num_samples=4, save_dir="eval_results"):
    
    # Config (must match training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obstacle_classes = ["chair", "table", "bomb"]
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes - 1) + 1
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # DDPM
    ddpm = DDPM(timesteps=1000, device=device)
    
    # Init the sim
    dataset = CostmapDataset(n_samples=num_samples, H=64, W=64)
    dataset.obstacle_classes = obstacle_classes
    
    # Channel names for labeling
    channel_names = ["own_bin", "own_rad", "other_bin_1", "other_bin_2", "other_rad_1", "other_rad_2", "goal"]

    #TODO: Make it for N samples

    #all_generated = []

    #features, targets, obstacle_positions, obstacle_radii, goal = dataset[0]
    features, targets = dataset[0]
    goal = dataset.goal

    all_generated = {}
    all_targets = {}
    
    print("Generating costmaps...")
    for cls in obstacle_classes:
        conditioning = features[cls].unsqueeze(0).to(device)
        gt_costmap = targets[cls].squeeze().numpy()
        
        shape = (1, 1, 64, 64)
        generated = ddpm.sample(model.experts[cls], conditioning, shape)
        generated = generated.squeeze().cpu().numpy()
        
        all_generated[cls] = generated
        all_targets[cls] = gt_costmap
        print(f"  {cls}: done")
    
    # Fuse costmaps
    print("Fusing costmaps...")
    #fused, responsibilities = fuse_costmaps_softmax_surface(all_generated, obstacle_positions, obstacle_radii, temperature=5)
    # Fuse costmaps (choose a rule)
    fused = np.maximum.reduce([all_generated[k] for k in obstacle_classes])

    # NEW: responsibilities from cost contributions
    responsibilities = responsibilities_from_costmaps(
        all_generated,
        alpha=10.0,
        power=2.0,
        free_space="uniform"
    )

    
    # Interactive plot
    #print("Launching interactive plot...")
    #interactive_costmap_plot(fused, responsibilities, obstacle_classes, goal)

    #New plotting
    #user_interactive_plot(fused, responsibilities, obstacle_positions, obstacle_radii, goal)
    interactive_costmap_plot(fused, responsibilities, obstacle_classes, goal)
    print("Evaluation complete.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="eval_results")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.num_samples, args.save_dir)
