"""
Minimal Evaluation Script - Costmap and Trajectory Only
"""
from irl_agent import *
import argparse
import os
from utils import *
import numpy as np
import torch
import plotly.graph_objects as go
from skimage.graph import route_through_array
from plotter import *
from diffusion_utils import schedule_betas
from DataGenerator.dataset_costmap import (
    MultiClassCostmapDataset, 
    get_cond_channels,
    make_class_occupancy_maps, 
    make_goal_map, 
    make_orientation_maps
)
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES, NUM_ORIENTATIONS, orientation_to_vector
from UNet.UNet import UNet
from train import sample_ddpm_with_cond
from dataLoader import MetadataDataLoader

from plotly.subplots import make_subplots

def get_affected_class_ids(affected_objects_list):
    """Extracts unique class IDs from the affected objects dictionary."""
    if not affected_objects_list:
        return []
    # Extract class_id from the dicts and return unique list
    return list(set([obj['class_id'] for obj in affected_objects_list]))

def visualize_generated_vs_target(generated, target, path, obstacles_by_class, start_pos, goal_pos):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Generated", "Ground Truth (x0)"))
    
    # Generated
    fig.add_trace(go.Heatmap(z=generated, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    
    # Target
    fig.add_trace(go.Heatmap(z=target, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    
    # Add path to both
    path_arr = np.array(path)
    for col in [1, 2]:
        fig.add_trace(go.Scatter(x=path_arr[:, 1], y=path_arr[:, 0], mode='lines',
                                  line=dict(color='cyan', width=3), showlegend=False), row=1, col=col)
    
    fig.update_layout(height=400, width=900)
    fig.update_yaxes(autorange='reversed')
    fig.show()

def find_affected_objects(delta_map, obstacles_by_class, threshold=0.005):
    """
    Find objects where the user's correction indicates increased cost (avoidance).
    
    Args:
        delta_map: [H, W] array where positive = user wants higher cost
        obstacles_by_class: dict {class_id: [{'pos': (r,c), 'orientation': int}, ...]}
        threshold: minimum delta value to consider (default 0.0)
    
    Returns:
        List of dicts: [{'class_id': int, 'class_name': str, 'pos': (r,c), 'delta': float}, ...]
    """
    affected = []
    
    for class_id, obstacles in obstacles_by_class.items():
        class_name = OBSTACLE_CLASSES[class_id]['name']
        
        for obs in obstacles:
            r, c = obs['pos']
            
            # Clamp to valid indices
            r_idx = int(np.clip(r, 0, delta_map.shape[0] - 1))
            c_idx = int(np.clip(c, 0, delta_map.shape[1] - 1))
            
            delta_val = delta_map[r_idx, c_idx]
            
            if delta_val > threshold:
                affected.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'pos': (r, c),
                    'orientation': obs['orientation'],
                    'delta': delta_val
                })
    
    return affected

def run_evaluation(model, dataset, args, device="cuda"):
    """Run evaluation - just show costmap and trajectory."""
    model.eval()
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)

    for idx, (cond, x0, obstacles_by_class, goal) in enumerate(dataset):
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        
        H, W = args.img_size, args.img_size
        start_pos = (H - 5, 5)
        goal_pos = (int(goal[0]), int(goal[1]))
        
        # Generate costmap
        with torch.no_grad():
            generated = sample_ddpm_with_cond(model, cond, diff_betas, alphas, alpha_bar, device=device)
        
        costmap = generated[0, 0].cpu().numpy()
        costmap = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
       

        #costmap = np.where(costmap < 0.009, 0, costmap)
        # Plan trajectory
        path, _ = route_through_array(costmap, start_pos, goal_pos, fully_connected=True, geometric=True)
        
        orig_rows, orig_cols = zip(*path)
        orig_path = list(zip(orig_rows,orig_cols))

        adj_rows, adj_cols = simulate_user_correction(
        orig_rows, orig_cols, obstacles_by_class,
        target_class=0, avoidance_radius=20.0, push_strength=15.0
        )
        adj_path = list(zip(adj_rows, adj_cols))

        delta_map = compute_geometric_delta(orig_path, adj_path, H, W)
        interaction_mask = compute_interaction_mask(delta_map)
        affected = find_affected_objects(delta_map, obstacles_by_class)
        
        for obj in affected:
            print(f"{obj['class_name']} at {obj['pos']}, delta={obj['delta']:.3f}") 
        
        
        #x0_np = (x0[0].cpu().numpy() + 1) / 2
        #visualize_generated_vs_target(costmap, x0_np, path, obstacles_by_class, start_pos, goal_pos)
       
        costmap_raw = generated[0, 0].cpu().numpy()
        print(f"Raw output - min: {costmap_raw.min():.4f}, max: {costmap_raw.max():.4f}, std: {costmap_raw.std():.4f}")

        x0_np = x0[0].cpu().numpy()
        print(f"x0 - min: {x0_np.min():.4f}, max: {x0_np.max():.4f}, std: {x0_np.std():.4f}")
        
        # Visualize
        visualize_comparison(
        costmap, orig_path, adj_path, delta_map, interaction_mask,
        obstacles_by_class, start_pos, goal_pos,
        title=f"Scene User Correction Around {OBSTACLE_CLASSES[0]['name']}"
        )

def seed_env(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-obs-per-class", type=int, default=1)
    parser.add_argument("--min-obs-per-class", type=int, default=1)
    parser.add_argument("--min-total-obs", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    
    args = parser.parse_args()
    seed_env(55)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    eval_ds = MultiClassCostmapDataset(
        n_samples=args.n_samples, H=args.img_size, W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    
    eval_dl = MetadataDataLoader(eval_ds, args.n_samples)
    
    cond_channels = get_cond_channels()
    model = UNet(in_channels=cond_channels + 1, lora_rank=args.lora_rank, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    run_evaluation(model, eval_dl, args, device=device)
