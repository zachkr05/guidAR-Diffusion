

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
from skimage.graph import route_through_array

def visualize_cost_delta(
    original_path_rows,
    original_path_cols,
    adjusted_path_rows,
    adjusted_path_cols,
    H: int,
    W: int,
    obstacles: dict = None
):
    """Visualize the cost delta computed from trajectory difference."""
    from irl_agent import compute_geometric_delta
    
    # Convert to list of tuples
    original_traj = list(zip(original_path_rows, original_path_cols))
    user_traj = list(zip(adjusted_path_rows, adjusted_path_cols))
    
    # Compute delta
    delta = compute_geometric_delta(original_traj, user_traj, H, W)
    
    fig = go.Figure()
    
    # Delta heatmap (diverging colorscale: blue = decrease, red = increase)
    fig.add_trace(go.Heatmap(
        z=delta,
        colorscale='RdBu_r',  # Red for positive (avoid), Blue for negative (prefer)
        zmid=0,  # Center colorscale at 0
        colorbar=dict(title='Cost Delta')
    ))
    
    # Original path
    fig.add_trace(go.Scatter(
        x=list(original_path_cols),
        y=list(original_path_rows),
        mode='lines+markers',
        name='Original Path',
        line=dict(color='black', width=3),
        marker=dict(size=4)
    ))
    
    # Adjusted path
    fig.add_trace(go.Scatter(
        x=list(adjusted_path_cols),
        y=list(adjusted_path_rows),
        mode='lines+markers',
        name='User Adjusted Path',
        line=dict(color='lime', width=3),
        marker=dict(size=4)
    ))
    
    # Obstacles if provided
    if obstacles:
        symbols = ['circle', 'square', 'diamond', 'cross']
        colors = ['red', 'blue', 'green', 'orange']
        for class_id, coords in obstacles.items():
            if len(coords) > 0:
                fig.add_trace(go.Scatter(
                    x=coords[:, 1],
                    y=coords[:, 0],
                    mode='markers',
                    name=f'Class {class_id}',
                    marker=dict(
                        symbol=symbols[class_id % len(symbols)],
                        size=12,
                        color=colors[class_id % len(colors)],
                        line=dict(width=1, color='black')
                    )
                ))
    
    fig.update_layout(
        height=500,
        width=600,
        title='Cost Delta from User Adjustment<br>(Red = Increase Cost, Blue = Decrease Cost)',
        yaxis=dict(autorange='reversed', scaleanchor='x'),
        xaxis=dict(constrain='domain')
    )
    
    fig.show()
    
    # Print stats
    print(f"Delta stats: min={delta.min():.3f}, max={delta.max():.3f}, mean={delta.mean():.3f}")
    print(f"Positive (avoid) pixels: {(delta > 0.01).sum()}")
    print(f"Negative (prefer) pixels: {(delta < -0.01).sum()}")
    
    return delta

def add_user_adjustments(
    path_rows,
    path_cols,
    class_1_coords: np.ndarray,
    avoidance_radius: float = 23.0,
    push_distance: float = 3.0
):
    """
    Push path points away from class 1 obstacles when within avoidance_radius.
    
    Args:
        path_rows: Row coordinates of path
        path_cols: Column coordinates of path
        class_1_coords: np.array of shape (N, 2) with (row, col) for class 1
        avoidance_radius: Trigger adjustment when closer than this
        push_distance: How far to push away
    
    Returns:
        Tuple of (adjusted_rows, adjusted_cols)
    """
    path_rows = np.array(path_rows, dtype=np.float32)
    path_cols = np.array(path_cols, dtype=np.float32)
    print("in func")  
    if len(class_1_coords) == 0:
        return path_rows, path_cols
    
    adjusted_rows = path_rows.copy()
    adjusted_cols = path_cols.copy()
    
    for i in range(len(path_rows)):
        pt = np.array([path_rows[i], path_cols[i]])
        
        # Distance to each class 1 obstacle
        diffs = class_1_coords - pt  # (N, 2)
        dists = np.linalg.norm(diffs, axis=1)
        print(dists) 
        # Find closest
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        
        if closest_dist < avoidance_radius:
            # Direction away from obstacle
            away = pt - class_1_coords[closest_idx]
            norm = np.linalg.norm(away)
            
            if norm > 1e-6:
                away = away / norm
                adjusted_rows[i] += away[0] * push_distance
                adjusted_cols[i] += away[1] * push_distance
    
    return adjusted_rows, adjusted_cols


def visualize_comparison(ground_truth, prediction, obstacles, goal):
    start_pos = (5,5)
    print(goal)
    goal = goal.numpy()
    goal_pos = (int(goal[0][0]), int(goal[0][1]))


    fig = make_subplots(
            rows=1, cols = 2,
            subplot_titles=("Ground Truth", "Prediction"),
            horizontal_spacing = 0.05
            )

    gt_indices, gt_weight = route_through_array(
        ground_truth, 
        start_pos, 
        goal_pos, 
        fully_connected=True, 
        geometric=True
    )



    pred_indices, pred_weight = route_through_array(
        prediction, 
        start_pos, 
        goal_pos, 
        fully_connected=True, 
        geometric=True
    )

    gt_path_rows, gt_path_cols = zip(*gt_indices)
    pred_rows, pred_cols = zip(*pred_indices)

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

    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']

    for class_id, coords in obstacles.items():
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (class_id == 0):
            adj_rows, adj_cols = add_user_adjustments(pred_rows,pred_cols, coords)
            delta = visualize_cost_delta(
            pred_rows, pred_cols,      # original
            adj_rows, adj_cols,        # adjusted
            H=ground_truth.shape[0],
            W=ground_truth.shape[1],
            obstacles=obstacles
        )
        fig.add_trace(go.Scatter(x=cols, y=rows, mode = 'markers', name = f'Class {class_id}',
            marker=dict(symbol=symbols[class_id % len(symbols)],
                size=12,
                color= colors[class_id % len(colors)],
                line=dict(width=1, color='black')
                )
            ))

    fig.add_trace(go.Scatter(
        x=adj_cols, 
        y=adj_rows, 
        mode='lines+markers', 
        name='A* Path Prediction',
        line=dict(color='white', width=3),
        marker=dict(size=4, color='white')
    ), row=1, col=2)


    fig.add_trace(go.Scatter(
        x=gt_path_cols, 
        y=gt_path_rows, 
        mode='lines+markers', 
        name='A* Path GT',
        line=dict(color='white', width=3),
        marker=dict(size=4, color='white')
    ), row=1, col=1)

    fig.update_layout(height=400,width=1200,  title_text = "Diffusion Costmap Comparison")
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange="reversed")

    fig.show()

def run_full_pipeline(model, dataset, args, device="cuda"):

    model.eval()
    
    betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)

    for batch_idx, (cond, x0, obstacles, goal) in enumerate(dataset):
        if batch_idx == 0 or batch_idx==4:
            continue

        np_map = {}
        cond = cond.to(device) #[16, 4 + 1, 64,64]
        x0 = x0.to(device)
#        print(obstacles)
        for class_id, coord_list in obstacles.items():
            np_map[class_id] = np.array([[pair[0].item(), pair[1].item()]for pair in coord_list])



        print(np_map)
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
            visualize_comparison(map_gt, map_pred, np_map,goal )
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
    parser.add_argument("--n-obs-per-class", type=int, default=1, help="Max obstacles per class")
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
