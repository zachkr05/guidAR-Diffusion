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
#from film import FiLMGenerator, compute_irl_attribution
#from irl_finetune import finetune_film_on_edit
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage.graph import route_through_array
def visualize_comparison(ground_truth, prediction, obstacles, goal):
    """Simple comparison visualization (no FiLM before/after)."""
    start_pos = (5, 5)
    goal_np = goal.numpy() if hasattr(goal, 'numpy') else goal
    goal_pos = (int(goal_np[0][0]), int(goal_np[0][1]))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Ground Truth", "Prediction"),
        horizontal_spacing=0.05
    )

    gt_indices, _ = route_through_array(
        ground_truth, start_pos, goal_pos, fully_connected=True, geometric=True
    )
    pred_indices, _ = route_through_array(
        prediction, start_pos, goal_pos, fully_connected=True, geometric=True
    )

    gt_path_rows, gt_path_cols = zip(*gt_indices)
    pred_rows, pred_cols = zip(*pred_indices)

    fig.add_trace(go.Heatmap(
        z=ground_truth, colorscale='Viridis', zmin=0, zmax=1, showscale=True
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=prediction, colorscale='Viridis', zmin=0, zmax=1, showscale=True
    ), row=1, col=2)

    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']

    for class_id, coords in obstacles.items():
        if len(coords) > 0:
            rows = coords[:, 0]
            cols = coords[:, 1]
            fig.add_trace(go.Scatter(
                x=cols, y=rows, mode='markers', name=f'Class {class_id}',
                marker=dict(
                    symbol=symbols[class_id % len(symbols)],
                    size=12,
                    color=colors[class_id % len(colors)],
                    line=dict(width=1, color='black')
                )
            ))

    fig.add_trace(go.Scatter(
        x=list(pred_cols), y=list(pred_rows),
        mode='lines+markers', name='Predicted Path',
        line=dict(color='white', width=3),
        marker=dict(size=4, color='white')
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=list(gt_path_cols), y=list(gt_path_rows),
        mode='lines+markers', name='GT Path',
        line=dict(color='white', width=3),
        marker=dict(size=4, color='white')
    ), row=1, col=1)

    fig.update_layout(height=400, width=1200, title_text="Costmap Comparison")
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange="reversed")

    fig.show()

def sample_ddpm_with_film(model, cond, betas, alphas, alpha_bar, gammas, film_betas, device="cuda", num_steps=50):
    """
    DDPM sampling with FiLM modulation.
    
    Note: 'film_betas' is the FiLM shift parameter, renamed to avoid collision
    with diffusion 'betas'.
    """
    B = cond.shape[0]
    H, W = cond.shape[2], cond.shape[3]
    
    # Start from noise
    x = torch.randn(B, 1, H, W, device=device)
    
    # Use fewer steps for speed (DDIM-style skip)
    step_size = len(betas) // num_steps
    timesteps = list(range(len(betas) - 1, 0, -step_size))
    
    for t in timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # Concatenate condition with noisy x
        model_input = torch.cat([cond, x], dim=1)
        
        # Predict noise WITH FiLM modulation
        with torch.no_grad():
            noise_pred = model(model_input, t_batch, gammas=gammas, betas=film_betas)
        
        # DDPM update step
        alpha = alphas[t]
        alpha_bar_t = alpha_bar[t]
        beta = betas[t]
        
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar_t).sqrt()) * noise_pred)
        x = x + (beta.sqrt()) * noise
    
    # Normalize to [0, 1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    
    return x


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
    
    original_traj = list(zip(original_path_rows, original_path_cols))
    user_traj = list(zip(adjusted_path_rows, adjusted_path_cols))
    
    delta = compute_geometric_delta(original_traj, user_traj, H, W)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=delta,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Cost Delta')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(original_path_cols),
        y=list(original_path_rows),
        mode='lines+markers',
        name='Original Path',
        line=dict(color='black', width=3),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(adjusted_path_cols),
        y=list(adjusted_path_rows),
        mode='lines+markers',
        name='User Adjusted Path',
        line=dict(color='lime', width=3),
        marker=dict(size=4)
    ))
    
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
        title='Cost Delta from User Adjustment<br>(Red = Increase Cost, Blue = Decrease ost)',
        yaxis=dict(autorange='reversed', scaleanchor='x'),
        xaxis=dict(constrain='domain')
    )
    
    fig.show()
    
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
    """Push path points away from class obstacles when within avoidance_radius."""
    path_rows = np.array(path_rows, dtype=np.float32)
    path_cols = np.array(path_cols, dtype=np.float32)
    
    if len(class_1_coords) == 0:
        return path_rows, path_cols
    
    adjusted_rows = path_rows.copy()
    adjusted_cols = path_cols.copy()
    
    for i in range(len(path_rows)):
        pt = np.array([path_rows[i], path_cols[i]])
        
        diffs = class_1_coords - pt
        dists = np.linalg.norm(diffs, axis=1)
        
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        
        if closest_dist < avoidance_radius:
            away = pt - class_1_coords[closest_idx]
            norm = np.linalg.norm(away)
            
            if norm > 1e-6:
                away = away / norm
                adjusted_rows[i] += away[0] * push_distance
                adjusted_cols[i] += away[1] * push_distance
    
    return adjusted_rows, adjusted_cols


def visualize_comparison_with_film(
    ground_truth,
    prediction,
    modulated_prediction,
    obstacles,
    goal,
    adj_rows,
    adj_cols,
    pred_rows,
    pred_cols
):
    """Visualize GT, original prediction, and FiLM-modulated prediction."""
    start_pos = (5, 5)
    goal = goal.numpy()
    goal_pos = (int(goal[0][0]), int(goal[0][1]))
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Ground Truth", "Original Prediction", "FiLM Modulated"),
        horizontal_spacing=0.05
    )
    
    # Ground truth path
    gt_indices, _ = route_through_array(ground_truth, start_pos, goal_pos, fully_connected=True, geometric=True)
    gt_path_rows, gt_path_cols = zip(*gt_indices)
    
    # Modulated prediction path
    mod_indices, _ = route_through_array(modulated_prediction, start_pos, goal_pos, fully_connected=True, geometric=True)
    mod_rows, mod_cols = zip(*mod_indices)
    
    # Heatmaps
    fig.add_trace(go.Heatmap(z=ground_truth, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=prediction, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=modulated_prediction, colorscale='Viridis', zmin=0, zmax=1, showscale=True), row=1, col=3)
    
    # Obstacles on all plots
    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']
    for class_id, coords in obstacles.items():
        if len(coords) > 0:
            for col in [1, 2, 3]:
                fig.add_trace(go.Scatter(
                    x=coords[:, 1], y=coords[:, 0],
                    mode='markers', name=f'Class {class_id}' if col == 1 else None,
                    showlegend=(col == 1),
                    marker=dict(symbol=symbols[class_id % len(symbols)], size=10,
                                color=colors[class_id % len(colors)], line=dict(width=1, color='black'))
                ), row=1, col=col)
    
    # Paths
    fig.add_trace(go.Scatter(x=list(gt_path_cols), y=list(gt_path_rows), mode='lines+markers',
                             name='GT Path', line=dict(color='white', width=3), marker=dict(size=4)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=list(pred_cols), y=list(pred_rows), mode='lines+markers',
                             name='Original Path', line=dict(color='white', width=3), marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(adj_cols), y=list(adj_rows), mode='lines+markers',
                             name='User Edit', line=dict(color='lime', width=2, dash='dash'), marker=dict(size=3)), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=list(mod_cols), y=list(mod_rows), mode='lines+markers',
                             name='Modulated Path', line=dict(color='cyan', width=3), marker=dict(size=4)), row=1, col=3)
    
    fig.update_layout(height=400, width=1400, title_text="Diffusion Costmap: Before and After FiLM Modulation")
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange="reversed")
    
    fig.show()

"""
def run_full_pipeline(model, film_generator, dataset, args, device="cuda"):
    
    model.eval()
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    for batch_idx, (cond, x0, obstacles, goal) in enumerate(dataset):
        if batch_idx == 0 or batch_idx == 4:
            continue
        
        np_map = {}
        cond = cond.to(device)
        x0 = x0.to(device)
        
        for class_id, coord_list in obstacles.items():
            np_map[class_id] = np.array([[pair[0].item(), pair[1].item()] for pair in coord_list])
        
        print(f"\n{'='*60}")
        print(f"Sample {batch_idx}")
        print(f"Obstacles: {np_map}")
        
        # Step 1: Generate initial cost map (no FiLM)
        with torch.no_grad():
            generated_maps = sample_ddpm_with_cond(
                model, cond, diff_betas, alphas, alpha_bar, device=device
            )
        
        map_pred = generated_maps[0, 0].cpu().numpy()
        map_gt = (x0[0, 0].cpu().numpy() + 1.0) / 2.0
        
        # Step 2: Plan path on original prediction
        start_pos = (5, 5)
        goal_np = goal.numpy()
        goal_pos = (int(goal_np[0][0]), int(goal_np[0][1]))
        
        pred_indices, _ = route_through_array(map_pred, start_pos, goal_pos, fully_connected=True, geometric=True)
        pred_rows, pred_cols = zip(*pred_indices)
        
        # Step 3: Simulate user adjustment (for class 0)
        adj_rows, adj_cols = pred_rows, pred_cols  # Default: no adjustment
        target_class = None
        
        for class_id, coords in np_map.items():
            if class_id == 0 and len(coords) > 0:  # Adjust for class 0
                adj_rows, adj_cols = add_user_adjustments(pred_rows, pred_cols, coords)
                target_class = class_id
                break
        
        if target_class is not None:
            print(f"User adjusted trajectory to avoid class {target_class}")
            
            # Step 4: Visualize the cost delta
            delta = visualize_cost_delta(
                pred_rows, pred_cols,
                adj_rows, adj_cols,
                H=map_gt.shape[0],
                W=map_gt.shape[1],
                obstacles=np_map
            )
            
            # Step 5: Fine-tune FiLM based on edit
            print("Fine-tuning FiLM generator...")
            gammas, film_betas = finetune_film_on_edit(
                model=model,
                film_generator=film_generator,
                semantic_map=cond[:, :NUM_CLASSES],
                original_traj=list(zip(pred_rows, pred_cols)),
                user_traj=list(zip(adj_rows, adj_cols)),
                cost_map_gt=None,
                num_steps=20,
                lr=0.01,
                device=device
            )
            
            print(f"Learned FiLM params:")
            print(f"  Gammas: {gammas}")
            print(f"  Betas: {film_betas}")
            
            # Step 6: Re-generate with FiLM modulation
            print("Regenerating cost map with FiLM modulation...")
            with torch.no_grad():
                modulated_maps = sample_ddpm_with_film(
                    model, cond, diff_betas, alphas, alpha_bar,
                    gammas=gammas, film_betas=film_betas,
                    device=device
                )
            
            map_modulated = modulated_maps[0, 0].cpu().numpy()
            
            # Step 7: Visualize comparison
            visualize_comparison_with_film(
                ground_truth=map_gt,
                prediction=map_pred,
                modulated_prediction=map_modulated,
                obstacles=np_map,
                goal=goal,
                adj_rows=adj_rows,
                adj_cols=adj_cols,
                pred_rows=pred_rows,
                pred_cols=pred_cols
            )
            
            # Stats
            print(f"Original prediction - min: {map_pred.min():.4f}, max: {map_pred.max():.4f}")
            print(f"Modulated prediction - min: {map_modulated.min():.4f}, max: {map_modulated.max():.4f}")
            print(f"Difference norm: {np.linalg.norm(map_modulated - map_pred):.4f}")
        
        else:
            print("No class 0 obstacles found, skipping FiLM adaptation")

"""

from film import GPRFiLMManager, apply_gpr_film_modulation
from irl_finetune import finetune_gpr_film_on_edit
"""
def run_full_pipeline(model, dataset, args, device="cuda"):
    model.eval()
    
    # Initialize GPR-FiLM manager (replaces neural FiLM generator)
    gpr_film_manager = GPRFiLMManager(num_classes=NUM_CLASSES, grid_size=args.img_size)
    
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    for batch_idx, (cond, x0, obstacles, goal) in enumerate(dataset):
        if batch_idx == 0 or batch_idx == 4:
            continue
        
        np_map = {}
        cond = cond.to(device)
        x0 = x0.to(device)
        
        for class_id, coord_list in obstacles.items():
            np_map[class_id] = np.array([[pair[0].item(), pair[1].item()] for pair in coord_list])
        
        print(f"\n{'='*60}")
        print(f"Sample {batch_idx}")
        
        # Step 1: Generate base costmap (no FiLM)
        with torch.no_grad():
            generated_maps = sample_ddpm_with_cond(
                model, cond, diff_betas, alphas, alpha_bar, device=device
            )
        
        map_pred = generated_maps[0, 0].cpu().numpy()
        map_gt = (x0[0, 0].cpu().numpy() + 1.0) / 2.0
        H, W = map_pred.shape
        
        # Step 2: Plan path on original prediction
        start_pos = (5, 5)
        goal_np = goal.numpy()
        goal_pos = (int(goal_np[0][0]), int(goal_np[0][1]))
        
        pred_indices, _ = route_through_array(map_pred, start_pos, goal_pos, fully_connected=True, geometric=True)
        pred_rows, pred_cols = zip(*pred_indices)
        
        # Step 3: Simulate user adjustment (for class 0)
        adj_rows, adj_cols = pred_rows, pred_cols
        target_class = None
        
        for class_id, coords in np_map.items():
            if class_id == 0 and len(coords) > 0:
                adj_rows, adj_cols = add_user_adjustments(pred_rows, pred_cols, coords)
                target_class = class_id
                break
        
        if target_class is not None:
            print(f"User adjusted trajectory to avoid class {target_class}")
            
            # Step 4: Visualize cost delta
            delta = visualize_cost_delta(
                pred_rows, pred_cols,
                adj_rows, adj_cols,
                H=H, W=W,
                obstacles=np_map
            )
            
            # Step 5: Update GPR-FiLM (instant - no gradient training!)
            class_activations = cond[0, :NUM_CLASSES].cpu().numpy()
            
            result = finetune_gpr_film_on_edit(
                gpr_film_manager=gpr_film_manager,
                current_costmap=map_pred,
                class_activations=class_activations,
                original_traj=list(zip(pred_rows, pred_cols)),
                user_traj=list(zip(adj_rows, adj_cols)),
                affected_class=target_class,
                H=H, W=W,
                verbose=True
            )
            
            # Step 6: Apply GPR-FiLM modulation to get new costmap
            map_modulated = apply_gpr_film_modulation(
                base_costmap=map_pred,
                class_activations=class_activations,
                gpr_film_manager=gpr_film_manager
            )
            
            # Step 7: Visualize comparison
            visualize_comparison_with_film(
                ground_truth=map_gt,
                prediction=map_pred,
                modulated_prediction=map_modulated,
                obstacles=np_map,
                goal=goal,
                adj_rows=adj_rows,
                adj_cols=adj_cols,
                pred_rows=pred_rows,
                pred_cols=pred_cols
            )
            
            print(f"Original cost range: [{map_pred.min():.3f}, {map_pred.max():.3f}]")
            print(f"Modulated cost range: [{map_modulated.min():.3f}, {map_modulated.max():.3f}]")
"""

def run_full_pipeline(model, dataset, args, device="cuda"):
    model.eval()
    
    # Initialize GPR-FiLM manager ONCE - persists across all scenes
    gpr_film_manager = GPRFiLMManager(num_classes=NUM_CLASSES, grid_size=args.img_size)
    
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    for batch_idx, (cond, x0, obstacles, goal) in enumerate(dataset):
        np_map = {}
        cond = cond.to(device)
        x0 = x0.to(device)
        
        for class_id, coord_list in obstacles.items():
            np_map[class_id] = np.array([[pair[0].item(), pair[1].item()] for pair in coord_list])
        
        print(f"\n{'='*60}")
        print(f"Scene {batch_idx}")
        print(f"GPR observations so far: {sum(len(obs) for obs in gpr_film_manager.observations.values())}")
        
        # Step 1: Generate base costmap
        with torch.no_grad():
            generated_maps = sample_ddpm_with_cond(
                model, cond, diff_betas, alphas, alpha_bar, device=device
            )
        
        map_pred = generated_maps[0, 0].cpu().numpy()
        map_gt = (x0[0, 0].cpu().numpy() + 1.0) / 2.0
        H, W = map_pred.shape
        class_activations = cond[0, :NUM_CLASSES].cpu().numpy()
        
        # Step 2: Apply EXISTING learned preferences (from previous scenes)
        if sum(len(obs) for obs in gpr_film_manager.observations.values()) > 0:
            print("Applying learned preferences from previous scenes...")
            map_with_prefs = apply_gpr_film_modulation(
                base_costmap=map_pred,
                class_activations=class_activations,
                gpr_film_manager=gpr_film_manager
            )
        else:
            map_with_prefs = map_pred
            print("No learned preferences yet - using base costmap")
        
        # Step 3: Plan path on costmap WITH preferences
        start_pos = (5, 5)
        goal_np = goal.numpy()
        goal_pos = (int(goal_np[0][0]), int(goal_np[0][1]))
        
        pred_indices, _ = route_through_array(map_with_prefs, start_pos, goal_pos, fully_connected=True, geometric=True)
        pred_rows, pred_cols = zip(*pred_indices)
        
        # Step 4: Simulate user edit (or get real user input)
        user_made_edit = False
        adj_rows, adj_cols = pred_rows, pred_cols
        target_class = None
        
        for class_id, coords in np_map.items():
            if class_id == 0 and len(coords) > 0:  # Example: always edit for class 0
                adj_rows, adj_cols = add_user_adjustments(pred_rows, pred_cols, coords)
                
                # Check if user actually changed the trajectory
                orig_arr = np.array(list(zip(pred_rows, pred_cols)))
                adj_arr = np.array(list(zip(adj_rows, adj_cols)))
                displacement = np.linalg.norm(orig_arr - adj_arr, axis=1).mean()
                
                if displacement > 1.0:  # Threshold: only learn if significant edit
                    user_made_edit = True
                    target_class = class_id
                    print(f"User edited trajectory (avg displacement: {displacement:.1f})")
                break
        
        # Step 5: If user made an edit, UPDATE the GPR-FiLM
        if user_made_edit and target_class is not None:
            print(f"Learning from user edit for class {target_class}...")
            
            # Visualize the edit
            delta = visualize_cost_delta(
                pred_rows, pred_cols,
                adj_rows, adj_cols,
                H=H, W=W,
                obstacles=np_map
            )
            
            # Update GPR with this observation
            result = finetune_gpr_film_on_edit(
                gpr_film_manager=gpr_film_manager,
                current_costmap=map_with_prefs,
                class_activations=class_activations,
                original_traj=list(zip(pred_rows, pred_cols)),
                user_traj=list(zip(adj_rows, adj_cols)),
                affected_class=target_class,
                H=H, W=W,
                verbose=True
            )
            
            # Re-apply modulation with updated GPR
            map_modulated = apply_gpr_film_modulation(
                base_costmap=map_pred,
                class_activations=class_activations,
                gpr_film_manager=gpr_film_manager
            )
            
            # Visualize before/after
            visualize_comparison_with_film(
                ground_truth=map_gt,
                prediction=map_pred,
                modulated_prediction=map_modulated,
                obstacles=np_map,
                goal=goal,
                adj_rows=adj_rows,
                adj_cols=adj_cols,
                pred_rows=pred_rows,
                pred_cols=pred_cols
            )
        else:
            print("No user edit - preferences unchanged")
            
            # Still visualize the result with current preferences
            visualize_comparison(map_gt, map_with_prefs, np_map, goal)
        
        # Print current GPR state
        print("\nCurrent learned preferences:")
        preds = gpr_film_manager.get_all_predictions()
        for cid in range(NUM_CLASSES):
            p = preds[cid]
            if p['n_observations'] > 0:
                print(f"  Class {cid}: γ={p['gamma']:.3f}, β={p['beta']:.3f} (n={p['n_observations']})")
            else:
                print(f"  Class {cid}: no data")
def seed_env(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Diffusion Hyperparameters
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=8)
    
    # Evaluation settings
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-obs-per-class", type=int, default=1)
    parser.add_argument("--min-obs-per-class", type=int, default=1)
    parser.add_argument("--min-total-obs", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    
    # FiLM settings
    parser.add_argument("--film-hidden-dim", type=int, default=64)
    
    args = parser.parse_args()
    
    seed_env(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset
    eval_ds = MultiClassCostmapDataset(
        n_samples=args.n_samples,
        H=args.img_size,
        W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False)
    
    # Load UNet (frozen)
    model = UNet(
        in_channels=NUM_CLASSES + 2,
        lora_rank=args.lora_rank,
        num_classes=NUM_CLASSES
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found! Using random weights.")
    
    
   # print(f"FiLM generator params: {sum(p.numel() for p in film_generator.parameters())}")
    
    # Run
    run_full_pipeline(model, eval_dl, args, device=device)
