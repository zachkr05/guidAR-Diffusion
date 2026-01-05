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
#from film_nn import NeuralFiLMManager, extract_trajectory_features
from film_nn import SingleShotFiLMManager
from DataGenerator.dataset_costmap import make_class_occupancy_maps, make_goal_map
import scipy.ndimage as ndi

def visualize_user_adjustment_overlay(
    costmap,
    orig_rows,
    orig_cols,
    adj_rows,
    adj_cols,
    obstacles,
    title="User Trajectory Adjustment"
):
    """
    Visualizes the specific edit made by the user by drawing displacement vectors
    between the original planner output and the user's adjusted path.
    """
    fig = go.Figure()

    # 1. The Costmap Background
    fig.add_trace(go.Heatmap(
        z=costmap, 
        colorscale='Viridis', 
        zmin=0, zmax=1, 
        showscale=False,
        hoverinfo='skip'
    ))

    # 2. Obstacles
    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']
    
    for class_id, coords in obstacles.items():
        if len(coords) > 0:
            coords = np.array(coords)
            fig.add_trace(go.Scatter(
                x=coords[:, 1], y=coords[:, 0],
                mode='markers',
                name=f'Class {class_id}',
                marker=dict(
                    symbol=symbols[class_id % len(symbols)],
                    size=10,
                    color=colors[class_id % len(colors)],
                    line=dict(width=1, color='black')
                )
            ))

    # 3. The Original Path (Ghosted)
    fig.add_trace(go.Scatter(
        x=orig_cols, y=orig_rows,
        mode='lines',
        name='Original Path',
        line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dot'),
        hoverinfo='skip'
    ))

    # 4. The Adjusted Path (User Intent)
    fig.add_trace(go.Scatter(
        x=adj_cols, y=adj_rows,
        mode='lines+markers',
        name='User Adjusted',
        line=dict(color='lime', width=3),
        marker=dict(size=4, color='lime')
    ))

    # 5. Displacement Vectors (Visualizing the "Push")
    # We draw lines connecting the old point to the new point
    # We create a single trace with NaN breaks to draw multiple independent lines
    
    vec_x = []
    vec_y = []
    
    # Convert to numpy for easier handling
    o_r, o_c = np.array(orig_rows), np.array(orig_cols)
    a_r, a_c = np.array(adj_rows), np.array(adj_cols)
    
    # Filter for points where significant movement happened to reduce clutter
    dists = np.sqrt((o_r - a_r)**2 + (o_c - a_c)**2)
    mask = dists > 0.5 # Only show vectors for movements > 0.5 pixels
    
    for i in np.where(mask)[0]:
        # Start point (Original)
        vec_x.append(o_c[i])
        vec_y.append(o_r[i])
        # End point (Adjusted)
        vec_x.append(a_c[i])
        vec_y.append(a_r[i])
        # Break line
        vec_x.append(None)
        vec_y.append(None)

    if len(vec_x) > 0:
        fig.add_trace(go.Scatter(
            x=vec_x, y=vec_y,
            mode='lines',
            name='Adjustment Force',
            line=dict(color='magenta', width=2),
            hoverinfo='skip',
            showlegend=True
        ))

    fig.update_layout(
        height=600, width=800,
        title_text=title,
        plot_bgcolor='black',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange="reversed") # Important for image coords

    fig.show()

def sample_ddpm_with_strong_film(
    model, 
    cond, 
    betas, 
    alphas, 
    alpha_bar, 
    class_gammas, 
    class_betas, 
    device="cuda",
    num_steps=50,
    film_strength=1.0,
    film_start_step=0.8
):
    """
    Robust FiLM injection using 'Prediction & Re-noising'.
    This avoids signal collapse (gray images) by enforcing a valid noise level.
    """
    B = cond.shape[0]
    H, W = cond.shape[2], cond.shape[3]
    num_classes = NUM_CLASSES
    
    # 1. Setup Regions
    class_activations = cond[:, :num_classes, :, :].cpu().numpy()
    class_regions = {}
    for class_id in range(num_classes):
        class_act = class_activations[0, class_id]
        if class_act.max() > 0:
            # Smooth region
            region = ndi.maximum_filter(class_act, size=5)
            region = ndi.gaussian_filter(region.astype(np.float32), sigma=1.0)
            region = region / (region.max() + 1e-8)
            class_regions[class_id] = region
        else:
            class_regions[class_id] = np.zeros((H, W), dtype=np.float32)
            
    # 2. Start Noise
    x = torch.randn(B, 1, H, W, device=device)
    
    # 3. Schedule
    # Use simple linear skipping for deployment speed
    step_size = max(len(betas) // num_steps, 1)
    # Create list of [T, T-step, ..., step, 0]
    timesteps = list(range(len(betas) - 1, 0, -step_size))
    total_steps = len(timesteps)
    film_start_idx = int(total_steps * (1 - film_start_step))
    
    for step_idx, t in enumerate(timesteps):
        # A. Predict x0
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x_in = torch.cat([cond, x], dim=1)
        
        with torch.no_grad():
            noise_pred = model(x_in, t_batch)
            
        # Recover x0 from noise
        alpha_bar_t = alpha_bar[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        
        pred_x0 = (x - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
        
        # B. Inject FiLM into x0
        if step_idx >= film_start_idx:
            # Convert to [0,1] for manipulation
            x0_np = pred_x0[0, 0].cpu().numpy()
            x0_01 = (x0_np + 1) / 2.0
            x0_01 = np.clip(x0_01, 0, 1)
            
            # Progress ramp
            progress = (step_idx - film_start_idx) / max(total_steps - film_start_idx, 1)
            current_strength = film_strength * progress
            
            total_delta = np.zeros_like(x0_01)
            
            for class_id in range(num_classes):
                gamma = class_gammas.get(class_id, 0.0)
                beta = class_betas.get(class_id, 0.0)
                if abs(gamma) < 0.01 and abs(beta) < 0.01: continue
                
                # Apply affine shift
                region = class_regions[class_id]
                delta = (x0_01 * gamma + beta) * region * current_strength
                total_delta += delta
                
            # Apply and Re-Normalize to [-1, 1]
            x0_new_01 = np.clip(x0_01 + total_delta, 0, 1)
            x0_new_np = x0_new_01 * 2.0 - 1.0
            
            # Update estimate
            pred_x0 = torch.from_numpy(x0_new_np).float().unsqueeze(0).unsqueeze(0).to(device)

        # C. Re-Noise to next step (t-step_size)
        # Instead of posterior calculation, we simply add the correct amount 
        # of noise for the TARGET timestep.
        
        next_t = t - step_size
        
        if next_t <= 0:
            # Final step: output the clean image directly
            x = pred_x0
            break
        else:
            # Forward Process: q(x_{t-1} | x_0)
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * noise
            
            alpha_bar_next = alpha_bar[next_t]
            noise = torch.randn_like(x)
            
            x = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1 - alpha_bar_next) * noise

    x = (x + 1) / 2
    return x.clamp(0, 1)

def sample_ddpm_with_film_injection(
    model, 
    cond, 
    betas, 
    alphas, 
    alpha_bar, 
    class_gammas,  # Dict[int, float]
    class_betas,   # Dict[int, float]  
    device="cuda",
    num_steps=50
):
    """
    Sample with FiLM applied at each denoising step.
    """
    B = cond.shape[0]
    H, W = cond.shape[2], cond.shape[3]
    num_classes = NUM_CLASSES
    
    class_activations = cond[:, :num_classes, :, :].cpu().numpy()
    
    x = torch.randn(B, 1, H, W, device=device)
    
    step_size = max(len(betas) // num_steps, 1)
    timesteps = list(range(len(betas) - 1, 0, -step_size))
    
    for t in timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x_in = torch.cat([cond, x], dim=1)
        
        with torch.no_grad():
            noise_pred = model(x_in, t_batch)
        
        # === FiLM INJECTION ===
        noise_np = noise_pred[0, 0].cpu().numpy()
        
        for class_id in range(num_classes):
            gamma = class_gammas.get(class_id, 0.0)
            beta = class_betas.get(class_id, 0.0)
            
            if abs(gamma) < 0.01 and abs(beta) < 0.01:
                continue
            
            class_act = class_activations[0, class_id]
            class_region = ndi.maximum_filter(class_act, size=12)
            class_region = ndi.gaussian_filter(class_region.astype(np.float32), sigma=3.0)
            
            noise_np = noise_np * (1 + gamma * class_region) + beta * class_region
        
        noise_pred = torch.from_numpy(noise_np).float().unsqueeze(0).unsqueeze(0).to(device)
        # === END FiLM INJECTION ===
        
        alpha = alphas[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = betas[t]
        
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = (1 / alpha.sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred)
        x = x + (beta_t.sqrt()) * noise
    
    x = (x + 1) / 2
    return x.clamp(0, 1)

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
from film import GPRFiLMManager, apply_gpr_film_modulation
from irl_finetune import finetune_gpr_film_on_edit

def run_full_pipeline(model, dataset, args, device="cuda"):
    model.eval()
    
    # Single-shot FiLM manager
    film_manager = SingleShotFiLMManager(
        num_classes=NUM_CLASSES,
        grid_size=args.img_size,
        ema_decay=0.5,
        device=device
    )
    
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    # =========================================================================
    # TRAINING PHASE: Learn from user edits
    # =========================================================================
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    for batch_idx, (cond, x0, obstacles, goal) in enumerate(dataset):
        np_map = {}
        cond = cond.to(device)
        x0 = x0.to(device)
        
        for class_id, coord_list in obstacles.items():
            np_map[class_id] = np.array([[pair[0].item(), pair[1].item()] for pair in coord_list])
        
        print(f"\n--- Scene {batch_idx} ---")
        print(f"Observations so far: {film_manager.get_observation_count()}")
        
        #seed_env(1000 + deploy_idx)
        # Generate base costmap
        with torch.no_grad():
            generated_maps = sample_ddpm_with_cond(
                model, cond, diff_betas, alphas, alpha_bar, device=device
            )
        
        map_pred = generated_maps[0, 0].cpu().numpy()
        map_gt = (x0[0, 0].cpu().numpy() + 1.0) / 2.0
        H, W = map_pred.shape
        class_activations = cond[0, :NUM_CLASSES].cpu().numpy()
        
        # Apply current preferences BEFORE planning
        if film_manager.get_observation_count() > 0:
            map_with_prefs = film_manager.apply_to_costmap(map_pred, class_activations, strength=1.0)
        else:
            map_with_prefs = map_pred
        
        # Plan path
        start_pos = (5, 5)
        goal_np = goal.numpy()
        goal_pos = (int(goal_np[0][0]), int(goal_np[0][1]))
        
        pred_indices, _ = route_through_array(map_with_prefs, start_pos, goal_pos, fully_connected=True, geometric=True)
        pred_rows, pred_cols = zip(*pred_indices)
        
        # Simulate user edit
        user_made_edit = False
        adj_rows, adj_cols = pred_rows, pred_cols
        target_class = None
        
        for class_id, coords in np_map.items():
            if class_id == 0 and len(coords) > 0:
                adj_rows, adj_cols = add_user_adjustments(pred_rows, pred_cols, coords)
                
                orig_arr = np.array(list(zip(pred_rows, pred_cols)))
                adj_arr = np.array(list(zip(adj_rows, adj_cols)))
                displacement = np.linalg.norm(orig_arr - adj_arr, axis=1).mean()
                
                if displacement > 1.0:
                    user_made_edit = True
                    target_class = class_id
                break
        #visualize_comparison(map_gt, map_pred, np_map, goal)  
        if user_made_edit and target_class is not None:
            print(f"User edit for class {target_class}")
            visualize_user_adjustment_overlay(
                costmap=map_pred,
                orig_rows=pred_rows,
                orig_cols=pred_cols,
                adj_rows=adj_rows,
                adj_cols=adj_cols,
                obstacles=np_map,
                title=f"User avoiding Class {target_class}"
            )    
            from irl_agent import compute_geometric_delta
            delta = compute_geometric_delta(
                list(zip(pred_rows, pred_cols)),
                list(zip(adj_rows, adj_cols)),
                H, W
            )
            target_costmap = np.clip(map_pred + delta, 0, 1)
             
            # Single-shot update
            result = film_manager.update_from_edit(
                semantic_map=class_activations,
                current_costmap=map_pred,
                target_costmap=target_costmap,
                affected_class=target_class,
                verbose=True
            )
        else:
            print("No user edit")
    
    # =========================================================================
    # DEPLOYMENT PHASE: Test on new random scenes
    # =========================================================================
    print("\n" + "="*60)
    print("DEPLOYMENT PHASE - Testing learned preferences on new scenes")
    print("="*60)
    print(f"\nFinal learned FiLM params:")
    print(film_manager.get_params_summary())
    
    # Generate new random scenes for deployment
    n_deploy_scenes = 5
    
    learned_gammas = film_manager.gammas
    learned_betas = film_manager.betas

    for deploy_idx in range(n_deploy_scenes):
        print(f"\n--- Deployment Scene {deploy_idx + 1}/{n_deploy_scenes} ---")
        
        # Generate a random scene
        scene = generate_random_scene(
            H=args.img_size,
            W=args.img_size,
            seed=1000 + deploy_idx  # Different seed than training
        )
        
        # Build conditioning tensor
        from DataGenerator.dataset_costmap import make_goal_map
        goal_map = make_goal_map(args.img_size, args.img_size, scene['goal'])
        cond_np = np.concatenate([scene['class_activations'], goal_map[None, :, :]], axis=0)
        cond = torch.from_numpy(cond_np).float().unsqueeze(0).to(device)
        
        # Generate base costmap
        with torch.no_grad():
#            generated_maps = sample_ddpm_with_cond(
 #               model, cond, diff_betas, alphas, alpha_bar, device=device
  #          )
            generated_maps = sample_ddpm_with_strong_film(
                model=model,
                cond=cond,
                betas=diff_betas,
                alphas=alphas,
                alpha_bar=alpha_bar,
                class_gammas=learned_gammas,  # <--- Learned Scale
                class_betas=learned_betas,    # <--- Learned Shift
                device=device,
                num_steps=50,          # Faster sampling for deployment
                film_strength=0.5,     # Full strength
                film_start_step=0.8    # Inject in last 20% of steps (Structure is set, texture is mutable)
            )       
        map_pred = generated_maps[0, 0].cpu().numpy()
        
        # Apply learned FiLM preferences
        map_with_prefs = film_manager.apply_to_costmap(
            map_pred, 
            scene['class_activations'], 
            strength=1.0
        )
        
        # Plan paths on both
        start_pos = scene['start']
        goal_pos = scene['goal']
        
        path_before, _ = route_through_array(map_pred, start_pos, goal_pos, fully_connected=True, geometric=True)
        path_after, _ = route_through_array(map_with_prefs, start_pos, goal_pos, fully_connected=True, geometric=True)
        
        # Visualize deployment result
        visualize_deployment_scene(
            base_costmap=map_pred,
            modulated_costmap=map_with_prefs,
            path_before=path_before,
            path_after=path_after,
            scene=scene,
            deploy_idx=deploy_idx
        )
        
        # Print stats
        max_change = np.abs(map_with_prefs - map_pred).max()
        mean_change = np.abs(map_with_prefs - map_pred).mean()
        print(f"Costmap change: max={max_change:.3f}, mean={mean_change:.3f}")


def generate_random_scene(H: int, W: int, seed: int = None) -> dict:
    """Generate a random scene for deployment testing."""
    if seed is not None:
        np.random.seed(seed)
    
    # Random start and goal
    start = (np.random.randint(H//2, H-5), np.random.randint(5, W//2))
    goal = (np.random.randint(5, H//2), np.random.randint(W//2, W-5))
    
    # Random obstacles per class
    obstacles_by_class = {}
    for class_id in range(NUM_CLASSES):
        n_obs = np.random.randint(1, 3)
        obstacles = []
        for _ in range(n_obs):
            r = np.random.randint(10, H-10)
            c = np.random.randint(10, W-10)
            obstacles.append((r, c))
        obstacles_by_class[class_id] = obstacles
    
    # Build class activation maps
    from DataGenerator.dataset_costmap import make_class_occupancy_maps
    class_activations = make_class_occupancy_maps(H, W, obstacles_by_class)
    
    return {
        'start': start,
        'goal': goal,
        'obstacles_by_class': obstacles_by_class,
        'class_activations': class_activations,
        'H': H,
        'W': W
    }


def visualize_deployment_scene(
    base_costmap: np.ndarray,
    modulated_costmap: np.ndarray,
    path_before: list,
    path_after: list,
    scene: dict,
    deploy_idx: int
):
    """Visualize a deployment scene comparing before/after FiLM."""
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Base Costmap", "With Learned Preferences", "Difference"),
        horizontal_spacing=0.05
    )
    
    # Base costmap
    fig.add_trace(go.Heatmap(
        z=base_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False
    ), row=1, col=1)
    
    # Modulated costmap
    fig.add_trace(go.Heatmap(
        z=modulated_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False
    ), row=1, col=2)
    
    # Difference
    diff = modulated_costmap - base_costmap
    max_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
    fig.add_trace(go.Heatmap(
        z=diff, colorscale='RdBu_r', zmin=-max_diff, zmax=max_diff, showscale=True
    ), row=1, col=3)
    
    # Obstacles
    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']
    
    for class_id, obs_list in scene['obstacles_by_class'].items():
        if len(obs_list) > 0:
            coords = np.array(obs_list)
            for col in [1, 2, 3]:
                fig.add_trace(go.Scatter(
                    x=coords[:, 1], y=coords[:, 0],
                    mode='markers',
                    name=f'Class {class_id}' if col == 1 else None,
                    showlegend=(col == 1),
                    marker=dict(
                        symbol=symbols[class_id % len(symbols)],
                        size=12,
                        color=colors[class_id % len(colors)],
                        line=dict(width=1, color='black')
                    )
                ), row=1, col=col)
    
    # Paths
    path_before = np.array(path_before)
    path_after = np.array(path_after)
    
    fig.add_trace(go.Scatter(
        x=path_before[:, 1], y=path_before[:, 0],
        mode='lines+markers', name='Path (Before)',
        line=dict(color='white', width=3), marker=dict(size=3)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=path_after[:, 1], y=path_after[:, 0],
        mode='lines+markers', name='Path (After)',
        line=dict(color='cyan', width=3), marker=dict(size=3)
    ), row=1, col=2)
    
    # Start and goal markers
    for col in [1, 2]:
        fig.add_trace(go.Scatter(
            x=[scene['start'][1]], y=[scene['start'][0]],
            mode='markers', name='Start' if col == 1 else None,
            showlegend=(col == 1),
            marker=dict(symbol='circle', size=15, color='cyan', line=dict(width=2, color='black'))
        ), row=1, col=col)
        
        fig.add_trace(go.Scatter(
            x=[scene['goal'][1]], y=[scene['goal'][0]],
            mode='markers', name='Goal' if col == 1 else None,
            showlegend=(col == 1),
            marker=dict(symbol='star', size=18, color='yellow', line=dict(width=2, color='black'))
        ), row=1, col=col)
    
    fig.update_layout(
        height=450,
        width=1400,
        title_text=f"Deployment Scene {deploy_idx + 1}: Learned Preferences Applied"
    )
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange='reversed')
    
    fig.show()


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
