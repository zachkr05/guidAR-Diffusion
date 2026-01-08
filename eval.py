"""
Evaluation script with Class-Isolated Neural Preference Adapter.

Uses the clean channel stack:
- Base Costmap, Start Map, Goal Map (global context)
- EDF, Frontalness, Orientation (per class)

This ensures corrections for one class don't affect other classes.
"""

import argparse
import os
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.graph import route_through_array

# --- Local Imports ---
from diffusion_utils import schedule_betas
from DataGenerator.dataset_costmap import MultiClassCostmapDataset, get_cond_channels
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES, NUM_ORIENTATIONS, orientation_to_vector
from DataGenerator.dataset_costmap import make_class_occupancy_maps, make_goal_map, make_orientation_maps
from UNet.UNet import UNet
from train import sample_ddpm_with_cond

# Import the class-isolated learner
from film import (
    OnlinePreferenceLearner, 
    build_feature_stack,
    compute_edf,
)
from irl_agent import compute_geometric_delta


# =============================================================================
# Core Logic: Edit Sequence Detection
# =============================================================================

def find_edit_sequences(orig_rows, orig_cols, adj_rows, adj_cols, threshold=1.0, gap_tolerance=5):
    """Identifies distinct sequences of edits along a path."""
    orig_arr = np.array(list(zip(orig_rows, orig_cols)))
    adj_arr = np.array(list(zip(adj_rows, adj_cols)))
    
    displacements = np.linalg.norm(adj_arr - orig_arr, axis=1)
    edit_indices = np.where(displacements > threshold)[0]
    
    if len(edit_indices) == 0:
        return []

    sequences = []
    current_seq = [edit_indices[0]]
    
    for i in range(1, len(edit_indices)):
        if edit_indices[i] - edit_indices[i-1] <= gap_tolerance:
            current_seq.append(edit_indices[i])
        else:
            sequences.append(current_seq)
            current_seq = [edit_indices[i]]
    sequences.append(current_seq)
    
    results = []
    for seq_indices in sequences:
        indices = np.array(seq_indices)
        sub_orig = orig_arr[indices]
        sub_adj = adj_arr[indices]
        sub_disp = displacements[indices]
        
        total_mass = np.sum(sub_disp)
        if total_mass > 0:
            centroid_r = np.sum(sub_adj[:, 0] * sub_disp) / total_mass
            centroid_c = np.sum(sub_adj[:, 1] * sub_disp) / total_mass
            centroid = (centroid_r, centroid_c)
        else:
            centroid = (np.mean(sub_adj[:, 0]), np.mean(sub_adj[:, 1]))
            
        results.append({
            'indices': indices,
            'centroid': centroid,
            'max_displacement': np.max(sub_disp),
            'orig_sub_path': list(map(tuple, sub_orig)),
            'adj_sub_path': list(map(tuple, sub_adj))
        })
        
    return results


# =============================================================================
# Visualization Helpers
# =============================================================================

def add_obstacle_markers_with_orientation(fig, obstacles_by_class, row=None, col=None, show_legend=True):
    """Add obstacle markers with orientation arrows."""
    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']
    arrow_length = 4.0
    
    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue
        color = colors[class_id % len(colors)]
        symbol = symbols[class_id % len(symbols)]
        class_name = OBSTACLE_CLASSES[class_id]['name']
        
        rows_list, cols_list = [], []
        arrow_x, arrow_y = [], []
        
        for obs in obstacles:
            r, c = obs['pos']
            orientation = obs['orientation']
            rows_list.append(r)
            cols_list.append(c)
            dx, dy = orientation_to_vector(orientation)
            arrow_x.extend([c, c + dx * arrow_length, None])
            arrow_y.extend([r, r - dy * arrow_length, None])
        
        marker_trace = go.Scatter(
            x=cols_list, y=rows_list, mode='markers',
            name=f'{class_name}' if show_legend else None,
            showlegend=show_legend,
            marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color='black')),
            legendgroup=f'class_{class_id}'
        )
        
        arrow_trace = go.Scatter(
            x=arrow_x, y=arrow_y, mode='lines', showlegend=False,
            line=dict(color=color, width=2), legendgroup=f'class_{class_id}'
        ) if arrow_x else None
        
        if row and col:
            fig.add_trace(marker_trace, row=row, col=col)
            if arrow_trace:
                fig.add_trace(arrow_trace, row=row, col=col)
        else:
            fig.add_trace(marker_trace)
            if arrow_trace:
                fig.add_trace(arrow_trace)


def visualize_feature_stack(features_np, obstacles_by_class, title="Feature Stack"):
    """
    Visualize all channels in the feature stack.
    
    Args:
        features_np: [C, H, W] numpy array
        obstacles_by_class: for overlay
    """
    num_classes = (features_np.shape[0] - 3) // 3
    
    n_cols = 3 + num_classes  # global + one per class
    fig = make_subplots(
        rows=3, cols=n_cols,
        subplot_titles=[
            "Base Costmap", "Start Map", "Goal Map",
        ] + [f"Class {c}" for c in range(num_classes)] + [
            "", "", ""
        ] + [f"Class {c} EDF" for c in range(num_classes)] + [
            "", "", ""
        ] + [f"Class {c} Frontal" for c in range(num_classes)]
    )
    
    # Row 1: Global features
    fig.add_trace(go.Heatmap(z=features_np[0], colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=features_np[1], colorscale='Blues', showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=features_np[2], colorscale='Oranges', showscale=False), row=1, col=3)
    
    # Row 2: EDF per class
    for c in range(num_classes):
        edf_idx = 3 + c * 3
        fig.add_trace(go.Heatmap(z=features_np[edf_idx], colorscale='Blues_r', showscale=False), row=2, col=c+1)
    
    # Row 3: Frontalness per class
    for c in range(num_classes):
        frontal_idx = 3 + c * 3 + 1
        fig.add_trace(go.Heatmap(z=features_np[frontal_idx], colorscale='Reds', showscale=False), row=3, col=c+1)
    
    fig.update_layout(height=600, width=300*n_cols, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()


def visualize_user_adjustment_overlay(costmap, orig_rows, orig_cols, adj_rows, adj_cols,
                                      obstacles_by_class, edit_sequences, start_pos, goal_pos,
                                      title="User Trajectory Adjustment"):
    """Visualize the user's trajectory correction."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False))
    add_obstacle_markers_with_orientation(fig, obstacles_by_class, show_legend=True)
    
    fig.add_trace(go.Scatter(
        x=orig_cols, y=orig_rows, mode='lines', name='Original Path',
        line=dict(color='rgba(255,255,255,0.5)', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=adj_cols, y=adj_rows, mode='lines+markers', name='User Adjusted',
        line=dict(color='lime', width=3), marker=dict(size=4, color='lime')
    ))
    
    for i, seq in enumerate(edit_sequences):
        cy, cx = seq['centroid']
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text', name=f'Centroid {i+1}',
            marker=dict(symbol='x', size=15, color='magenta', line=dict(width=2, color='white')),
            text=[f"C{i+1}"], textposition="top center", textfont=dict(color='magenta', size=14)
        ))
    
    fig.add_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]], mode='markers', name='Start',
        marker=dict(symbol='circle', size=18, color='cyan', line=dict(width=2, color='white'))
    ))
    
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]], y=[goal_pos[0]], mode='markers', name='Goal',
        marker=dict(symbol='star', size=20, color='yellow', line=dict(width=2, color='black'))
    ))

    fig.update_layout(height=600, width=800, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()


def visualize_irl_target(current_costmap, delta_map, target_costmap,
                         centroid, sub_orig, sub_adj, seq_idx,
                         title="IRL Target Generation"):
    """Visualizes the IRL target generation."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Current Costmap", f"Geometric Delta (Seq {seq_idx})", "Resulting Target Map")
    )
    
    fig.add_trace(go.Heatmap(z=current_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    
    max_val = max(abs(delta_map.min()), abs(delta_map.max()), 0.1)
    fig.add_trace(go.Heatmap(z=delta_map, colorscale='RdBu_r', zmin=-max_val, zmax=max_val, showscale=True), row=1, col=2)
    
    orig_r, orig_c = zip(*sub_orig)
    adj_r, adj_c = zip(*sub_adj)
    
    fig.add_trace(go.Scatter(x=orig_c, y=orig_r, mode='lines', line=dict(color='black', width=1, dash='dot'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=adj_c, y=adj_r, mode='lines', line=dict(color='cyan', width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[centroid[1]], y=[centroid[0]], mode='markers', marker=dict(symbol='x', color='black', size=10), showlegend=False), row=1, col=2)

    fig.add_trace(go.Heatmap(z=target_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=3)

    fig.update_layout(height=400, width=1200, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()


def visualize_deployment_scene(base_costmap, modulated_costmap, path_before, path_after,
                               scene, deploy_idx, features_np=None):
    """Visualize deployment scene with before/after comparison."""
    n_cols = 4 if features_np is not None else 3
    titles = ["Base Costmap", "With Learned Preferences", "Difference"]
    if features_np is not None:
        titles.append("Chair Frontalness")
    
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles)
    
    fig.add_trace(go.Heatmap(z=base_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=modulated_costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    
    diff = modulated_costmap - base_costmap
    max_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
    fig.add_trace(go.Heatmap(z=diff, colorscale='RdBu_r', zmin=-max_diff, zmax=max_diff, showscale=True), row=1, col=3)
    
    if features_np is not None:
        # Show Chair Frontalness (class 0, frontalness is index 4)
        frontal_idx = 3 + 0 * 3 + 1  # Class 0 frontalness
        fig.add_trace(go.Heatmap(z=features_np[frontal_idx], colorscale='Reds', zmin=0, zmax=1, showscale=True), row=1, col=4)
    
    obs = scene['obstacles_by_class']
    for c in range(1, n_cols + 1):
        add_obstacle_markers_with_orientation(fig, obs, row=1, col=c, show_legend=(c == 1))
    
    pb = np.array(path_before)
    pa = np.array(path_after)
    
    fig.add_trace(go.Scatter(x=pb[:, 1], y=pb[:, 0], mode='lines', line=dict(color='white', width=2), name='Before'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pa[:, 1], y=pa[:, 0], mode='lines', line=dict(color='cyan', width=3), name='After'), row=1, col=2)
    
    for c in [1, 2]:
        fig.add_trace(go.Scatter(
            x=[scene['start'][1]], y=[scene['start'][0]], mode='markers',
            marker=dict(symbol='circle', size=15, color='cyan'), showlegend=False
        ), row=1, col=c)
        fig.add_trace(go.Scatter(
            x=[scene['goal'][1]], y=[scene['goal'][0]], mode='markers',
            marker=dict(symbol='star', size=18, color='yellow'), showlegend=False
        ), row=1, col=c)
    
    fig.update_layout(height=400, width=350 * n_cols, title_text=f"Deployment Scene {deploy_idx + 1}")
    fig.update_yaxes(matches='y', autorange='reversed')
    fig.show()


# =============================================================================
# Simulation Helpers
# =============================================================================

def add_user_adjustments_robust(path_rows, path_cols, obstacles_by_class, target_class=0,
                                avoidance_radius=30.0, push_strength=25.0, influence_width=15):
    """Simulates a user 'tugging' the path away from obstacles of a specific class."""
    path_rows = np.array(path_rows, dtype=np.float32)
    path_cols = np.array(path_cols, dtype=np.float32)
    n_points = len(path_rows)
    
    if target_class not in obstacles_by_class or len(obstacles_by_class[target_class]) == 0:
        return path_rows, path_cols
    
    total_shift_r = np.zeros(n_points)
    total_shift_c = np.zeros(n_points)
    
    for obs in obstacles_by_class[target_class]:
        obs_r, obs_c = obs['pos']
        dists = np.sqrt((path_rows - obs_r)**2 + (path_cols - obs_c)**2)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist > avoidance_radius:
            continue
            
        push_vec_c = path_cols[min_dist_idx] - obs_c
        push_vec_r = path_rows[min_dist_idx] - obs_r
        norm = np.sqrt(push_vec_c**2 + push_vec_r**2) + 1e-6
        push_dir_c = push_vec_c / norm
        push_dir_r = push_vec_r / norm
        
        obs_dx, obs_dy = orientation_to_vector(obs['orientation'])
        is_in_front = (push_dir_c * obs_dx + push_dir_r * -obs_dy) > 0.5
        
        if is_in_front:
            cross_prod = obs_dx * push_vec_r - (-obs_dy) * push_vec_c
            if cross_prod > 0:
                push_dir_c, push_dir_r = -obs_dy, -obs_dx
            else:
                push_dir_c, push_dir_r = obs_dy, obs_dx

        indices = np.arange(n_points)
        gaussian_weights = np.exp(-0.5 * ((indices - min_dist_idx) / influence_width)**2)
        proximity_scale = np.clip(1.5 - (min_dist / avoidance_radius), 0.5, 1.5)
        
        total_shift_r += push_dir_r * push_strength * gaussian_weights * proximity_scale
        total_shift_c += push_dir_c * push_strength * gaussian_weights * proximity_scale

    adj_rows = path_rows + total_shift_r
    adj_cols = path_cols + total_shift_c
    
    blend_mask = np.ones(n_points)
    blend_mask[:5] = np.linspace(0, 1, 5)
    blend_mask[-5:] = np.linspace(1, 0, 5)
    adj_rows = path_rows + (adj_rows - path_rows) * blend_mask
    adj_cols = path_cols + (adj_cols - path_cols) * blend_mask

    return adj_rows, adj_cols


def generate_random_scene(H, W, seed=None):
    """Generate a random scene with obstacles and start/goal."""
    if seed is not None:
        np.random.seed(seed)
    
    start = (np.random.randint(H // 2, H - 5), np.random.randint(5, W // 2))
    goal = (np.random.randint(5, H // 2), np.random.randint(W // 2, W - 5))
    
    obstacles_by_class = {}
    for class_id in range(NUM_CLASSES):
        n_obs = np.random.randint(1, 3)
        obstacles = []
        for _ in range(n_obs):
            r = np.random.randint(10, H - 10)
            c = np.random.randint(10, W - 10)
            orientation = np.random.randint(0, NUM_ORIENTATIONS)
            obstacles.append({'pos': (r, c), 'orientation': orientation})
        obstacles_by_class[class_id] = obstacles
    
    return {
        'start': start,
        'goal': goal,
        'obstacles_by_class': obstacles_by_class,
        'H': H,
        'W': W
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(model, dataset, args, device="cuda"):
    model.eval()
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    # Initialize the class-isolated learner
    learner = OnlinePreferenceLearner(
        num_classes=NUM_CLASSES,
        device=device,
        lr=0.005,
        buffer_size=10
    )
    
    print("=" * 60)
    print("TRAINING PHASE (Class-Isolated Neural Learning)")
    print("=" * 60)
    print(f"Channel stack: 3 global + 3×{NUM_CLASSES} class-specific = {3 + 3*NUM_CLASSES} channels")
    print(f"  Global: Base Costmap, Start Map, Goal Map")
    print(f"  Per-class: EDF, Frontalness, Orientation")
    print("=" * 60)
    
    for batch_idx, (cond, x0, obstacles_by_class, goal) in enumerate(dataset):
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        
        H, W = args.img_size, args.img_size
        start_pos = (H - 5, 5)
        goal_pos = (int(goal[0]), int(goal[1]))
        
        print(f"\n--- Scene {batch_idx} ---")
        print(f"    Start: {start_pos}, Goal: {goal_pos}")
        
        # Generate base costmap from diffusion model
        with torch.no_grad():
            generated = sample_ddpm_with_cond(model, cond, diff_betas, alphas, alpha_bar, device=device)
        
        map_base = generated[0, 0]
        map_base_np = map_base.cpu().numpy()
        map_base_np = (map_base_np - map_base_np.min()) / (map_base_np.max() - map_base_np.min() + 1e-8)
        
        # Build feature stack
        features_np = build_feature_stack(
            H, W, obstacles_by_class, start_pos, goal_pos, map_base_np, NUM_CLASSES
        )
        features = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
        
        # Get modulated costmap
        gamma, beta = learner.predict(features)
        map_base_tensor = torch.from_numpy(map_base_np).float().to(device).unsqueeze(0).unsqueeze(0)
        map_modulated = learner.apply_modulation(map_base_tensor, gamma, beta)
        map_modulated_np = map_modulated[0, 0].cpu().numpy()
        
        # Plan path
        path_indices, _ = route_through_array(
            map_modulated_np, start_pos, goal_pos,
            fully_connected=True, geometric=True
        )
        rows, cols = zip(*path_indices)
        
        # Simulate user adjustment (targeting class 0 = chairs)
        adj_rows, adj_cols = add_user_adjustments_robust(
            rows, cols, obstacles_by_class,
            target_class=0, avoidance_radius=30.0, push_strength=7
        )
        
        # Find edit sequences
        edit_sequences = find_edit_sequences(rows, cols, adj_rows, adj_cols, threshold=1.0)
        
        if len(edit_sequences) > 0:
            print(f"  User Edit Detected! Found {len(edit_sequences)} distinct edit sequences.")
            
            # Optional: Visualize feature stack for first scene
            if batch_idx == 0:
                visualize_feature_stack(features_np, obstacles_by_class, 
                                        title=f"Scene {batch_idx}: Feature Stack")
            
            # Visualize user adjustment
            visualize_user_adjustment_overlay(
                costmap=map_modulated_np,
                orig_rows=rows, orig_cols=cols,
                adj_rows=adj_rows, adj_cols=adj_cols,
                obstacles_by_class=obstacles_by_class,
                edit_sequences=edit_sequences,
                start_pos=start_pos, goal_pos=goal_pos,
                title=f"Scene {batch_idx}: User Correction ({len(edit_sequences)} sequences)"
            )
            
            # Train on each edit sequence
            for i, seq in enumerate(edit_sequences):
                centroid = seq['centroid']
                print(f"    -> Processing Sequence {i+1}: Centroid={centroid}, Pts={len(seq['indices'])}")
                
                sub_orig = seq['orig_sub_path']
                sub_adj = seq['adj_sub_path']
                
                # Compute geometric delta
                delta_seq = compute_geometric_delta(sub_orig, sub_adj, H, W)
                target_map_np = np.clip(map_modulated_np + delta_seq, 0, 1)
                
                # Visualize IRL target
                visualize_irl_target(
                    current_costmap=map_modulated_np,
                    delta_map=delta_seq,
                    target_costmap=target_map_np,
                    centroid=centroid,
                    sub_orig=sub_orig,
                    sub_adj=sub_adj,
                    seq_idx=i+1,
                    title=f"Scene {batch_idx} | Sequence {i+1} Target Generation"
                )
                
                # Prepare target tensor
                target_tensor = torch.from_numpy(target_map_np).float().to(device).unsqueeze(0).unsqueeze(0)
                
                # Train with class-isolated learning
                learner.train_on_edit(
                    features=features,
                    base_costmap_tensor=map_base_tensor,
                    target_costmap_tensor=target_tensor,
                    iterations=25
                )
                
                # Show per-class weight norms
                class_norms = learner.get_class_weights_norm()
                print(f"       Class weight norms: {[f'{n:.2f}' for n in class_norms]}")
        else:
            print("  No significant edit needed.")
    
    # =========================================================================
    # DEPLOYMENT PHASE
    # =========================================================================
    print("\n" + "=" * 60)
    print("DEPLOYMENT PHASE")
    print("=" * 60)
    #print(f"\n{learner.get_params_summary()}\n")
    
    # Show final class weight norms
    class_norms = learner.get_class_weights_norm()
    for c in range(NUM_CLASSES):
        class_name = OBSTACLE_CLASSES[c]['name']
        print(f"  {class_name} block weight norm: {class_norms[c]:.4f}")
    print()
    
    n_deploy_scenes = 5
    for deploy_idx in range(n_deploy_scenes):
        print(f"\n--- Deployment Scene {deploy_idx + 1}/{n_deploy_scenes} ---")
        scene = generate_random_scene(args.img_size, args.img_size, seed=1000 + deploy_idx)
        
        # Build conditioning for diffusion model
        class_activations = make_class_occupancy_maps(args.img_size, args.img_size, scene['obstacles_by_class'])
        orientation_maps = make_orientation_maps(args.img_size, args.img_size, scene['obstacles_by_class'])
        goal_map = make_goal_map(args.img_size, args.img_size, scene['goal'])
        
        cond_np = np.concatenate([class_activations, orientation_maps, goal_map[None, :, :]], axis=0)
        cond = torch.from_numpy(cond_np).float().unsqueeze(0).to(device)
        
        # Generate base costmap
        with torch.no_grad():
            generated = sample_ddpm_with_cond(model, cond, diff_betas, alphas, alpha_bar, device=device)
        map_base = generated[0, 0]
        map_base_np = map_base.cpu().numpy()
        map_base_np = (map_base_np - map_base_np.min()) / (map_base_np.max() - map_base_np.min() + 1e-8)
        
        # Build feature stack for preference adapter
        features_np = build_feature_stack(
            args.img_size, args.img_size,
            scene['obstacles_by_class'],
            scene['start'], scene['goal'],
            map_base_np, NUM_CLASSES
        )
        features = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
        
        # Get modulated costmap
        gamma, beta = learner.predict(features)
        map_base_tensor = torch.from_numpy(map_base_np).float().to(device).unsqueeze(0).unsqueeze(0)
        map_modulated = learner.apply_modulation(map_base_tensor, gamma, beta)
        map_modulated_np = map_modulated[0, 0].cpu().numpy()
        
        # Plan paths
        path_before, _ = route_through_array(
            map_base_np, scene['start'], scene['goal'],
            fully_connected=True, geometric=True
        )
        path_after, _ = route_through_array(
            map_modulated_np, scene['start'], scene['goal'],
            fully_connected=True, geometric=True
        )
        
        visualize_deployment_scene(
            base_costmap=map_base_np,
            modulated_costmap=map_modulated_np,
            path_before=path_before,
            path_after=path_after,
            scene=scene,
            deploy_idx=deploy_idx,
            features_np=features_np
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
    seed_env(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    eval_ds = MultiClassCostmapDataset(
        n_samples=args.n_samples, H=args.img_size, W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    
    class MetadataDataLoader:
        def __init__(self, dataset, n_samples):
            self.dataset = dataset
            self.n_samples = n_samples
        
        def __iter__(self):
            for i in range(self.n_samples):
                yield self.dataset.get_sample_with_metadata(i)
        
        def __len__(self):
            return self.n_samples
    
    eval_dl = MetadataDataLoader(eval_ds, args.n_samples)
    cond_channels = get_cond_channels()
    model = UNet(in_channels=cond_channels + 1, lora_rank=args.lora_rank, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint not found, using random weights")
    
    run_full_pipeline(model, eval_dl, args, device=device) 
