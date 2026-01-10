import argparse
import os
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.graph import route_through_array

from utils import simulate_user_correction 
from plotter_test import visualize_learning_step, visualize_comparison, add_obstacle_markers
from diffusion_utils import schedule_betas
from DataGenerator.dataset_costmap import (
    MultiClassCostmapDataset, 
    get_cond_channels
)
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES
from UNet.UNet import UNet
from train import sample_ddpm_with_cond
from dataLoader import MetadataDataLoader

from irl_agent import compute_geometric_delta
from finetune_utils import finetune_multiple_classes


def find_affected_objects(delta_map, obstacles_by_class, threshold=0.005):
    """Find objects where the user's correction indicates increased cost."""
    affected = []
    for class_id, obstacles in obstacles_by_class.items():
        class_name = OBSTACLE_CLASSES[class_id]['name']
        for obs in obstacles:
            r, c = obs['pos']
            r_idx = int(np.clip(r, 0, delta_map.shape[0] - 1))
            c_idx = int(np.clip(c, 0, delta_map.shape[1] - 1))
            
            delta_val = delta_map[r_idx, c_idx]
            
            if abs(delta_val) > threshold:
                affected.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'pos': (r, c),
                    'delta': delta_val
                })
    return affected


def visualize_generated_vs_target(generated, target, path, obstacles_by_class, start_pos, goal_pos):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Generated", "Ground Truth (x0)"))
    
    fig.add_trace(go.Heatmap(z=generated, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=target, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    
    path_arr = np.array(path)
    for col in [1, 2]:
        fig.add_trace(go.Scatter(x=path_arr[:, 1], y=path_arr[:, 0], mode='lines',
                                  line=dict(color='cyan', width=3), showlegend=False), row=1, col=col)
    
    fig.update_layout(height=400, width=900)
    fig.update_yaxes(autorange='reversed')
    fig.show()


def run_online_learning_loop(model, dataset, args, device="cuda"):
    """
    Runs the 'Generate -> Correct -> Finetune' loop.
    Changes to the model PERSIST to the next scene.
    """
    diff_betas, alphas, alpha_bar = schedule_betas(args.timesteps, args.beta_start, args.beta_end, device=device)
    
    print(f"Starting Online Learning on {len(dataset)} scenes...")

    for idx, (cond, x0, obstacles_by_class, goal) in enumerate(dataset):
        print(f"\n{'='*60}")
        print(f"Scene {idx}")
        print(f"{'='*60}")
        
        # Print obstacle info
        for class_id, obs_list in obstacles_by_class.items():
            class_name = OBSTACLE_CLASSES[class_id]['name']
            if obs_list:
                positions = [obs['pos'] for obs in obs_list]
                print(f"  {class_name} (class {class_id}): {len(obs_list)} obstacles at {positions}")
        
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        
        H, W = args.img_size, args.img_size
        start_pos = (H - 5, 5)
        goal_pos = (int(goal[0]), int(goal[1]))
        
        # ==========================================
        # PHASE 1: GENERATE (Inference with current model)
        # ==========================================
        model.eval()
        with torch.no_grad():
            generated = sample_ddpm_with_cond(model, cond, diff_betas, alphas, alpha_bar, device=device)
        
        costmap = generated[0, 0].cpu().numpy()
        costmap_norm = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
        
        print(f"Generated costmap - min: {costmap.min():.4f}, max: {costmap.max():.4f}, std: {costmap.std():.4f}")
        
        # =========================================
        # PHASE 2: INTERACT (Simulate User)
        # ==========================================
        try:
            path, _ = route_through_array(costmap_norm, start_pos, goal_pos, fully_connected=True, geometric=True)
        except:
            print("No path found. Skipping.")
            continue

        orig_rows, orig_cols = zip(*path)
        orig_path = list(zip(orig_rows, orig_cols))

        # Simulate User Correction (avoid Class 0 - Chairs)
        adj_rows, adj_cols = simulate_user_correction(
            orig_rows, orig_cols, obstacles_by_class,
            target_class=0,
            avoidance_radius=15.0, 
            push_strength=2.0 
        )
        adj_path = list(zip(adj_rows, adj_cols))

        # Calculate geometric delta
        delta_map = compute_geometric_delta(orig_path, adj_path, H, W, cost_increase=1.0)
        
        # Identify affected objects
        affected_objs = find_affected_objects(delta_map, obstacles_by_class)
        affected_class_ids = list(set([obj['class_id'] for obj in affected_objs]))
        
        if not affected_class_ids:
            print("User accepted the path (No correction needed).")
            continue
        
        # Print affected objects
        print(f"\nAffected objects:")
        for obj in affected_objs:
            print(f"  {obj['class_name']} at {obj['pos']}, delta={obj['delta']:.4f}")
        print(f"Affected class IDs: {affected_class_ids}")

        # ==========================================
        # PHASE 3: LEARN (Finetune LoRA)
        # ==========================================
        print("\nFinetuning model...")
        
        loss = finetune_multiple_classes(
            model=model,
            cond_input=cond,
            base_costmap=generated,
            geometric_delta=delta_map,
            affected_class_ids=affected_class_ids,
            learning_rate=1e-3,
            num_steps=10
        )
        print(f"Finetune complete. Final loss: {loss:.5f}")

        # ==========================================
        # PHASE 4: VERIFY (Did it learn?)
        # ==========================================
        model.eval()
        with torch.no_grad():
            generated_post = sample_ddpm_with_cond(model, cond, diff_betas, alphas, alpha_bar, device=device)
            
        costmap_post = generated_post[0, 0].cpu().numpy()
        costmap_post_norm = (costmap_post - costmap_post.min()) / (costmap_post.max() - costmap_post.min() + 1e-8)
        
        print(f"Post-training costmap - min: {costmap_post.min():.4f}, max: {costmap_post.max():.4f}, std: {costmap_post.std():.4f}")
        
        # Calculate improvement
        diff = costmap_post_norm - costmap_norm
        print(f"Change in costmap - min: {diff.min():.4f}, max: {diff.max():.4f}, std: {diff.std():.4f}")
        
        # Visualize: Pre-Correction vs Post-Correction
        visualize_learning_step(
            costmap_norm, 
            costmap_post_norm, 
            orig_path, 
            adj_path, 
            delta_map, 
            obstacles_by_class, 
            idx,
            affected_classes=affected_class_ids
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
    parser.add_argument("--n-obs-per-class", type=int, default=3)
    parser.add_argument("--min-obs-per-class", type=int, default=1)
    parser.add_argument("--min-total-obs", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    
    args = parser.parse_args()
    seed_env(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Dataset
    eval_ds = MultiClassCostmapDataset(
        n_samples=args.n_samples, 
        H=args.img_size, 
        W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    eval_dl = MetadataDataLoader(eval_ds, args.n_samples)
    
    # Initialize Model
    cond_channels = get_cond_channels()
    print(f"Conditioning channels: {cond_channels}")
    print(f"Model input channels: {cond_channels + 1}")
    
    model = UNet(in_channels=cond_channels + 1, lora_rank=args.lora_rank, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}")
    
    # Run the Learning Loop
    run_online_learning_loop(model, eval_dl, args, device=device)
