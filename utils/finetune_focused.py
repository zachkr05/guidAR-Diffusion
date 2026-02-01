"""
IRL finetuning for MoE diffusion costmap generation.

Simpler approach: Direct planning loss without complex masking.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np



def compute_path_difference_mask(orig_path, user_path, H, W, device, threshold=5.0, sigma=5.0):
    """
    Returns a mask that's 1.0 where the paths differ significantly,
    0.0 where they're the same.
    """
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else orig_path.cpu().numpy()
    user_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()
    
    # For each point on user_path, find distance to nearest point on orig_path
    from scipy.spatial.distance import cdist
    
    # Resample paths to same length for comparison
    n_points = max(len(orig_np), len(user_np))
    
    # Distance from each user point to closest original point
    dists = cdist(user_np, orig_np).min(axis=1)  # (len(user_path),)
    
    # Points where user deviated significantly
    changed_mask = dists > threshold
    changed_points = user_np[changed_mask]
    
    if len(changed_points) == 0:
        # No significant changes, return empty mask
        return torch.zeros((1, 1, H, W), device=device)
    
    # Create spatial mask around changed points
    mask = torch.zeros((1, 1, H, W), device=device)
    xs = changed_points[:, 0].astype(int).clip(0, W-1)
    ys = changed_points[:, 1].astype(int).clip(0, H-1)
    mask[0, 0, ys, xs] = 1.0
    
    # Dilate with Gaussian blur
    mask = gaussian_blur(mask, kernel_size=int(6*sigma)|1, sigma=sigma)
    mask = mask / (mask.max() + 1e-8)  # Normalize to [0, 1]
    
    return mask


def make_path_target(path, H, W, device, sigma=3.0):
    """
    Converts a path (N, 2) [x, y] into a soft probability map (1, 1, H, W).
    Uses Gaussian blur for smoother gradients.
    """
    target = torch.zeros((1, 1, H, W), device=device)
    
    if isinstance(path, np.ndarray):
        path = torch.from_numpy(path).float().to(device)
    
    xs = path[:, 0].long().clamp(0, W - 1)
    ys = path[:, 1].long().clamp(0, H - 1)
    
    target[0, 0, ys, xs] = 1.0
    
    # Gaussian blur for smooth gradients
    kernel_size = int(6 * sigma) | 1  # Ensure odd
    target = gaussian_blur(target, kernel_size, sigma)
    
    target = target / (target.sum() + 1e-8)
    return target


def gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur to tensor."""
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
    kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    # Apply
    padding = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=padding)


def finetune_models_focused(
    model,
    batch,
    orig_path,
    user_path,
    device,
    lr,
    target_class,
    epochs,
    ddpm,
    planner,
    # Loss weights - simplified
    w_diffusion=1.0,
    w_plan=1.0,
):
    """
    Simplified IRL finetuning.
    
    Key insight: Let the planning loss do the work. The diffusion loss
    provides stability, the planning loss provides the learning signal.
    """
    
    features, targets, positions, radii, goal = batch
    H, W = 128, 128
    # =========================================================================
    # Step 2: Setup
    # =========================================================================
    expert_model = model.experts[target_class]
    expert_model.set_finetune(active=True)
    
    optimizer = AdamW(
        [p for p in expert_model.parameters() if p.requires_grad],
        lr=lr
    )
    
    # Create path targets with good blur for gradient flow
    user_path_target = make_path_target(user_path, H, W, device, sigma=5.0)
    orig_path_target = make_path_target(orig_path, H, W, device, sigma=5.0)
    
    # Ground truth and conditioning
    x_0_gt = targets[target_class].to(device)
    conditioning = features[target_class].to(device)
    
    # Other class maps (frozen during training)
    other_class_maps = {}
    for cls in model.obstacle_classes:
        if cls != target_classes:
            other_class_maps[cls] = targets[cls].to(device).detach()
    
    loss_history = []
    
    # =========================================================================
    # Step 3: Training loop
    # ========================================================================
    for epoch in tqdm(range(epochs), desc="IRL Finetuning"):
        model.train()
        optimizer.zero_grad()
        
        B = x_0_gt.shape[0]
        
        t = torch.randint(0, ddpm.timesteps // 4, (B,), device=device).long()
        
        x_t, noise = ddpm.q_sample(x_0_gt, t)
        noise_pred = expert_model(x_t, t, conditioning)
        x_0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)
        
        # =================================================================
        # Loss 1: Diffusion loss (keeps model stable)
        # =================================================================
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        # =================================================================
        # Loss 2: Planning loss (THE MAIN LEARNING SIGNAL)
        # =================================================================
        maps_to_fuse = [x_0_pred]
        for cls in model.obstacle_classes:
            if cls != target_classes:
                maps_to_fuse.append(other_class_maps[cls])
        
        stacked = torch.stack(maps_to_fuse, dim=1).squeeze(2)
        fused_map = torch.logsumexp(stacked, dim=1, keepdim=True)
        
        # Convert to cost
        cost_map = F.softplus(fused_map) + 0.1
        
        # Run planner to get visitation probabilities
        pred_visitation = planner(cost_map, goal.to(device))
        
        # Cross-entropy: push visitation toward user's path
        # This is the key: we want pred_visitation to be HIGH along user_path
        loss_plan = -(user_path_target * torch.log(pred_visitation + 1e-8)).sum()
        
        # Also: we DON'T want to make the original path worse
        # (unless the user explicitly moved away from it)
        loss_preserve = -(orig_path_target * torch.log(pred_visitation + 1e-8)).sum()
       



        edit_mask = compute_path_difference_mask(orig_path, user_path, H, W, device, threshold=5.0)

        # Only apply planning loss in the edited region
        user_target_in_edit = user_path_target * edit_mask
        user_target_in_edit = user_target_in_edit / (user_target_in_edit.sum() + 1e-8)

        loss_plan = -(user_target_in_edit * torch.log(pred_visitation + 1e-8)).sum()


        # ALSO: Preserve the original path where it WASN'T edited
        unchanged_mask = 1.0 - edit_mask
        orig_target_unchanged = orig_path_target * unchanged_mask
        orig_target_unchanged = orig_target_unchanged / (orig_target_unchanged.sum() + 1e-8)

        loss_preserve = -(orig_target_unchanged * torch.log(pred_visitation + 1e-8)).sum()

        # Combined
        total_loss = w_diffusion * loss_diffusion + w_plan * loss_plan + 0.5 * loss_preserve 
        # =================================================================
        # Total loss
        # =================================================================
        # Start with equal weight, adjust based on what you see
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(expert_model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        loss_history.append({
            'total': total_loss.item(),
            'diffusion': loss_diffusion.item(),
            'plan': loss_plan.item(),
        })
        
        # Diagnostics
        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                # Check if cost is changing
                delta = x_0_pred - x_0_gt
                
                # Where is the user's path? Check cost there
                user_path_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()
                xs = user_path_np[:, 0].astype(int).clip(0, W-1)
                ys = user_path_np[:, 1].astype(int).clip(0, H-1)
                
                cost_on_user_path = x_0_pred[0, 0, ys, xs].mean()
                cost_on_user_path_gt = x_0_gt[0, 0, ys, xs].mean()
                
                # Visitation on user path
                vis_on_user_path = pred_visitation[0, 0, ys, xs].sum()
                
                print(f"\n  [Epoch {epoch}] "
                      f"plan_loss={loss_plan.item():.4f}, "
                      f"cost_user_path={cost_on_user_path.item():.3f} (gt={cost_on_user_path_gt.item():.3f}), "
                      f"vis_user_path={vis_on_user_path.item():.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n[IRL] Training complete.")
    print(f"  Final diffusion loss: {loss_history[-1]['diffusion']:.6f}")
    print(f"  Final planning loss: {loss_history[-1]['plan']:.4f}")
    
    expert_model.set_finetune(active=False)
    return loss_history 
