"""
IRL finetuning for MoE diffusion costmap generation.

Simpler approach: Direct planning loss without complex masking.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist


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


def compute_path_difference_mask(orig_path, user_path, H, W, device, threshold=5.0, sigma=5.0):
    """
    
    Anywhere where User's path deviates make a mask and dilate it with guassian blur

    """
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else orig_path.cpu().numpy()
    user_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()
    
    # Resample paths to same length for comparison
    n_points = max(len(orig_np), len(user_np))
    dists = cdist(user_np, orig_np).min(axis=1)
    
    # Points where user deviated significantly
    changed_mask = dists > threshold
    changed_points = user_np[changed_mask]
    
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

def compute_orientation_aware_mask(orig_path, user_path, positions, orientations,
                                   obstacle_classes, H, W, device,
                                   threshold=5.0, sigma=5.0, angular_sigma=0.8):
    """
    Orientation-aware edit mask.

    For each pixel in the edit region, weight it by how aligned it is with
    the facing direction of the nearest obstacle. If the user pushed the path
    away from the FRONT of an obstacle, only that angular sector gets high weight.

    Args:
        angular_sigma: Controls how tight the angular weighting is.
                       Lower = tighter cone, higher = broader.
    Returns:
        orientation_mask: (1, 1, H, W) tensor, values in [0, 1]
        unchanged_direction_mask: (1, 1, H, W) tensor — regions around obstacles
                                  where the user did NOT edit (for regularization)
    """
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else orig_path.cpu().numpy()
    user_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()

    # Step 1: Find which points the user actually moved
    dists = cdist(user_np, orig_np).min(axis=1)
    changed_mask = dists > threshold
    changed_points = user_np[changed_mask]

    if len(changed_points) == 0:
        zeros = torch.zeros((1, 1, H, W), device=device)
        ones = torch.ones((1, 1, H, W), device=device)
        return zeros, ones

    # Step 2: Build a pixel grid
    rows, cols = np.mgrid[0:H, 0:W]  # rows = y coords, cols = x coords

    # Step 3: For each obstacle, compute angular weight map
    # angular_weight[y, x] = max over all obstacles of:
    #   exp(-angular_diff^2 / (2 * angular_sigma^2))
    # where angular_diff is the angle between:
    #   (a) vector from obstacle center to pixel (y,x)
    #   (b) obstacle's facing direction

    pos_dict = positions[0] if isinstance(positions, list) else positions

    angular_weight = np.zeros((H, W), dtype=np.float32)
    obstacle_proximity = np.zeros((H, W), dtype=np.float32)

    for cls in obstacle_classes:
        obs_list = pos_dict.get(cls, [])
        for i, pos in enumerate(obs_list):
            obs_r, obs_c = float(pos[0]), float(pos[1])

            # Get obstacle orientation
            if isinstance(orientations, dict):
                angle = orientations[cls][i]
            else:
                angle = orientations[0][cls][i]
            if hasattr(angle, 'item'):
                angle = angle.item()

            # Vector from obstacle to each pixel
            dy = rows - obs_r   # (H, W)
            dx = cols - obs_c   # (H, W)
            pixel_angle = np.arctan2(dy, dx)  # angle from obstacle to pixel

            # Angular difference (wrapped to [-pi, pi])
            ang_diff = pixel_angle - angle
            ang_diff = (ang_diff + np.pi) % (2 * np.pi) - np.pi

            # Directional weight: high in facing direction, low behind
            dir_weight = np.exp(-ang_diff**2 / (6 * angular_sigma**2))

            # Distance falloff from obstacle
            dist = np.sqrt(dy**2 + dx**2)
            proximity = np.exp(-dist**2 / (2 * (sigma * 3)**2))  # broad proximity

            # Take max across all obstacles (any obstacle's front matters)
            angular_weight = np.maximum(angular_weight, dir_weight * proximity)
            obstacle_proximity = np.maximum(obstacle_proximity, proximity)

    # Step 4: Combine with spatial edit mask
    # Start with the basic path-difference mask
    spatial_mask = torch.zeros((1, 1, H, W), device=device)
    xs = changed_points[:, 0].astype(int).clip(0, W-1)
    ys = changed_points[:, 1].astype(int).clip(0, H-1)
    spatial_mask[0, 0, ys, xs] = 1.0
    spatial_mask = gaussian_blur(spatial_mask, kernel_size=int(6*sigma)|1, sigma=sigma)
    spatial_mask = spatial_mask / (spatial_mask.max() + 1e-8)

    # Multiply: only high where BOTH the user edited AND it's in the obstacle's facing direction
    angular_weight_t = torch.from_numpy(angular_weight).float().to(device).unsqueeze(0).unsqueeze(0)
    orientation_mask = spatial_mask * angular_weight_t
    orientation_mask = orientation_mask / (orientation_mask.max() + 1e-8)

    # Step 5: Unchanged direction mask for regularization
    # Regions near obstacles but NOT in the edit direction
    # = obstacle proximity * (1 - orientation_mask)
    obstacle_proximity_t = torch.from_numpy(obstacle_proximity).float().to(device).unsqueeze(0).unsqueeze(0)
    unchanged_direction_mask = obstacle_proximity_t * (1.0 - orientation_mask)
    unchanged_direction_mask = unchanged_direction_mask / (unchanged_direction_mask.max() + 1e-8)

    return orientation_mask, unchanged_direction_mask


def finetune_models(
        model,
        batch,
        orig_path,
        user_path,
        device,
        lr,
        edit_regions,
        epochs,
        ddpm,
        planner,
        w_diffusion=1.0,
        w_plan=1.0,
        w_preserve=0.5,
        w_directional_reg=1.0,
):
    """
    Orientation-aware IRL co-finetuning.

    Key additions over the original:
      1. Orientation-aware edit mask — planning loss is focused on the angular
         sector in front of obstacles, not applied globally.
      2. Directional regularization — costmap values in directions the user
         did NOT edit are penalized if they drift from pre-finetune values.
    """
    features, targets, positions, radii, goal, orientations = batch
    H, W = 128, 128

    loss_history = []

    for region_mask, points, affected_classes in edit_regions:

        expert_models = {}
        x_0_gts = {}
        conditionings = {}

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=True)
            expert_models[cls] = model.experts[cls]
            x_0_gts[cls] = targets[cls].to(device)
            conditionings[cls] = features[cls].to(device)

        frozen_maps = {}
        for cls in model.obstacle_classes:
            if cls not in affected_classes:
                frozen_maps[cls] = targets[cls].to(device).detach()

        optimizer = AdamW(
            [p for expert in expert_models.values()
             for p in expert.parameters() if p.requires_grad],
            lr=lr,
        )

        # --- Pre-compute targets and masks ---
        user_path_target = make_path_target(user_path, H, W, device, sigma=5.0)
        orig_path_target = make_path_target(orig_path, H, W, device, sigma=5.0)

        # Orientation-aware masks
        orientation_mask, unchanged_dir_mask = compute_orientation_aware_mask(
            orig_path, user_path, positions, orientations,
            affected_classes, H, W, device,
            threshold=5.0, sigma=5.0, angular_sigma=1.5,
        )

        # Snapshot pre-finetune costmaps for directional regularization
        with torch.no_grad():
            pre_finetune_maps = {}
            for cls in affected_classes:
                B = x_0_gts[cls].shape[0]
                t_zero = torch.zeros((B,), device=device).long()
                x_t, noise = ddpm.q_sample(x_0_gts[cls], t_zero)
                noise_pred = expert_models[cls](x_t, t_zero, conditionings[cls])
                pre_finetune_maps[cls] = ddpm.predict_start_from_noise(
                    x_t, t_zero, noise_pred).detach()

        for epoch in tqdm(range(epochs), desc="IRL Co-Finetuning"):
            model.train()
            optimizer.zero_grad()

            B = list(x_0_gts.values())[0].shape[0]
            t = torch.randint(0, ddpm.timesteps // 4, (B,), device=device).long()

            x_0_preds = {}
            noises = {}
            noise_preds = {}

            for cls in affected_classes:
                x_t, noise = ddpm.q_sample(x_0_gts[cls], t)
                noise_pred = expert_models[cls](x_t, t, conditionings[cls])
                x_0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)
                x_0_preds[cls] = x_0_pred
                noises[cls] = noise
                noise_preds[cls] = noise_pred

            # --- Contribution-weighted credit assignment ---
            with torch.no_grad():
                contributions = {}
                for cls in affected_classes:
                    contributions[cls] = (x_0_preds[cls].detach() * orientation_mask).sum()
                total_contrib = sum(contributions.values()) + 1e-8
                expert_weights = {
                    cls: (contributions[cls] / total_contrib).item()
                    for cls in affected_classes
                }

            # Loss 1: Weighted diffusion loss
            loss_diffusion = sum(
                expert_weights[cls] * F.mse_loss(noise_preds[cls], noises[cls])
                for cls in affected_classes
            )

            # Fuse all maps
            maps_to_fuse = []
            for cls in model.obstacle_classes:
                if cls in x_0_preds:
                    maps_to_fuse.append(x_0_preds[cls])
                else:
                    maps_to_fuse.append(frozen_maps[cls])

            stacked = torch.stack(maps_to_fuse, dim=1).squeeze(2)
            fused_map = torch.logsumexp(stacked, dim=1, keepdim=True)
            cost_map = F.softplus(fused_map) + 0.1

            pred_visitation = planner(cost_map, goal.to(device))

            # Loss 2: Planning loss — masked by orientation-aware edit region
            user_target_in_edit = user_path_target * orientation_mask
            user_target_in_edit = user_target_in_edit / (user_target_in_edit.sum() + 1e-8)
            loss_plan = -(user_target_in_edit * torch.log(pred_visitation + 1e-8)).sum()

            # Loss 3: Preserve original path where unchanged
            unchanged_mask = 1.0 - orientation_mask
            orig_target_unchanged = orig_path_target * unchanged_mask
            orig_target_unchanged = orig_target_unchanged / (orig_target_unchanged.sum() + 1e-8)
            loss_preserve = -(orig_target_unchanged * torch.log(pred_visitation + 1e-8)).sum()

            # Loss 4: Directional regularization
            # Penalize costmap changes in angular regions the user did NOT edit
            # This prevents the model from reducing cost globally instead of directionally
            loss_dir_reg = torch.tensor(0.0, device=device)
            for cls in affected_classes:
                delta = x_0_preds[cls] - pre_finetune_maps[cls]
                # Only penalize changes in the unchanged-direction region
                loss_dir_reg = loss_dir_reg + (delta**2 * unchanged_dir_mask).mean()

            total_loss = (w_diffusion * loss_diffusion
                          + w_plan * loss_plan
                          + w_preserve * loss_preserve
                          + w_directional_reg * loss_dir_reg)

            total_loss.backward()
            for cls in affected_classes:
                torch.nn.utils.clip_grad_norm_(expert_models[cls].parameters(), 1.0)
            optimizer.step()

            loss_history.append({
                'total': total_loss.item(),
                'diffusion': loss_diffusion.item(),
                'plan': loss_plan.item(),
                'dir_reg': loss_dir_reg.item(),
                'expert_weights': expert_weights.copy(),
            })

            if epoch % 100 == 0 or epoch == epochs - 1:
                weight_str = ", ".join(f"{cls}={w:.2f}" for cls, w in expert_weights.items())
                print(f"\n  [Epoch {epoch}] total={total_loss.item():.4f}, "
                      f"diffusion={loss_diffusion.item():.4f}, plan={loss_plan.item():.4f}, "
                      f"dir_reg={loss_dir_reg.item():.4f}, "
                      f"weights=[{weight_str}]")

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=False)

    return loss_history


