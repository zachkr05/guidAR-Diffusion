"""
IRL finetuning for MoE diffusion costmap generation.

Orientation-aware approach with data augmentation:
  - Each epoch, obstacle orientations are randomly rotated
  - The sin/cos conditioning channels are rewritten to match
  - The edit mask rotates correspondingly (edit is always "in front")
  - This forces the LoRA layers to USE sin/cos features to determine
    which direction to reduce cost, rather than memorizing spatial locations.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist


def gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur to tensor."""
    coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
    kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=padding)


def compute_path_difference_mask(orig_path, user_path, H, W, device, threshold=5.0, sigma=5.0):
    """Basic spatial edit mask from path deviation."""
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else orig_path.cpu().numpy()
    user_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()

    dists = cdist(user_np, orig_np).min(axis=1)
    changed_mask = dists > threshold
    changed_points = user_np[changed_mask]

    mask = torch.zeros((1, 1, H, W), device=device)
    if len(changed_points) == 0:
        return mask
    xs = changed_points[:, 0].astype(int).clip(0, W - 1)
    ys = changed_points[:, 1].astype(int).clip(0, H - 1)
    mask[0, 0, ys, xs] = 1.0

    mask = gaussian_blur(mask, kernel_size=int(6 * sigma) | 1, sigma=sigma)
    mask = mask / (mask.max() + 1e-8)
    return mask


def make_path_target(path, H, W, device, sigma=3.0):
    """Convert path (N, 2) [x, y] into soft probability map (1, 1, H, W)."""
    target = torch.zeros((1, 1, H, W), device=device)
    if isinstance(path, np.ndarray):
        path = torch.from_numpy(path).float().to(device)
    xs = path[:, 0].long().clamp(0, W - 1)
    ys = path[:, 1].long().clamp(0, H - 1)
    target[0, 0, ys, xs] = 1.0
    kernel_size = int(6 * sigma) | 1
    target = gaussian_blur(target, kernel_size, sigma)
    target = target / (target.sum() + 1e-8)
    return target


# =========================================================================
# Orientation augmentation helpers
# =========================================================================

def analyze_edit_direction(orig_path, user_path, positions, orientations,
                          obstacle_classes, threshold=5.0):
    """
    For each obstacle, compute the RELATIVE angle between the obstacle's
    facing direction and the direction the user pushed the path toward.

    Returns:
        relative_edit_angles: dict {cls: [rel_angle_obs0, rel_angle_obs1, ...]}
            The angle offset FROM the obstacle's facing direction TO the
            user's edit direction. This is ROTATION-INVARIANT — it stays
            the same regardless of augmentation.
            None if the obstacle wasn't near any edit.
    """
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else orig_path.cpu().numpy()
    user_np = user_path if isinstance(user_path, np.ndarray) else user_path.cpu().numpy()

    dists = cdist(user_np, orig_np).min(axis=1)
    changed_mask = dists > threshold
    changed_points = user_np[changed_mask]

    pos_dict = positions[0] if isinstance(positions, list) else positions
    relative_edit_angles = {}

    for cls in obstacle_classes:
        relative_edit_angles[cls] = []
        obs_list = pos_dict.get(cls, [])
        for i, pos in enumerate(obs_list):
            obs_r, obs_c = float(pos[0]), float(pos[1])

            if len(changed_points) == 0:
                relative_edit_angles[cls].append(None)
                continue

            # Distance from obstacle t each changed point
            dx = changed_points[:, 0] - obs_c
            dy = changed_points[:, 1] - obs_r
            dists_to_obs = np.sqrt(dx**2 + dy**2)

            nearby_mask = dists_to_obs < 40.0
            if not nearby_mask.any():
                relative_edit_angles[cls].append(None)
                continue

            # Absolute angle from obstacle toward user's edit
            mean_dx = dx[nearby_mask].mean()
            mean_dy = dy[nearby_mask].mean()
            edit_angle_abs = np.arctan2(mean_dy, mean_dx)

            # Get obstacle's original facing direction
            if isinstance(orientations, dict):
                obs_angle = orientations[cls][i]
            else:
                obs_angle = orientations[0][cls][i]
            if hasattr(obs_angle, 'item'):
                obs_angle = obs_angle.item()

            # Relative angle: how far from "front" is the edit?
            rel = edit_angle_abs - obs_angle
            rel = (rel + np.pi) % (2 * np.pi) - np.pi
            relative_edit_angles[cls].append(rel)

    return relative_edit_angles


def compute_directional_edit_mask(positions, aug_orientations, obstacle_classes,
                                  relative_edit_angles, H, W, device,
                                  angular_sigma=1.2, spatial_sigma=15.0):
    """
    Build edit mask where weight is high in the direction the user edited,
    relative to the (augmented) obstacle orientation.

    Because we store RELATIVE angles, rotating the obstacle orientation
    automatically rotates where the mask is strong.
    """
    pos_dict = positions[0] if isinstance(positions, list) else positions
    rows, cols = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), dtype=np.float32)

    for cls in obstacle_classes:
        obs_list = pos_dict.get(cls, [])
        for i, pos in enumerate(obs_list):
            rel_angle = relative_edit_angles[cls][i]
            if rel_angle is None:
                continue

            obs_r, obs_c = float(pos[0]), float(pos[1])

            # Get augmented orientation
            aug_angle = aug_orientations[cls][i]
            if hasattr(aug_angle, 'item'):
                aug_angle = aug_angle.item()

            # The absolute edit direction in augmented space
            target_angle = aug_angle + rel_angle

            # Angle from obstacle to each pixel
            dy = rows - obs_r
            dx = cols - obs_c
            pixel_angle = np.arctan2(dy, dx)

            # Weight by angular proximity to edit direction
            ang_diff = pixel_angle - target_angle
            ang_diff = (ang_diff + np.pi) % (2 * np.pi) - np.pi
            dir_weight = np.exp(-ang_diff**2 / (2 * angular_sigma**2))

            # Spatial falloff
            dist = np.sqrt(dy**2 + dx**2)
            proximity = np.exp(-dist**2 / (2 * spatial_sigma**2))

            mask = np.maximum(mask, dir_weight * proximity)

    mask_t = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
    mask_t = mask_t / (mask_t.max() + 1e-8)
    return mask_t


def compute_proximity_mask(positions, obstacle_classes, relative_edit_angles,
                           H, W, device, spatial_sigma=15.0):
    """Isotropic proximity to obstacles that are near edits (for regularization)."""
    pos_dict = positions[0] if isinstance(positions, list) else positions
    rows, cols = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), dtype=np.float32)

    for cls in obstacle_classes:
        obs_list = pos_dict.get(cls, [])
        for i, pos in enumerate(obs_list):
            if relative_edit_angles[cls][i] is None:
                continue
            obs_r, obs_c = float(pos[0]), float(pos[1])
            dist = np.sqrt((rows - obs_r)**2 + (cols - obs_c)**2)
            proximity = np.exp(-dist**2 / (2 * spatial_sigma**2))
            mask = np.maximum(mask, proximity)

    mask_t = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
    mask_t = mask_t / (mask_t.max() + 1e-8)
    return mask_t


def augment_orientations(orientations, obstacle_classes):
    """
    Randomly rotate all obstacle orientations by the same delta.

    Returns:
        new_orientations: dict {cls: list of new angle values}
        delta_angle: the rotation applied (radians)
    """
    delta_angle = np.random.uniform(0, 2 * np.pi)

    new_orientations = {}
    for cls in obstacle_classes:
        if isinstance(orientations, dict):
            orig = orientations[cls]
        else:
            orig = orientations[0][cls]

        new_angles = []
        for a in orig:
            val = a.item() if hasattr(a, 'item') else float(a)
            new_angles.append(val + delta_angle)
        new_orientations[cls] = new_angles

    return new_orientations, delta_angle


def rewrite_sin_cos_channels(conditioning, new_orientations, cls, obstacle_classes,
                             positions, device):
    """
    Rewrite sin/cos channels in the conditioning tensor for augmented orientations.

    Channel layout per expert (for n_classes total):
        0: curr_bin, 1: curr_rad, 2: curr_sin, 3: curr_cos,
        4 to 4+(n-2): other_bin,
        4+(n-1) to 4+2(n-2): other_rad,
        4+2(n-1) to 4+3(n-2): other_sin,
        4+3(n-1) to 4+4(n-2): other_cos,
        last: goal
    """
    cond = conditioning.clone()
    pos_dict = positions[0] if isinstance(positions, list) else positions
    n_cls = len(obstacle_classes)

    # Current class: channels 2 (sin), 3 (cos)
    cond[:, 2, :, :] = 0.0
    cond[:, 3, :, :] = 0.0
    for i, pos in enumerate(pos_dict.get(cls, [])):
        r, c = int(pos[0]), int(pos[1])
        angle = new_orientations[cls][i]
        cond[:, 2, r, c] = np.sin(angle)
        cond[:, 3, r, c] = np.cos(angle)

    # Other classes
    other_classes = [c for c in obstacle_classes if c != cls]
    n_other = n_cls - 1
    other_sin_start = 4 + 2 * n_other  # after other_bin and other_rad
    other_cos_start = other_sin_start + n_other

    for j, other_cls in enumerate(other_classes):
        sin_ch = other_sin_start + j
        cos_ch = other_cos_start + j
        cond[:, sin_ch, :, :] = 0.0
        cond[:, cos_ch, :, :] = 0.0

        for i, pos in enumerate(pos_dict.get(other_cls, [])):
            r, c = int(pos[0]), int(pos[1])
            angle = new_orientations[other_cls][i]
            cond[:, sin_ch, r, c] = np.sin(angle)
            cond[:, cos_ch, r, c] = np.cos(angle)

    return cond


# =========================================================================
# Main finetuning loop
# =========================================================================

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
        w_directional_reg=0.3,
        angular_sigma=1.2,
):
    """
    Orientation-aware IRL co-finetuning with rotation augmentation.

    Each epoch:
      1. Sample a random rotation delta
      2. Rewrite sin/cos conditioning channels with rotated angles
      3. Recompute direction-aware edit mask (rotated accordingly)
      4. Forward pass + loss with augmented data

    Because every epoch sees a different rotation but the edit is always
    "in front of the obstacle," the LoRA must learn to read sin/cos
    features to determine where to reduce cost.
    """
    features, targets, positions, radii, goal, orientations = batch
    obstacle_classes = list(model.obstacle_classes) if hasattr(model, 'obstacle_classes') else list(features.keys())
    H, W = 128, 128

    # Analyze user's edit direction RELATIVE to each obstacle's original
    # orientation. This relative offset is rotation-invariant.
    relative_edit_angles = analyze_edit_direction(
        orig_path, user_path, positions, orientations,
        obstacle_classes, threshold=5.0)

    # Precompute isotropic proximity mask (doesn't change with augmentation)
    proximity_mask = compute_proximity_mask(
        positions, obstacle_classes, relative_edit_angles, H, W, device)

    loss_history = []

    for region_mask, points, affected_classes in edit_regions:

        expert_models = {}
        x_0_gts = {}
        base_conditionings = {}

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=True)
            expert_models[cls] = model.experts[cls]
            x_0_gts[cls] = targets[cls].to(device)
            base_conditionings[cls] = features[cls].to(device)

        frozen_maps = {}
        for cls in model.obstacle_classes:
            if cls not in affected_classes:
                frozen_maps[cls] = targets[cls].to(device).detach()

        optimizer = AdamW(
            [p for expert in expert_models.values()
             for p in expert.parameters() if p.requires_grad],
            lr=lr,
        )

        user_path_target = make_path_target(user_path, H, W, device, sigma=5.0)
        orig_path_target = make_path_target(orig_path, H, W, device, sigma=5.0)

        # Snapshot pre-finetune outputs for regularization
        with torch.no_grad():
            pre_finetune_maps = {}
            for cls in affected_classes:
                B = x_0_gts[cls].shape[0]
                t_zero = torch.zeros((B,), device=device).long()
                x_t, noise = ddpm.q_sample(x_0_gts[cls], t_zero)
                noise_pred = expert_models[cls](x_t, t_zero, base_conditionings[cls])
                pre_finetune_maps[cls] = ddpm.predict_start_from_noise(
                    x_t, t_zero, noise_pred).detach()

        for epoch in tqdm(range(epochs), desc="IRL Co-Finetuning"):
            model.train()
            optimizer.zero_grad()

            # =========================================================
            # AUGMENTATION: Random rotation each epoch
            # =========================================================
            n_rotations = 8
            rotation_angles = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)
            delta_angle = rotation_angles[epoch % n_rotations]

            aug_orientations = {}
            for cls in obstacle_classes:
                orig = orientations[cls] if isinstance(orientations, dict) else orientations[0][cls]
                aug_orientations[cls] = [
                    (a.item() if hasattr(a, 'item') else float(a)) + delta_angle
                    for a in orig
                ]


            # Rewrite sin/cos conditioning channels
            aug_conditionings = {}
            for cls in affected_classes:
                aug_conditionings[cls] = rewrite_sin_cos_channels(
                    base_conditionings[cls], aug_orientations, cls,
                    obstacle_classes, positions, device)

            # Recompute edit mask with rotated orientations
            # The relative_edit_angles are fixed, so as aug_orientations rotate,
            # the absolute edit direction in the mask rotates with them
            aug_edit_mask = compute_directional_edit_mask(
                positions, aug_orientations, obstacle_classes,
                relative_edit_angles, H, W, device,
                angular_sigma=angular_sigma, spatial_sigma=15.0)

            # Unchanged direction mask = proximity minus edit mask
            unchanged_dir_mask = torch.clamp(proximity_mask - aug_edit_mask, min=0.0)
            unchanged_dir_mask = unchanged_dir_mask / (unchanged_dir_mask.max() + 1e-8)

            # =========================================================
            # Forward pass
            # =========================================================
            B = list(x_0_gts.values())[0].shape[0]
            t = torch.randint(0, ddpm.timesteps // 4, (B,), device=device).long()

            x_0_preds = {}
            noises = {}
            noise_preds = {}

            for cls in affected_classes:
                x_t, noise = ddpm.q_sample(x_0_gts[cls], t)
                noise_pred = expert_models[cls](x_t, t, aug_conditionings[cls])
                x_0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)
                x_0_preds[cls] = x_0_pred
                noises[cls] = noise
                noise_preds[cls] = noise_pred

            # Credit assignment
            with torch.no_grad():
                contributions = {}
                for cls in affected_classes:
                    contributions[cls] = (x_0_preds[cls].detach() * aug_edit_mask).sum()
                total_contrib = sum(contributions.values()) + 1e-8
                expert_weights = {
                    cls: (contributions[cls] / total_contrib).item()
                    for cls in affected_classes
                }

            # Loss 1: Diffusion loss
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

            # Loss 2: Planning loss masked by directional edit region
            user_target_in_edit = user_path_target * aug_edit_mask
            user_target_in_edit = user_target_in_edit / (user_target_in_edit.sum() + 1e-8)
            loss_plan = -(user_target_in_edit * torch.log(pred_visitation + 1e-8)).sum()

            # Loss 3: Preserve original path outside edit region
            unchanged_spatial = 1.0 - aug_edit_mask
            orig_target_unchanged = orig_path_target * unchanged_spatial
            orig_target_unchanged = orig_target_unchanged / (orig_target_unchanged.sum() + 1e-8)
            loss_preserve = -(orig_target_unchanged * torch.log(pred_visitation + 1e-8)).sum()

            # Loss 4: Directional regularization — penalize drift in non-edited directions
            loss_dir_reg = torch.tensor(0.0, device=device)
            for cls in affected_classes:
                delta = x_0_preds[cls] - pre_finetune_maps[cls]
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
                      f"diff={loss_diffusion.item():.4f}, plan={loss_plan.item():.4f}, "
                      f"dir_reg={loss_dir_reg.item():.4f}, "
                      f"aug_Δθ={np.degrees(delta_angle):.0f}°, "
                      f"weights=[{weight_str}]")

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=False)

    return loss_history
