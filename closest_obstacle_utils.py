"""
Utilities for identifying and isolating the obstacle closest to user edits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_path_delta_centroid(orig_path: np.ndarray, user_path: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of where the user's correction differs most from original.
    
    Args:
        orig_path: (N, 2) original planned path [x, y]
        user_path: (M, 2) user-corrected path [x, y]
    
    Returns:
        (2,) centroid of the edit region [x, y]
    """
    # Resample both paths to same length for comparison
    n_points = 100
    
    def resample_path(path, n):
        if len(path) < 2:
            return path
        # Cumulative distance along path
        diffs = np.diff(path, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cum_dist = np.concatenate([[0], np.cumsum(dists)])
        total_dist = cum_dist[-1]
        
        if total_dist < 1e-6:
            return np.tile(path[0], (n, 1))
        
        # Interpolate at uniform distances
        target_dists = np.linspace(0, total_dist, n)
        resampled = np.zeros((n, 2))
        for i, d in enumerate(target_dists):
            idx = np.searchsorted(cum_dist, d, side='right') - 1
            idx = np.clip(idx, 0, len(path) - 2)
            t = (d - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx] + 1e-8)
            resampled[i] = path[idx] + t * (path[idx + 1] - path[idx])
        return resampled
    
    orig_resampled = resample_path(orig_path, n_points)
    user_resampled = resample_path(user_path, n_points)
    
    # Compute per-point displacement
    displacements = np.linalg.norm(user_resampled - orig_resampled, axis=1)
    
    # Weight points by displacement magnitude to find edit centroid
    weights = displacements / (displacements.sum() + 1e-8)
    centroid = (weights[:, None] * user_resampled).sum(axis=0)
    
    return centroid


def find_closest_obstacle(
    edit_centroid: np.ndarray,
    positions: List[Dict[str, np.ndarray]],
    radii: List[Dict[str, np.ndarray]],
    target_class: str,
    batch_idx: int = 0
) -> Tuple[int, float]:
    """
    Find which obstacle of target_class is closest to the edit centroid.
    
    Args:
        edit_centroid: (2,) [x, y] location of edit
        positions: List of dicts mapping class -> (N_obs, 2) positions
        radii: List of dicts mapping class -> (N_obs,) radii
        target_class: Which class to search within (e.g., "chair")
        batch_idx: Which batch element to look at
    
    Returns:
        (closest_idx, distance) - index of closest obstacle and its distance
    """
    obs_positions = positions[batch_idx][target_class]  # (N_obs, 2)
    obs_radii = radii[batch_idx][target_class]  # (N_obs,)
    
    if len(obs_positions) == 0:
        raise ValueError(f"No obstacles of class {target_class} found")
    
    # Compute distance from edit centroid to each obstacle center
    distances = np.linalg.norm(obs_positions - edit_centroid, axis=1)
    
    # Optionally account for radius (distance to edge rather than center)
    distances_to_edge = distances - obs_radii
    
    closest_idx = np.argmin(distances_to_edge)
    closest_dist = distances_to_edge[closest_idx]
    
    return closest_idx, closest_dist


def create_single_obstacle_mask(
    position: np.ndarray,
    radius: float,
    H: int,
    W: int,
    device: torch.device,
    margin: float = 1.5
) -> torch.Tensor:
    """
    Create a soft mask that isolates a single obstacle region.
    
    Args:
        position: (2,) [x, y] center of obstacle
        radius: Radius of obstacle
        H, W: Grid dimensions
        device: Torch device
        margin: Multiplier for radius to create soft falloff region
    
    Returns:
        (1, 1, H, W) mask tensor, 1.0 at obstacle, falls off to 0
    """
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
   )
    
    cx, cy = position[0], position[1]
    dist_from_center = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    
    # Soft mask: 1 inside radius, smooth falloff to 0 at margin * radius
    inner_radius = radius
    outer_radius = radius * margin
    
    mask = torch.zeros((H, W), device=device)
    mask[dist_from_center <= inner_radius] = 1.0
    
    # Smooth transition zone
    transition_zone = (dist_from_center > inner_radius) & (dist_from_center < outer_radius)
    if transition_zone.any():
        t = (dist_from_center[transition_zone] - inner_radius) / (outer_radius - inner_radius)
        mask[transition_zone] = 1.0 - t  # Linear falloff
    
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def create_focused_target(
    full_target: torch.Tensor,
    positions: List[Dict[str, np.ndarray]],
    radii: List[Dict[str, np.ndarray]],
    target_class: str,
    closest_idx: int,
    batch_idx: int = 0,
    margin: float = 2.0
) -> torch.Tensor:
    """
    Create a target costmap that only includes the closest obstacle.
    
    Args:
        full_target: (B, 1, H, W) original target with all obstacles of class
        positions, radii: Obstacle metadata
        target_class: Class being trained
        closest_idx: Index of closest obstacle
        batch_idx: Batch element index
        margin: Margin multiplier for mask
    
    Returns:
        (B, 1, H, W) target with only the closest obstacle
    """
    B, C, H, W = full_target.shape
    device = full_target.device
    
    obs_pos = positions[batch_idx][target_class][closest_idx]
    obs_rad = radii[batch_idx][target_class][closest_idx]
    
    mask = create_single_obstacle_mask(obs_pos, obs_rad, H, W, device, margin)
    
    # Apply mask to isolate just this obstacle
    # The mask keeps the obstacle region, zeros elsewhere
    focused_target = full_target * mask
    
    return focused_target


def get_focused_training_batch(
    batch: Tuple,
    orig_path: np.ndarray,
    user_path: np.ndarray,
    target_class: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Main entry point: Given a batch and user correction, return focused training data.
    
    Args:
        batch: (features, targets, positions, radii, goal) from dataloader
        orig_path: Original planned path
        user_path: User-corrected path
        target_class: Class to finetune (e.g., "chair")
        device: Torch device
    
    Returns:
        (focused_target, mask, closest_idx):
            - focused_target: Target costmap with only relevant obstacle
            - mask: The isolation mask for loss weighting
            - closest_idx: Which obstacle was selected
    """
    features, targets, positions, radii, goal = batch
    
    # 1. Find where the user made their edit
    edit_centroid = compute_path_delta_centroid(orig_path, user_path)
    print(f"Edit centroid detected at: ({edit_centroid[0]:.1f}, {edit_centroid[1]:.1f})")
    
    # 2. Find closest obstacle to that edit
    closest_idx, dist = find_closest_obstacle(
        edit_centroid, positions, radii, target_class, batch_idx=0
    )
    obs_pos = positions[0][target_class][closest_idx]
    obs_rad = radii[0][target_class][closest_idx]
    print(f"Closest {target_class} is #{closest_idx} at ({obs_pos[0]:.1f}, {obs_pos[1]:.1f}), "
          f"radius={obs_rad:.1f}, distance to edit={dist:.1f}")
    
    # 3. Create focused target
    full_target = targets[target_class].to(device)
    B, C, H, W = full_target.shape
    
    focused_target = create_focused_target(
        full_target, positions, radii, target_class, closest_idx, 
        batch_idx=0, margin=2.0
    )
    
    # 4. Also return the mask for potential loss weighting
    mask = create_single_obstacle_mask(obs_pos, obs_rad, H, W, device, margin=2.0)
    
    return focused_target, mask, closest_idx


# ============================================================================
# Alternative: Masked Loss approach (trains on full map but weights loss)
# ============================================================================

def compute_focused_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    background_weight: float = 0.1
) -> torch.Tensor:
    """
    Compute MSE loss with higher weight on the masked (relevant) region.
    
    Args:
        pred: (B, 1, H, W) predicted costmap
        target: (B, 1, H, W) ground truth
        mask: (B, 1, H, W) mask for relevant obstacle (1 = important)
        background_weight: How much to weight loss outside mask
    
    Returns:
        Weighted MSE loss
    """
    # Weight map: 1.0 where mask is active, background_weight elsewhere
    weights = mask + background_weight * (1.0 - mask)
    
    squared_error = (pred - target) ** 2
    weighted_loss = (squared_error * weights).mean()
    
    return weighted_loss
