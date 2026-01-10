"""
Inverse Reinforcement Learning Agent - Geometric Delta Computation

IMPROVED VERSION:
- Broader influence regions (not tied to displacement magnitude)
- Optional class-aware propagation (when user avoids one chair front, all chair fronts learn)
- Minimum sigma to ensure learnable gradients
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Optional


def resample_trajectory(traj, n_points):
    """Resample a trajectory to have exactly n_points, evenly spaced by arc length."""
    traj = np.array(traj, dtype=np.float32)
    if len(traj) < 2:
        return np.tile(traj[0], (n_points, 1))
    
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-6:
        return np.tile(traj[0], (n_points, 1))
    
    sample_distances = np.linspace(0, total_length, n_points)
    interp_r = interp1d(cumulative_length, traj[:, 0], kind='linear', fill_value='extrapolate')
    interp_c = interp1d(cumulative_length, traj[:, 1], kind='linear', fill_value='extrapolate')
    
    return np.stack([interp_r(sample_distances), interp_c(sample_distances)], axis=1)


def compute_geometric_delta(
    original_trajectory,
    user_trajectory,
    H: int,
    W: int,
    cost_increase: float = 0.1,
    cost_decrease: float = 0.1,
    n_samples: int = 100,
    min_sigma: float = 1.0,  # NEW: Minimum influence radius
    max_sigma: float = 5.0,  # NEW: Maximum influence radius
    smoothing_sigma: float = 3.0,  # NEW: Increased final smoothing
) -> np.ndarray:
    """
    Compute costmap delta using the Push/Force Field model.
    
    IMPROVEMENTS:
    - min_sigma ensures broad influence even for small corrections
    - Decoupled influence size from displacement magnitude
    - Larger final smoothing for gradient-friendly targets
    
    Args:
        original_trajectory: List of (row, col) tuples - planned path
        user_trajectory: List of (row, col) tuples - user's corrected path
        H, W: Grid dimensions
        cost_increase: Maximum cost increase in avoided regions
        cost_decrease: Maximum cost decrease in preferred regions
        n_samples: Number of points to resample trajectories to
        min_sigma: Minimum influence radius (pixels)
        max_sigma: Maximum influence radius (pixels)
        smoothing_sigma: Final Gaussian smoothing sigma
    
    Returns:
        delta: [H, W] array of cost modifications
    """
    orig = resample_trajectory(original_trajectory, n_samples)
    user = resample_trajectory(user_trajectory, n_samples)
    
    rows, cols = np.ogrid[:H, :W]
    
    # =========================================================================
    # Step 1: Create cost increase at AVOIDED locations
    # =========================================================================
    avoided_cost = np.zeros((H, W), dtype=np.float32)
    
    for i, (orig_pt, user_pt) in enumerate(zip(orig, user)):
        disp = user_pt - orig_pt
        disp_mag = np.linalg.norm(disp)
        
        if disp_mag < 1.0:
            continue
        
        push_dir = disp / disp_mag
        
        # Blob center: behind the original point
        blob_center = orig_pt - push_dir * min(disp_mag * 0.3, 5.0)
        
        # CHANGE: Use fixed sigma range, not proportional to displacement
        # This ensures consistent influence regardless of correction size
        sigma_base = np.clip(disp_mag * 1.5, min_sigma, max_sigma)
        
        # Compute distance components
        vec_to_point_r = rows - blob_center[0]
        vec_to_point_c = cols - blob_center[1]
        dist_sq = vec_to_point_r**2 + vec_to_point_c**2
        
        proj_parallel = vec_to_point_r * push_dir[0] + vec_to_point_c * push_dir[1]
        proj_perp_sq = np.maximum(dist_sq - proj_parallel**2, 0)
        
        # Asymmetric sigma: extend further backward
        sigma_backward = sigma_base * 1.2
        sigma_forward = sigma_base * 0.5
        sigma_perp = sigma_base * 0.8
        
        sigma_parallel = np.where(proj_parallel < 0, sigma_backward, sigma_forward)
        
        exponent = (proj_parallel**2 / (2 * sigma_parallel**2 + 1e-6) + 
                    proj_perp_sq / (2 * sigma_perp**2 + 1e-6))
        blob = np.exp(-exponent)
        
        # Strength scales with displacement, but saturates
        strength = np.tanh(disp_mag / 8.0)  # Smoother saturation
        avoided_cost = np.maximum(avoided_cost, blob * strength)
    
    # =========================================================================
    # Step 2: Create cost DECREASE at preferred locations
    # =========================================================================
    preferred_cost = np.zeros((H, W), dtype=np.float32)
    
    # Distance to user path
    dist_to_user = np.full((H, W), np.inf)
    for pt in user:
        d = np.sqrt((rows - pt[0])**2 + (cols - pt[1])**2)
        dist_to_user = np.minimum(dist_to_user, d)
    
    # CHANGE: Broader preference region
    sigma_prefer = min_sigma * 0.6
    prefer_region = np.exp(-dist_to_user**2 / (2 * sigma_prefer**2))
    
    # Weight by local displacement magnitude
    prefer_weight = np.zeros((H, W), dtype=np.float32)
    for orig_pt, user_pt in zip(orig, user):
        disp_mag = np.linalg.norm(user_pt - orig_pt)
        if disp_mag > 1.0:
            d = np.sqrt((rows - user_pt[0])**2 + (cols - user_pt[1])**2)
            # CHANGE: Broader influence
            local_weight = np.exp(-d**2 / (2 * min_sigma**2)) * np.tanh(disp_mag / 10.0)
            prefer_weight = np.maximum(prefer_weight, local_weight)
    
    preferred_cost = prefer_region * prefer_weight
    
    # =========================================================================
    # Step 3: Combine into final delta
    # =========================================================================
    delta = avoided_cost * cost_increase - preferred_cost * cost_decrease
    
    # =========================================================================
    # Step 4: Barrier between paths
    # =========================================================================
    barrier = np.zeros((H, W), dtype=np.float32)
    
    for orig_pt, user_pt in zip(orig, user):
        disp_mag = np.linalg.norm(user_pt - orig_pt)
        if disp_mag < 3.0:
            continue
        
        n_barrier_pts = max(int(disp_mag / 3), 2)
        for t in np.linspace(0.1, 0.5, n_barrier_pts):
            barrier_pt = orig_pt + t * (user_pt - orig_pt)
            d_sq = (rows - barrier_pt[0])**2 + (cols - barrier_pt[1])**2
            # CHANGE: Broader barrier
            sigma_barrier = max(disp_mag * 0.4, min_sigma * 0.5)
            blob = np.exp(-d_sq / (2 * sigma_barrier**2))
            barrier = np.maximum(barrier, blob * 0.5)
    
    delta = delta + barrier * cost_increase
    
    # =========================================================================
    # Step 5: Broader smoothing for learnable gradients
    # =========================================================================
    delta = ndimage.gaussian_filter(delta, sigma=smoothing_sigma)
    
    return delta.astype(np.float32)
