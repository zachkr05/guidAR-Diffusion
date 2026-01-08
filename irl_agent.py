"""
Inverse Reinforcement Learning Agent - Geometric Delta Computation

Implements the "Push/Force Field" model for learning from user trajectory corrections.

Key insight: When a user moves point A to point B, it implies a "repulsive force" at A.
The cost should rise AT THE ORIGINAL LOCATION (where the user didn't want to go),
not symmetrically between the paths.

This creates a "wall" exactly where the user doesn't want to go, without 
unnecessarily lowering costs in empty space.
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d


def resample_trajectory(traj, n_points):
    """Resample a trajectory to have exactly n_points, evenly spaced by arc length."""
    traj = np.array(traj, dtype=np.float32)
    if len(traj) < 2:
        return np.tile(traj[0], (n_points, 1))
    
    # Compute cumulative arc length
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-6:
        return np.tile(traj[0], (n_points, 1))
    
    # Sample at evenly spaced arc lengths
    sample_distances = np.linspace(0, total_length, n_points)
    interp_r = interp1d(cumulative_length, traj[:, 0], kind='linear', fill_value='extrapolate')
    interp_c = interp1d(cumulative_length, traj[:, 1], kind='linear', fill_value='extrapolate')
    
    return np.stack([interp_r(sample_distances), interp_c(sample_distances)], axis=1)


def compute_displacement_field(orig_pts, user_pts, H, W):
    """
    Compute a vector field representing the user's "push" at each location.
    
    For each pair of corresponding points (orig, user), we create a 
    displacement vector that points from orig toward user. This represents
    the direction the user wanted to move away from obstacles.
    
    Returns:
        displacement_magnitude: [H, W] - how strongly each location was "pushed"
        displacement_vectors: [H, W, 2] - the direction of push at each location
    """
    displacement_magnitude = np.zeros((H, W), dtype=np.float32)
    displacement_vectors = np.zeros((H, W, 2), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    rows, cols = np.ogrid[:H, :W]
    
    for orig_pt, user_pt in zip(orig_pts, user_pts):
        # Displacement vector at this point
        disp = user_pt - orig_pt
        disp_magnitude = np.linalg.norm(disp)
        
        if disp_magnitude < 0.5:
            continue  # No significant displacement
        
        # Normalize direction
        disp_dir = disp / disp_magnitude
        
        # Create influence region around the ORIGINAL point
        # (This is where the user DIDN'T want to go)
        sigma = max(disp_magnitude * 0.5, 3.0)
        dist_sq = (rows - orig_pt[0])**2 + (cols - orig_pt[1])**2
        influence = np.exp(-dist_sq / (2 * sigma**2))
        
        # Weight by displacement magnitude (larger corrections = stronger signal)
        weight = influence * disp_magnitude
        
        displacement_magnitude += weight
        displacement_vectors[:, :, 0] += weight * disp_dir[0]
        displacement_vectors[:, :, 1] += weight * disp_dir[1]
        weight_sum += influence
    
    # Normalize
    mask = weight_sum > 1e-6
    displacement_vectors[mask, 0] /= weight_sum[mask]
    displacement_vectors[mask, 1] /= weight_sum[mask]
    
    return displacement_magnitude, displacement_vectors


def compute_geometric_delta(
    original_trajectory,
    user_trajectory,
    H: int,
    W: int,
    cost_increase: float = 0.15,
    cost_decrease: float = 0.15,
    n_samples: int = 100
) -> np.ndarray:
    """
    Compute costmap delta using the Push/Force Field model.
    
    Key principles:
    1. INCREASE cost where the user LEFT (the avoided region)
    2. Slightly DECREASE cost where the user WENT (to reinforce the preference)
    3. The "barrier" should be on the ORIGINAL path side, not symmetric
    4. Larger displacements = stronger cost modification
    
    Ars:
        original_trajectory: List of (row, col) tuples - planned path
        user_trajectory: List of (row, col) tuples - user's corrected path
        H, W: Grid dimensions
        cost_increase: Maximum cost increase in avoided regions
        cost_decrease: Maximum cost decrease in preferred regions
        n_samples: Number of points to resample trajectories to
    
    Returns:
        delta: [H, W] array of cost modifications
    """
    # Resample trajectories to equal number of points
    orig = resample_trajectory(original_trajectory, n_samples)
    user = resample_trajectory(user_trajectory, n_samples)
    
    rows, cols = np.ogrid[:H, :W]
    
    # =========================================================================
    # Step 1: Compute displacement field (the "push" vectors)
    # =========================================================================
    disp_magnitude, disp_vectors = compute_displacement_field(orig, user, H, W)
    
    # =========================================================================
    # Step 2: Create cost increase at AVOIDED locations (where user left)
    # =========================================================================
    # The avoided region is where the original path was, weighted by how much
    # the user pushed away from it
    
    avoided_cost = np.zeros((H, W), dtype=np.float32)
    
    for i, (orig_pt, user_pt) in enumerate(zip(orig, user)):
        disp = user_pt - orig_pt
        disp_mag = np.linalg.norm(disp)
        
        if disp_mag < 1.0:
            continue
        
        # Direction user pushed (normalized)
        push_dir = disp / disp_mag
        
        # Create asymmetric blob: strong on the ORIGINAL side, fading toward user
        # Center the blob BEHIND the original point (opposite to push direction)
        blob_center = orig_pt - push_dir * (disp_mag * 0.3)
        
        dist_sq = (rows - blob_center[0])**2 + (cols - blob_center[1])**2
        
        # Elongated in the push direction (elliptical)
        # Points perpendicular to push direction decay faster
        vec_to_point_r = rows - blob_center[0]
        vec_to_point_c = cols - blob_center[1]
        
        # Project onto push direction and perpendicular
        proj_parallel = vec_to_point_r * push_dir[0] + vec_to_point_c * push_dir[1]
        proj_perp_sq = dist_sq - proj_parallel**2
        
        # Asymmetric: extend further backward (opposite to push), shorter forward
        sigma_backward = disp_mag * 0.8
        sigma_forward = disp_mag * 0.3
        sigma_perp = disp_mag * 0.5
        
        # Use different sigma based on direction
        sigma_parallel = np.where(proj_parallel < 0, sigma_backward, sigma_forward)
        
        # Elliptical Gaussian
        exponent = (proj_parallel**2 / (2 * sigma_parallel**2 + 1e-6) + 
                    proj_perp_sq / (2 * sigma_perp**2 + 1e-6))
        blob = np.exp(-exponent)
        
        # Scale by displacement magnitude (larger push = more important)
        strength = min(disp_mag / 10.0, 1.0)
        avoided_cost = np.maximum(avoided_cost, blob * strength)
    
    # =========================================================================
    # Step 3: Create slight cost DECREASE at preferred locations (where user went)
    # =========================================================================
    preferred_cost = np.zeros((H, W), dtype=np.float32)
    
    # Distance to user path
    dist_to_user = np.full((H, W), np.inf)
    for pt in user:
        d = np.sqrt((rows - pt[0])**2 + (cols - pt[1])**2)
        dist_to_user = np.minimum(dist_to_user, d)
    
    # Only decrease cost very close to the user's chosen path
    # and only where there was significant displacement
    sigma_prefer = 3.0
    prefer_region = np.exp(-dist_to_user**2 / (2 * sigma_prefer**2))
    
    # Weight by local displacement magnitude
    prefer_weight = np.zeros((H, W), dtype=np.float32)
    for orig_pt, user_pt in zip(orig, user):
        disp_mag = np.linalg.norm(user_pt - orig_pt)
        if disp_mag > 1.0:
            d = np.sqrt((rows - user_pt[0])**2 + (cols - user_pt[1])**2)
            local_weight = np.exp(-d**2 / (2 * 5.0**2)) * min(disp_mag / 15.0, 1.0)
            prefer_weight = np.maximum(prefer_weight, local_weight)
    
    preferred_cost = prefer_region * prefer_weight
    
    # =========================================================================
    # Step 4: Combine into final delta
    # =========================================================================
    delta = avoided_cost * cost_increase - preferred_cost * cost_decrease
    
    # =========================================================================
    # Step 5: Add a "barrier" between paths where displacement was large
    # =========================================================================
    # This prevents the planner from cutting back through
    barrier = np.zeros((H, W), dtype=np.float32)
    
    for orig_pt, user_pt in zip(orig, user):
        disp_mag = np.linalg.norm(user_pt - orig_pt)
        if disp_mag < 3.0:
            continue
        
        # Create barrier blobs along the line from orig to user,
        # but weighted toward the original side
        n_barrier_pts = max(int(disp_mag / 3), 2)
        for t in np.linspace(0.1, 0.5, n_barrier_pts):  # Bias toward original
            barrier_pt = orig_pt + t * (user_pt - orig_pt)
            d_sq = (rows - barrier_pt[0])**2 + (cols - barrier_pt[1])**2
            sigma_barrier = disp_mag * 0.25
            blob = np.exp(-d_sq / (2 * sigma_barrier**2))
            barrier = np.maximum(barrier, blob * 0.5)
    
    delta = delta + barrier * cost_increase
    
    # =========================================================================
    # Step 6: Light smoothing to prevent artifacts
    # =========================================================================
    delta = ndimage.gaussian_filter(delta, sigma=1.5)
    
    return delta.astype(np.float32)


def compute_geometric_delta_simple(
    original_trajectory,
    user_trajectory,
    H: int,
    W: int,
    cost_increase: float = 0.4,
    cost_decrease: float = 0.2
) -> np.ndarray:
    """
    Simplified version of geometric delta (legacy compatibility).
    
    Uses distance-based approach with asymmetric barrier.
    """
    orig = resample_trajectory(original_trajectory, 100)
    user = resample_trajectory(user_trajectory, 100)
    
    rows, cols = np.ogrid[:H, :W]
    
    # Distance to each trajectory
    dist_to_orig = np.full((H, W), np.inf)
    for pt in orig:
        d = np.sqrt((rows - pt[0])**2 + (cols - pt[1])**2)
        dist_to_orig = np.minimum(dist_to_orig, d)
    
    dist_to_user = np.full((H, W), np.inf)
    for pt in user:
        d = np.sqrt((rows - pt[0])**2 + (cols - pt[1])**2)
        dist_to_user = np.minimum(dist_to_user, d)
    
    # Signed difference: positive where user is further (avoided region)
    diff = dist_to_user - dist_to_orig
    
    max_influence = 15.0
    near_traj = (dist_to_orig < max_influence) | (dist_to_user < max_influence)
    
    delta = np.zeros((H, W), dtype=np.float32)
    
    # Avoided region (user moved away from here)
    avoided = (diff > 2.0) & near_traj
    if avoided.any():
        avoided_strength = 1.0 - np.exp(-diff[avoided] / 10.0)
        avoided_strength = np.clip(avoided_strength, 0, 1)
        delta[avoided] = avoided_strength * cost_increase
    
    # Preferred region (user moved toward here)
    preferred = (diff < -2.0) & near_traj
    if preferred.any():
        preferred_strength = 1.0 - np.exp(diff[preferred] / 10.0)
        preferred_strength = np.clip(preferred_strength, 0, 1)
        delta[preferred] = -preferred_strength * cost_decrease
    
    # Asymmetric barrier between paths (biased toward original)
    between_mask = np.zeros((H, W), dtype=np.float32)
    for i in range(len(orig)):
        disp = user[i] - orig[i]
        disp_mag = np.linalg.norm(disp)
        if disp_mag > 3:
            # Barrier point biased toward original (30% of the way)
            barrier_pt = orig[i] + 0.3 * disp
            sigma = max(disp_mag / 3, 2)
            d_sq = (rows - barrier_pt[0])**2 + (cols - barrier_pt[1])**2
            blob = np.exp(-d_sq / (2 * sigma**2))
            between_mask = np.maximum(between_mask, blob)
    
    delta = delta + between_mask * cost_increase * 0.5
    
    # Smooth
    delta = ndimage.gaussian_filter(delta, sigma=2.0)
    
    return delta
