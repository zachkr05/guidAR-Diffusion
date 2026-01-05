
import copy
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d



def compute_geometric_delta(
    original_trajectory ,
    user_trajectory ,
    H: int,
    W: int,
    cost_increase: float = 0.4,  # REDUCED - more conservative
    cost_decrease: float = 0.2   # REDUCED - more conservative
) -> np.ndarray:
    """Compute costmap delta geometrically from trajectory difference.
    
    Conservative cost modifications for stable, interpretable learning.
    """
    
    def resample(traj, n):
        traj = np.array(traj, dtype=np.float32)
        if len(traj) < 2:
            return np.tile(traj[0], (n, 1))
        diffs = np.diff(traj, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dists = np.concatenate([[0], np.cumsum(dists)])
        total = cum_dists[-1]
        if total < 1e-6:
            return np.tile(traj[0], (n, 1))
        sample_d = np.linspace(0, total, n)
        interp_r = interp1d(cum_dists, traj[:, 0], kind='linear', fill_value='extrapolate')
        interp_c = interp1d(cum_dists, traj[:, 1], kind='linear', fill_value='extrapolate')
        return np.stack([interp_r(sample_d), interp_c(sample_d)], axis=1)
    
    orig = resample(original_trajectory, 100)
    user = resample(user_trajectory, 100)
    
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
    
    # Signed difference
    diff = dist_to_user - dist_to_orig
    
    max_influence = 15.0  # REDUCED - tighter region
    near_traj = (dist_to_orig < max_influence) | (dist_to_user < max_influence)
    
    delta = np.zeros((H, W), dtype=np.float32)
    
    # Avoided region - linear scaling (not exponential)
        
    avoided = (diff > 2.0) & near_traj
    if avoided.any():
        # Exponential: rises quickly near obstacle, saturates at 1
        avoided_strength = 1.0 - np.exp(-diff[avoided] / 10.0)
        avoided_strength = np.clip(avoided_strength, 0, 1)
        delta[avoided] = avoided_strength * cost_increase

    # Preferred region - exponential valley
    preferred = (diff < -2.0) & near_traj
    if preferred.any():
        preferred_strength = 1.0 - np.exp(diff[preferred] / 10.0)  # Note: diff is negative here
        preferred_strength = np.clip(preferred_strength, 0, 1)
        delta[preferred] = -preferred_strength * cost_decrease

    # Fill between with moderate barrier
    between_mask = np.zeros((H, W), dtype=np.float32)
    for i in range(len(orig)):
        mid = (orig[i] + user[i]) / 2
        dist_between = np.linalg.norm(user[i] - orig[i])
        if dist_between > 3:
            sigma = max(dist_between / 3, 2)
            d_sq = (rows - mid[0])**2 + (cols - mid[1])**2
            blob = np.exp(-d_sq / (2 * sigma**2))
            between_mask = np.maximum(between_mask, blob)
    
    # Moderate barrier in between region
    delta = delta + between_mask * cost_increase * 0.5
    
    # Smooth but preserve strength
    delta = ndimage.gaussian_filter(delta, sigma=2.0)
    
    return delta
