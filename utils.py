



import numpy as np



def simulate_user_correction(path_rows, path_cols, obstacles, target_class, avoidance_radius, push_strength, influence_width=7):
    path_rows = np.array(path_rows)
    path_cols = np.array(path_cols)
    n_points = len(path_rows)

    total_shift_r = np.zeros(n_points)
    total_shift_c = np.zeros(n_points)

    for obs in obstacles[target_class]:
        obs_r, obs_c = obs['pos']
        dists = np.sqrt((path_rows - obs_r)**2 + (path_cols - obs_c)**2)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]

        if min_dist > avoidance_radius:
            continue

        #Push the pts away fromt he obstacle
        push_vec_c = path_cols[min_dist_idx] - obs_c
        push_vec_r = path_rows[min_dist_idx] - obs_r
        norm = np.sqrt(push_vec_c**2 + push_vec_r**2) + 1e-6
        push_dir_c = push_vec_c / norm
        push_dir_r = push_vec_r / norm

        # Gaussian influence along path
        indices = np.arange(n_points)
        gaussian_weights = np.exp(-0.5 * ((indices - min_dist_idx) / influence_width)**2)
        proximity_scale = np.clip(1.5 - (min_dist / avoidance_radius), 0.5, 1.5)
        
        total_shift_r += push_dir_r * push_strength * gaussian_weights * proximity_scale
        total_shift_c += push_dir_c * push_strength * gaussian_weights * proximity_scale

    adj_rows = path_rows + total_shift_r
    adj_cols = path_cols + total_shift_c
    
    # Blend to keep start/end fixed
    blend_mask = np.ones(n_points)
    blend_mask[:5] = np.linspace(0, 1, 5)
    blend_mask[-5:] = np.linspace(1, 0, 5)
    adj_rows = path_rows + (adj_rows - path_rows) * blend_mask
    adj_cols = path_cols + (adj_cols - path_cols) * blend_mask
    
    return adj_rows, adj_cols


def compute_interaction_mask(delta_map):
    """
    Convert geometric delta to interaction mask.
    Mask = 1.0 where learning should happen, 0.0 elsewhere.
    """
    mask = np.abs(delta_map)
    max_val = mask.max() + 1e-6
    mask = mask / max_val
    return mask

