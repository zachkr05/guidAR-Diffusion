
import numpy as np
from scipy.ndimage import distance_transform_edt
from DataGenerator.sim  import NUM_CLASSES


import skimage.graph
import torch
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better for structural learning than linear.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)



def make_edf_maps(H, W, obstacles_by_class, normalize=True):
    
    
    edf_maps = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    max_dist = np.sqrt(H**2 + W**2)

    for class_id in range(NUM_CLASSES):

        obstacles = obstacles_by_class.get(class_id, [])
        
        mask = np.zeros((H,W), dtype=bool)
        
        # Get pos of obstacles and set 1 to free space and 0 to obstacles
        # BINARY MASK
        if len(obstacles) > 0:
            for obs in obstacles:
                r, c = obs['pos']
                r_idx = np.clip(r, 1,H-1)
                c_idx = np.clip(c, 1, W-1)
                mask[r_idx, c_idx] = False
        else:
            if normalize:
                edf_maps[class_id] = np.ones((H,W), dtype=np.float32)
            else:
                edf_maps[class_id] = np.full((H,W),max_dist, dtype=np.float32)
            continue

        edf = distance_transform_edt(mask)

        if normalize:
            edf = edf / max_dist

        edf_maps[class_id] = edf.astype(np.float32)

    return edf_maps


def make_orientation_maps(H, W, obstacles_by_class):
    
    orient_maps = np.zeros((NUM_CLASSES *2, H, W), dtype = np.float32)

    for class_id in range(NUM_CLASSES):
        obstacles = obstacles_by_class.get(class_id, [])

        sparse_sin = np.zeros((H,W), dtype=np.float32)
        sparse_cos = np.zeros((H, W), dtype = np.float32)

        binary_mask = np.ones((H, W), dtype=int)

        if not obstacles:
            continue

        for obs in obstacles:
            r, c = obs['pos']
            theta = obs['orientation'] # Assuming radians. If degrees, convert first!
            
            r_idx = int(np.clip(r, 0, H-1))
            c_idx = int(np.clip(c, 0, W-1))
            
            # Paint the value
            sparse_sin[r_idx, c_idx] = np.sin(theta)
            sparse_cos[r_idx, c_idx] = np.cos(theta)
            binary_mask[r_idx, c_idx] = 0

        dist_map, indices = distance_transform_edt(binary_mask, return_distances=True, return_indices=True)
        
        nearest_r = indices[0]
        nearest_c = indices[1]

        dense_sin = sparse_sin[nearest_r, nearest_c]
        dense_cos = sparse_cos[nearest_r, nearest_c]

        final_sin = dense_sin
        final_cos = dense_cos

        orient_maps[class_id * 2]     = final_sin
        orient_maps[class_id * 2 + 1] = final_cos
    
    return orient_maps

def find_path(generated_costmap, allow_diagonal=False):

        costmap_np = generated_costmap.squeeze().cpu().numpy() # convert from GPU --> CPU

        traversal_costs = np.exp(costmap_np*3.0) # current each pixel is -1,1; so we need to get only positives costs therefore we can use the exp function

        mcp = skimage.graph.MCP(traversal_costs, fully_connected=False) #fully connected = False disables diagonal movement

        mcp.find_costs(starts=[start_rc])

        path = mcp.traceback(goal_rc)

        return path

# --- 3. User Correction Simulation ---
def simulate_user_correction(path_rows, path_cols, obstacles, target_class, avoidance_radius, push_strength, influence_width=7):
    """
    Simulate a user correcting the path to avoid obstacles of a specific class.
    
    Args:
        path_rows: array of row coordinates along path
        path_cols: array of col coordinates along path
        obstacles: dict {class_id: [{'pos': (r, c), ...}, ...]}
        target_class: which class to avoid more strongly
        avoidance_radius: distance threshold for correction
        push_strength: how much to push the path away
        influence_width: Gaussian spread along path (in points)
    
    Returns:
        adj_rows, adj_cols: adjusted path coordinates
    """
    path_rows = np.array(path_rows)
    path_cols = np.array(path_cols)
    n_points = len(path_rows)
    total_shift_r = np.zeros(n_points)
    total_shift_c = np.zeros(n_points)
    
    if target_class not in obstacles:
        return path_rows, path_cols
    
    for obs in obstacles[target_class]:
        obs_r, obs_c = obs['pos']
        dists = np.sqrt((path_rows - obs_r)**2 + (path_cols - obs_c)**2)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist > avoidance_radius:
            continue
            
        # Push the pts away from the obstacle
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


