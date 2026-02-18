#utils.py
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle
from pathlib import Path
import torch
from matplotlib.widgets import RectangleSelector, CheckButtons
from matplotlib import cm
import numpy as np
from skimage.graph import route_through_array
from scipy.interpolate import BSpline
from .spline import *
from torch.utils.data.dataloader import default_collate
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from skimage.draw import polygon

def compute_spline_path(costmap_np, goal, k=3, num_ctrl_pts=10):

    """
    Compute a shortest path through a costmap and fit a B-spline to it.

    Returns:
        path_xy: (N,2) raw path [x, y]
        spline_xy: (M,2) smooth spline [x, y]
        P: control points
        U: knot vector
    """
    H, W = costmap_np.shape
    # Normalize for route_through_array
    mn, mx = costmap_np.min(), costmap_np.max()
    normed = (costmap_np - mn) / (mx - mn + 1e-8) + 1e-8

    goal_y = int(np.clip(goal[0], 0, H - 1))
    goal_x = int(np.clip(goal[1], 0, W - 1))

    path_result = route_through_array(normed, [0, 0], [goal_y, goal_x],
                                       fully_connected=True, geometric=True)
    path_array = np.array(path_result[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]
    path_xy = np.column_stack([x_np, y_np])

    x_s, y_s, P, U = generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)
    spline_xy = np.column_stack([x_s, y_s])

    return path_xy, spline_xy, P, U



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



def get_edit_regions(orig_path, user_path, obstacle_classes, batch, 
                     height=128, width=128, min_area=50):
    """
    Identify distinct edit regions by detecting sign changes in the 
    cross product (signed area) between paths.
    
    Returns:
        List of [class_contributions, edit_points, area_mask] tuples
    """
   
    _, _, positions, radii, _, _ = batch
    prob_dict = obtain_probabilities(obstacle_classes, positions, radii, height, width)
    
    # Resample both paths to same number of points
    n_samples = max(len(orig_path), len(user_path), 800)
    
    def resample_path(path, n):
        t_orig = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, n)
        fx = interp1d(t_orig, path[:, 0], kind='linear')
        fy = interp1d(t_orig, path[:, 1], kind='linear')
        return np.column_stack([fx(t_new), fy(t_new)])
    
    orig_resampled = resample_path(orig_path, n_samples)
    user_resampled = resample_path(user_path, n_samples)
    
    # Compute cross product (signed area indicator) at each point
    # Positive = user is to the left of orig, Negative = user is to the right
    diff = user_resampled - orig_resampled
    
    # Use cross product with path tangent to get signed distance
    tangent = np.gradient(orig_resampled, axis=0)
    cross = diff[:, 0] * tangent[:, 1] - diff[:, 1] * tangent[:, 0]
    
    # Also compute absolute distance for threshold
    distances = np.linalg.norm(diff, axis=1)
    
    # Find where paths are "together" (distance < threshold)
    threshold = 2.0
    is_together = distances < threshold
    
    # Find sign of cross product (which side user path is on)
    sign = np.sign(cross)
    sign[is_together] = 0  # Mark as "no side" when paths are together
    
    # Detect region boundaries: where sign changes OR paths come together
    # A region is a contiguous stretch where sign is non-zero and constant
    
    regions = []
    in_region = False
    region_start = 0
    current_sign = 0
    
    for i in range(n_samples):
        if not in_region:
            # Start new region if we diverge
            if not is_together[i]:
                in_region = True
                region_start = i
                current_sign = sign[i]
        else:
            # End region if:
            # 1. Paths come back together
            # 2. Sign changes (paths cross)
            sign_changed = (sign[i] != 0 and sign[i] != current_sign)
            
            if is_together[i] or sign_changed:
                # Save the region
                regions.append((region_start, i))
                
                # If sign changed, start a new region immediately
                if sign_changed and not is_together[i]:
                    region_start = i
                    current_sign = sign[i]
                    in_region = True
                else:
                    in_region = False
    
    # Handle region that extends to the end
    if in_region:
        regions.append((region_start, n_samples - 1))
    
    # Build results for each region
    results = []
    
    for start_idx, end_idx in regions:
        if end_idx - start_idx < 5:  # Skip tiny regions
            continue
            
        orig_segment = orig_resampled[start_idx:end_idx+1]
        user_segment = user_resampled[start_idx:end_idx+1]
        
        if len(orig_segment) < 2:
            continue
        
        # Create polygon from the two path segments
        polygon_pts = np.vstack([orig_segment, user_segment[::-1]])
        
        # Create mask for this region
        cluster_mask = np.zeros((height, width), dtype=np.float32)
        
        rr, cc = polygon(polygon_pts[:, 1], polygon_pts[:, 0], shape=(height, width))
        if len(rr) == 0:
            continue
        cluster_mask[rr, cc] = 1.0
        
        # Skip if area too small
        if cluster_mask.sum() < min_area:
            continue
        
        # Get edit points
        ys, xs = np.where(cluster_mask > 0)
        cluster_points = np.column_stack([xs, ys])
        
        # Compute class contributions
        contributions = {}
        for cls in obstacle_classes:
            contributions[cls] = (prob_dict[cls] * cluster_mask).sum() / (cluster_mask.sum() + 1e-8)
        
        total = sum(contributions.values()) + 1e-8
        contributions = {cls: v / total for cls, v in contributions.items()}
        
        results.append([contributions, cluster_points, cluster_mask])
    
    return results

def obtain_probabilities(obstacle_classes, positions, radii, height=128 , width = 128, temperature=5.0):
    
    # print(type(positions)) print(positions) print(type(radii)) print(radii)
    
    rows, cols = np.ogrid[:height, :width]

    pos_dict = positions [0] if isinstance(positions, list) else positions
    rad_dict = radii[0] if isinstance(radii, list) else radii

    distances = {}
    valid_classes = []

    for cls in obstacle_classes:
        cls_positions = pos_dict.get(cls, [])

        if len(cls_positions) == 0:
            distances[cls] = np.full((height, width), np.inf)
        else:
            min_dist = np.full((height, width), np.inf)
            for pos in cls_positions:
                r, c = pos[0], pos[1]
                dist = np.sqrt((rows - r)**2 + (cols-c)**2)
                min_dist = np.minimum(min_dist, dist)
            distances[cls] = min_dist
            valid_classes.append(cls)

    dist_stack = np.stack([distances[cls] for cls in obstacle_classes], axis=0)

    min_dist_all = np.min(dist_stack, axis=0, keepdims=True)
    min_dist_all = np.where(np.isinf(min_dist_all,), 0, min_dist_all)

    rel_dist = dist_stack - min_dist_all

    closeness_stack = np.exp(-rel_dist / temperature)

    # Zero out classes with no obstacles
    mask = np.array([1.0 if cls in valid_classes else 0.0 for cls in obstacle_classes])
    closeness_stack = closeness_stack * mask[:, None, None]
    
    # Normalize to get responsibilities (sum to 1 at each cell)
    sum_closeness = np.sum(closeness_stack, axis=0, keepdims=True) + 1e-8
    resp_stack = closeness_stack / sum_closeness
    
    # Convert to dict
    prob_dict = {cls: resp_stack[i] for i, cls in enumerate(obstacle_classes)}
    
    return prob_dict


def collate_ignore_metadata(batch):
    """

    Fixes stacking bug cus some of the features vary in length 

    """

    features = default_collate([item[0] for item in batch])
    targets = default_collate([item[1] for item in batch])
    goals = default_collate([item[4] for item in batch])
    
    angles = default_collate([item[5] for item in batch]) 
    positions = [item[2] for item in batch]
    radii = [item[3] for item in batch]

    return features, targets, positions, radii, goals, angles



def compute_path_from_costmap(costmap_dict, goal, device="cuda"):
    fused_map = fuse_costmaps(costmap_dict)
    
    if isinstance(fused_map, torch.Tensor):
        fused_map = fused_map.cpu().detach().numpy()
    
    # ✅ DEBUG: Print shapes and values
    print(f"Fused map shape: {fused_map.shape}")
    print(f"Goal value: {goal}")
    print(f"Goal type: {type(goal)}")
    
    # Check if fused_map is correct shape
    if len(fused_map.shape) == 4:  # (B, C, H, W)
        fused_map = fused_map[0, 0]  # Extract first batch/channel
        print(f"Extracted to shape: {fused_map.shape}")
    elif len(fused_map.shape) == 3:  # (C, H, W) or (B, H, W)
        fused_map = fused_map[0]
        print(f"Extracted to shape: {fused_map.shape}")


    fused_map.squeeze()

    H, W = fused_map.shape
    print(f"H={H}, W={W}")
    print(f"Start: [0, 0], Goal: ({goal[0]}, {goal[1]})")
    
    # ✅ Ensure goal is within bounds
    goal_y = int(np.clip(goal[0], 0, H - 1))
    goal_x = int(np.clip(goal[1], 0, W - 1))
    
    print(f"Clipped goal: ({goal_y}, {goal_x})")
    
    path = route_through_array(fused_map, [0, 0], [goal_y, goal_x], fully_connected=True, geometric=True)
    
    if path[0] is None:
        raise ValueError("No valid path found through costmap")
    
    path_array = np.array(path[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]
    
    return np.column_stack([x_np, y_np])

def path_length(P: np.ndarray) -> float:
    d = np.diff(P, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))

def curvature_penalty(P: np.ndarray) -> float:
    """
    Stable smoothness/curvature proxy using second differences.
    Lower is smoother.
    """
    if len(P) < 3:
        return 0.0
    d2 = P[2:] - 2*P[1:-1] + P[:-2]              # (N-2,2)
    return float(np.sum(np.einsum("ij,ij->i", d2, d2)))  # sum ||d2||^2

def length_ratio_penalty(P: np.ndarray, U: np.ndarray, eps: float = 1e-9) -> float:
    """
    Symmetric, well-conditioned penalty on length mismatch:
      (log(Lp/Lu))^2
    """
    Lp = path_length(P)
    Lu = path_length(U)
    return float(np.log((Lp + eps) / (Lu + eps))**2)

def trajectory_cost(orig_path: np.ndarray,
                    user_path: np.ndarray,
                    wH: float = 1,
                    wF: float = 0.3,
                    wK: float = 0.01,
                    wL: float = 0.0,
                    scales: dict | None = None) -> dict:
    """
    Returns a dict with components + total cost.

    scales (optional): {"H": sH, "F": sF, "K": sK} to normalize magnitudes.
    Example: scales={"H": 1.0, "F": 1.0, "K": 100.0}
    """
    # assumes you already defined:
    # hausdorff_distance(A,B) and discrete_frechet_distance(A,B)
    H = hausdorff_distance(orig_path, user_path)
    #F = discrete_frechet_distance(orig_path, user_path)
    F = 0
    K = curvature_penalty(orig_path)
    L = length_ratio_penalty(orig_path, user_path)
    
    

    if scales is None:
        sH = sF = 1.0
        sK = 1.0
    else:
        sH = float(scales.get("H", 1.0))
        sF = float(scales.get("F", 1.0))
        sK = float(scales.get("K", 1.0))

    total = wH * (H / sH) + wF * (F / sF) + wK * (K / sK) + wL * L

    return total

def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns NxM matrix of Euclidean distances between points in A and B.
    A: (N,2), B: (M,2)
    """
    # Broadcasting: (N,1,2) - (1,M,2) -> (N,M,2) -> norm -> (N,M)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Symmetric (undirected) discrete Hausdorff distance between two point sequences.
    """
    D = pairwise_dist(A, B)
    # directed: max_i min_j d(a_i, b_j)
    h_AB = np.max(np.min(D, axis=1))
    h_BA = np.max(np.min(D, axis=0))
    return float(max(h_AB, h_BA))

def discrete_frechet_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Discrete Fréchet distance between two point sequences (Eiter & Mannila DP).
    Iterative version to avoid recursion depth issues.
    """
    D = pairwise_dist(A, B)  # (N, M)
    N, M = D.shape
    
    # ✅ Use iterative DP instead of recursion
    ca = np.full((N, M), np.inf, dtype=float)
    
    # Base case
    ca[0, 0] = D[0, 0]
    
    # Fill first column
    for i in range(1, N):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
    
    # Fill first row
    for j in range(1, M):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    
    # Fill rest of the table
    for i in range(1, N):
        for j in range(1, M):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                D[i, j]
            )
    
    return float(ca[N - 1, M - 1])



def get_user_adjustments(fused_costmap, obstacle_positions, radii, goal_position, orientations):

    map_np = fused_costmap[0,0].detach().cpu().numpy()

    max_val = np.max(map_np)
    min_val = np.min(map_np)

    map_np = ((map_np-min_val)/(max_val - min_val) +1e-8)
    
    fig, ax = plt.subplots()
    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    #Tech debt but obstacle positions is a list of dictionaries
    for cls, obs_list in obstacle_positions[0].items():
        for i, pos in enumerate(obs_list):
            ax.plot(pos[1],pos[0], color=colors[cls], marker='o', label=f'{cls}')

            angle = orientations[cls][i].item()  # assuming orientations passed in same structure
            arrow_len = 5
            dx = arrow_len * np.cos(angle)
            dy = arrow_len * np.sin(angle)
            ax.arrow(pos[1], pos[0], dx, dy, 
                     head_width=1.5, head_length=1.0, 
                     fc=colors[cls], ec=colors[cls], alpha=0.8)
    
    print(goal_position)
    goal = goal_position[0]

    ax.plot(goal[1], goal[0], color='olive', marker='*', label='goal')

    #Get path
    path = route_through_array(map_np, [0,0], (goal[0],goal[1]), fully_connected=True, geometric=True)
    path_array = np.array(path[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]

    k = 3
    num_ctrl_pts = 10
    x_s, y_s, P, U= generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)

    #Plot B-Spline and control points
    ax.plot(x_s, y_s, linewidth=2, label="Original B-spline curve")
    ax.plot(P[:, 0], P[:, 1], "o--", alpha=0.7, label="original control polygon")
    
    dragger = DraggableBSpline(ax, U, P, k, n_samples=800)

    ax.imshow(map_np, origin='lower')
    ax.legend(loc='best')
    plt.show()
    print("finished")
    
    orig_path = np.column_stack([x_s, y_s])
    user_path = np.column_stack([dragger.x, dragger.y])

    return orig_path, user_path, 

def visualize_3d(fused_cm):
    map_np = fused_cm[0,0].detach().cpu().numpy()


    H, W = map_np.shape
    x=np.arange(0,W)
    y=np.arange(0,H)
    X, Y = np.meshgrid(x,y)
    Z = map_np

    # 2. Create the 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)

    # 4. Customization
    ax.set_title("3D Fused Costmap")
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Cost Intensity')

    fig.colorbar(surf, shrink=0.5, aspect=5, label='Cost')

    ax.view_init(elev=60, azim=35)

    plt.show()

def fuse_costmaps(diffused_cm):


    """
        IMPORTANT, this only works for 1 CM see line for diffused_stacked
    """
    # stack all the tensor vals

    
    #TODO: fix for multiple CM
    diffused_stacked = torch.stack([v[0].squeeze(1) for v in diffused_cm.values()], dim=1)

    combined_map = torch.logsumexp(diffused_stacked, dim=1)# - np.exp(-1)
        
    #Restore to usual tensor
    return combined_map.unsqueeze(1)

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



