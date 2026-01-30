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
from spline import *


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
    
    path = route_through_array(fused_map, [0, 0], [goal_y, goal_x], fully_connected=True)
    
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



def get_user_adjustments(fused_costmap, obstacle_positions, radii, goal_position):

    map_np = fused_costmap[0,0].detach().cpu().numpy()


    max_val = np.max(map_np)
    min_val = np.min(map_np)

    map_np = ((map_np-min_val)/(max_val - min_val) +1e-8)
    #map_np[map_np > 0.8] = np.inf
    #https://stackoverflow.com/questions/32551536/draw-marker-in-image
    #plt.annotate('25, 50', xy=(25, 50))
    #print(obstacle_positions)
   
    fig, ax = plt.subplots()
    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    #Tech debt but obstacle positions is a list of dictionaries
    for cls, obs_list in obstacle_positions[0].items():
        for pos in obs_list:
            ax.plot(pos[1],pos[0], color=colors[cls], marker='o', label=f'{cls}')
    goal = goal_position[0]

    ax.plot(goal[1], goal[0], color='olive', marker='*', label='goal')

    #Get path
    path = route_through_array(map_np, [0,0], (goal[0],goal[1]), fully_connected=True)
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
    #plt.colorbar()
    ax.legend(loc='best')
    plt.show()
    print("finished")
    
    orig_path = np.column_stack([x_np, y_np])
    user_path = np.column_stack([dragger.x, dragger.y])

    return orig_path, user_path

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



