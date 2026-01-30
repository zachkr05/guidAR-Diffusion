import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import *
# Import local modules
from DataGenerator.dataGenerator import CostmapDataset
from MoE.ddpm import DDPM
from train import ExpertEnsemble
from torch.utils.data.dataloader import default_collate

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from DataGenerator.dataGenerator import CostmapDataset

from MoE.UNet import LightweightUNet
from MoE.ddpm import DDPM




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


def finetune_models(model, batch, user_path, device,lr, target_class,wH,wF,wK,wL,epochs):
   

    features, targets, positions, radii, goal = batch
    expert_model = model.experts[target_class]
    expert_model.set_finetune(active=True)

    optimizer = AdamW(
            [p for p in expert_model.parameters() if p.requires_grad],
            lr = lr
            )

    ddpm = DDPM(timesteps=1000, device = device)

    loss_history = []


    avg_cost = None
    alpha_baseline = 0.9

    for epoch in tqdm(range(epochs), desc = "IRL Finetuning"):

        model.train()
        optimizer.zero_grad()

        conditioning = features[target_class].to(device)
        x_0 = targets[target_class].to(device)
        costmap_dict = {}



        generated, log_prob = ddpm.sample_with_partial_logprob(
                    expert_model, 
                    conditioning, 
                    shape=x_0.shape,
                    logprob_steps=50
                )
        for cls in model.obstacle_classes:
            if cls == target_class:
                costmap_dict[cls] = [generated]
            else:
                costmap_dict[cls] = [targets[cls].to(device)]

        try:
            generated_path = compute_path_from_costmap(costmap_dict, goal[0].cpu().numpy(), device)

            k = min(3, len(generated_path) -1)
            num_ctrl_pts = min(10, len(generated_path))

            if len(generated_path) >= 4:
                x_s, y_s, P, U = generate_clamped_spline(
                    generated_path[:, 0], 
                    generated_path[:, 1], 
                    k, 
                    num_ctrl_pts
                )
                smooth_generated_path = np.column_stack([x_s, y_s])
            else:
                smooth_generated_path = generated_path

            traj_cost = trajectory_cost(
                smooth_generated_path,
                user_path,
                wH=wH,
                wF = wF,
                wK = wK,
                wL = wL
                )

            if avg_cost is None:
                avg_cost = traj_cost
            else:
                avg_cost = alpha_baseline * avg_cost + (1-alpha_baseline)*traj_cost

            advantage = traj_cost - avg_cost


            cost_tensor = torch.tensor(advantage, device = device, dtype=torch.float32)

            reinforce_loss = (log_prob * cost_tensor).mean() 



            reinforce_loss.backward()
            optimizer.step()
            #cost_tensor.backward()
            #optimizer.step()

            loss_history.append(traj_cost)

            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs} - Trajectory Cost: {traj_cost:.4f}") 

        except (ValueError, RuntimeError) as e:
                    print(f"\nWarning: Could not compute path at epoch {epoch + 1}: {e}")
                    continue 
    
    expert_model.set_finetune(active=False)

    return loss_history

def collate_ignore_metadata(batch):
    """

    Fixes stacking bug cus some of the features vary in length 

    """

    features = default_collate([item[0] for item in batch])
    targets = default_collate([item[1] for item in batch])
    goals = default_collate([item[4] for item in batch])
    
    
    positions = [item[2] for item in batch]
    radii = [item[3] for item in batch]

    return features, targets, positions, radii, goals

def evaluate():
    
    #Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"
    output_dir = "eval_results"
    obstacle_classes = ["chair", "table", "bomb"]
    H, W = 128, 128
    batch_size = 1
    
    # Calculate channels: 2 (curr) + 2*(n-1) (others) + 1 (goal)
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes - 1) + 1

    # --- Load Data ---
    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)

    #Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch5.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #Setup reverse diffusion process
    ddpm = DDPM(timesteps=1000, device=device)
    mse_results = {cls: [] for cls in obstacle_classes}
    diffused_cm = {cls: [] for cls in obstacle_classes}

    first_batch = None

    print("Starting evaluation...")
    with torch.no_grad():
        first_batch= next(iter(loader))
        features, targets, positions, radii, goal = first_batch

        for cls in obstacle_classes:
            print(f"Sampling for expert: {cls}...")
            
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            
            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            loss = F.mse_loss(generated, gt)
            mse_results[cls].append(loss.item())
            diffused_cm[cls].append(generated)

    fused_costmap = fuse_costmaps(diffused_cm)
    #print(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal)
    #print("original path: ", orig_path)
    finetune_models(model=model, batch=first_batch, user_path=user_path, device=device, lr=1e-6, epochs=100, target_class="chair", wH=0.3, wF = 0, wK = 0.01, wL=0.0)

    #visualize_costmap()
    

if __name__ == "__main__":
    evaluate() 
