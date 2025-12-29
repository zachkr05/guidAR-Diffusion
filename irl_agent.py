"""
IRL Agent for Learning Minimal Costmap Modifications

Given a user's trajectory edit, this agent:
1. Identifies which obstacle class was likely affected
2. Finds the minimal costmap modification that explains the trajectory
3. Updates only that class's LoRA adapter

Key insight: We want the SMALLEST change to the costmap that makes
the user's trajectory optimal (or near-optimal) under A*.

Loss = trajectory_match(A*(C + ΔC), τ_user) + λ_sparse * ||ΔC||₁
"""

import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES


# ==============================================================================
# A* Planner (Differentiable-friendly)
# ==============================================================================

def astar(costmap: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* pathfinding on a 2D costmap.
    
    Args:
        costmap: [H, W] cost values (higher = more expensive)
        start: (row, col) start position
        goal: (row, col) goal position
    
    Returns:
        List of (row, col) waypoints from start to goal
    """
    H, W = costmap.shape
    
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    # Priority queue: (f_score, counter, row, col)
    counter = 0
    open_set = [(0, counter, start[0], start[1])]
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, _, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            
            neighbor = (nr, nc)
            
            # Movement cost: diagonal costs sqrt(2), cardinal costs 1
            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            
            # Total cost = movement + costmap value at destination
            tentative_g = g_score[current] + move_cost * (1 + costmap[nr, nc])
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                counter += 1
                heapq.heappush(open_set, (f, counter, nr, nc))
    
    # No path found
    return [start, goal]


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


# ==============================================================================
# Trajectory Utilities
# ==============================================================================

def trajectory_to_tensor(trajectory: List[Tuple[int, int]], H: int, W: int) -> torch.Tensor:
    """Convert trajectory to normalized tensor [N, 2]."""
    traj = torch.tensor(trajectory, dtype=torch.float32)
    traj[:, 0] /= H  # Normalize row
    traj[:, 1] /= W  # Normalize col
    return traj


def resample_trajectory(trajectory: List[Tuple[int, int]], n_points: int) -> List[Tuple[float, float]]:
    """Resample trajectory to fixed number of points."""
    if len(trajectory) < 2:
        return trajectory * n_points
    
    # Compute cumulative distances
    traj = np.array(trajectory, dtype=np.float32)
    dists = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    if total_dist < 1e-6:
        return [trajectory[0]] * n_points
    
    # Sample at uniform distances
    sample_dists = np.linspace(0, total_dist, n_points)
    resampled = []
    
    for d in sample_dists:
        idx = np.searchsorted(cum_dists, d, side='right') - 1
        idx = max(0, min(idx, len(trajectory) - 2))
        
        # Linear interpolation
        t = (d - cum_dists[idx]) / (cum_dists[idx + 1] - cum_dists[idx] + 1e-8)
        t = max(0, min(1, t))
        
        r = trajectory[idx][0] * (1 - t) + trajectory[idx + 1][0] * t
        c = trajectory[idx][1] * (1 - t) + trajectory[idx + 1][1] * t
        resampled.append((r, c))
    
    return resampled


def frechet_distance(traj1: torch.Tensor, traj2: torch.Tensor) -> torch.Tensor:
    """
    Approximate Frechet distance between two trajectories.
    Both trajectories should be [N, 2] tensors.
    """
    # Simple approximation: mean of pointwise distances after resampling
    dists = torch.sqrt(torch.sum((traj1 - traj2)**2, dim=1))
    return torch.mean(dists)


def trajectory_cost(costmap: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute total cost along a trajectory (differentiable).
    
    Args:
        costmap: [H, W] tensor
        trajectory: [N, 2] tensor with normalized coordinates [0, 1]
    
    Returns:
        Scalar cost
    """
    H, W = costmap.shape
    
    # Convert to pixel coordinates
    coords = trajectory.clone()
    coords[:, 0] = coords[:, 0] * (H - 1)
    coords[:, 1] = coords[:, 1] * (W - 1)
    
    # Bilinear interpolation for differentiability
    coords = coords.clamp(0, H - 1.001)
    
    r = coords[:, 0]
    c = coords[:, 1]
    
    r0 = r.floor().long()
    r1 = (r0 + 1).clamp(max=H-1)
    c0 = c.floor().long()
    c1 = (c0 + 1).clamp(max=W-1)
    
    fr = r - r0.float()
    fc = c - c0.float()
    
    # Bilinear interpolation
    v00 = costmap[r0, c0]
    v01 = costmap[r0, c1]
    v10 = costmap[r1, c0]
    v11 = costmap[r1, c1]
    
    v0 = v00 * (1 - fc) + v01 * fc
    v1 = v10 * (1 - fc) + v11 * fc
    values = v0 * (1 - fr) + v1 * fr
    
    return torch.sum(values)


# ==============================================================================
# Class Identification
# ==============================================================================

def identify_affected_class(
    user_trajectory: List[Tuple[int, int]],
    original_trajectory: List[Tuple[int, int]],
    obstacle_positions: Dict[int, List[Tuple[int, int]]],
    H: int, W: int
) -> Tuple[int, float]:
    """
    Identify which obstacle class the user is avoiding more/less.
    
    Compares how close the user's trajectory gets to each class vs
    how close the original trajectory got.
    
    Returns:
        (class_id, confidence) - the class most likely affected
    """
    user_traj = np.array(user_trajectory)
    orig_traj = np.array(original_trajectory)
    
    class_scores = {}
    
    for class_id, obstacles in obstacle_positions.items():
        if len(obstacles) == 0:
            class_scores[class_id] = 0.0
            continue
        
        obs = np.array(obstacles)
        
        # Min distance from user trajectory to any obstacle of this class
        user_min_dists = []
        for point in user_traj:
            dists = np.sqrt(np.sum((obs - point)**2, axis=1))
            user_min_dists.append(np.min(dists))
        user_avg_min_dist = np.mean(user_min_dists)
        
        # Min distance from original trajectory
        orig_min_dists = []
        for point in orig_traj:
            dists = np.sqrt(np.sum((obs - point)**2, axis=1))
            orig_min_dists.append(np.min(dists))
        orig_avg_min_dist = np.mean(orig_min_dists)
        
        # Positive = user is farther from this class (wants MORE avoidance)
        # Negative = user is closer (wants LESS avoidance)
        class_scores[class_id] = user_avg_min_dist - orig_avg_min_dist
    
    # Find class with largest absolute change
    best_class = max(class_scores.keys(), key=lambda k: abs(class_scores[k]))
    confidence = abs(class_scores[best_class])
    
    return best_class, confidence


# ==============================================================================
# Costmap Modification Network
# ==============================================================================

class CostmapModifier(nn.Module):
    """
    Learns to predict minimal costmap modifications.
    
    Given:
        - Base costmap
        - User trajectory
        - Class activations
    
    Outputs:
        - ΔC: sparse modification to costmap
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Encode base costmap
        self.costmap_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # Encode trajectory as a "heatmap"
        self.traj_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # Encode class activations
        self.class_encoder = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # Combine and predict delta
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1], will be scaled
        )
        
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, costmap, traj_heatmap, class_activations):
        """
        Args:
            costmap: [B, 1, H, W]
            traj_heatmap: [B, 1, H, W] - heatmap of user trajectory
            class_activations: [B, NUM_CLASSES, H, W]
        
        Returns:
            delta_costmap: [B, 1, H, W]
        """
        c_feat = self.costmap_encoder(costmap)
        t_feat = self.traj_encoder(traj_heatmap)
        cls_feat = self.class_encoder(class_activations)
        
        combined = torch.cat([c_feat, t_feat, cls_feat], dim=1)
        delta = self.decoder(combined)
        
        return delta * self.delta_scale


def trajectory_to_heatmap(trajectory: List[Tuple[int, int]], H: int, W: int, sigma: float = 2.0) -> np.ndarray:
    """Convert trajectory to a soft heatmap."""
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    rows, cols = np.ogrid[:H, :W]
    
    for (r, c) in trajectory:
        dist_sq = (rows - r)**2 + (cols - c)**2
        heatmap = np.maximum(heatmap, np.exp(-dist_sq / (2 * sigma**2)))
    
    return heatmap


# ==============================================================================
# IRL Agent
# ==============================================================================

class IRLAgent:
    """
    Inverse Reinforcement Learning agent that learns minimal costmap modifications.
    
    Pipeline:
    1. User provides edited trajectory
    2. Agent identifies which class was affected
    3. Agent finds minimal ΔC to explain trajectory
    4. Agent updates that class's LoRA in the diffusion model
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        lambda_sparse: float = 0.01,
        lambda_smooth: float = 0.001,
        n_optim_steps: int = 100
    ):
        self.diffusion_model = diffusion_model
        self.device = device
        self.lr = lr
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.n_optim_steps = n_optim_steps
        
        # Costmap modifier network
        self.modifier = CostmapModifier().to(device)
        self.modifier_optimizer = torch.optim.Adam(self.modifier.parameters(), lr=lr)
    
    def find_minimal_modification(
        self,
        base_costmap: np.ndarray,
        user_trajectory: List[Tuple[int, int]],
        class_activations: np.ndarray,
        goal: Tuple[int, int],
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Find minimal costmap modification to explain user trajectory.
        
        Args:
            base_costmap: [H, W] current costmap
            user_trajectory: List of (row, col) waypoints from user
            class_activations: [NUM_CLASSES, H, W] obstacle locations
            goal: (row, col) goal position
        
        Returns:
            delta_costmap: [H, W] modification
            info: dict with optimization info
        """
        H, W = base_costmap.shape
        start = user_trajectory[0]
        
        # Convert to tensors
        base_costmap_t = torch.from_numpy(base_costmap).float().to(self.device)
        class_act_t = torch.from_numpy(class_activations).float().to(self.device)
        
        # Create trajectory heatmap
        traj_heatmap = trajectory_to_heatmap(user_trajectory, H, W)
        traj_heatmap_t = torch.from_numpy(traj_heatmap).float().to(self.device)
        
        # Resample user trajectory
        user_traj_resampled = resample_trajectory(user_trajectory, n_points=50)
        user_traj_t = torch.tensor(user_traj_resampled, dtype=torch.float32, device=self.device)
        user_traj_t[:, 0] /= H
        user_traj_t[:, 1] /= W
        
        # Learnable delta costmap (direct optimization)
        delta = torch.zeros(H, W, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        losses = []
        
        for step in range(self.n_optim_steps):
            optimizer.zero_grad()
            
            # Modified costmap
            modified_costmap = base_costmap_t + delta
            modified_costmap = torch.clamp(modified_costmap, 0, None)  # Non-negative costs
            
            # Plan on modified costmap (non-differentiable, but we use cost along user traj)
            # Key insight: if user trajectory is optimal, its cost should be minimal
            
            # Cost of user trajectory on modified costmap
            user_cost = trajectory_cost(modified_costmap, user_traj_t)
            
            # Generate alternative paths by perturbing and check they're more expensive
            # (This encourages user path to be optimal)
            perturbation_loss = 0.0
            n_perturbations = 5
            
            for _ in range(n_perturbations):
                # Perturb trajectory
                noise = torch.randn_like(user_traj_t) * 0.05
                perturbed_traj = user_traj_t + noise
                perturbed_traj = torch.clamp(perturbed_traj, 0, 1)
                
                perturbed_cost = trajectory_cost(modified_costmap, perturbed_traj)
                
                # User trajectory should be cheaper than perturbations
                perturbation_loss += F.relu(user_cost - perturbed_cost + 0.1)
            
            perturbation_loss /= n_perturbations
            
            # Sparsity loss (L1)
            sparsity_loss = torch.mean(torch.abs(delta))
            
            # Smoothness loss (total variation)
            dx = delta[1:, :] - delta[:-1, :]
            dy = delta[:, 1:] - delta[:, :-1]
            smoothness_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
            
            # Focus loss: delta should be near obstacles
            class_mask = torch.max(class_act_t, dim=0)[0]  # [H, W]
            focus_loss = torch.mean(torch.abs(delta) * (1 - class_mask))  # Penalize delta away from obstacles
            
            # Total loss
            loss = (
                user_cost * 0.1 +  # Want low cost for user trajectory
                perturbation_loss +  # Want user trajectory to be optimal
                self.lambda_sparse * sparsity_loss +
                self.lambda_smooth * smoothness_loss +
                0.01 * focus_loss
            )
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and step % 20 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, "
                      f"user_cost={user_cost.item():.4f}, "
                      f"sparsity={sparsity_loss.item():.4f}")
        
        delta_np = delta.detach().cpu().numpy()
        
        info = {
            'losses': losses,
            'final_loss': losses[-1],
            'delta_magnitude': np.abs(delta_np).mean(),
            'delta_max': np.abs(delta_np).max(),
        }
        
        return delta_np, info
    
    def learn_from_trajectory_edit(
        self,
        base_costmap: np.ndarray,
        user_trajectory: List[Tuple[int, int]],
        original_trajectory: List[Tuple[int, int]],
        obstacle_positions: Dict[int, List[Tuple[int, int]]],
        class_activations: np.ndarray,
        goal: Tuple[int, int],
        n_finetune_steps: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Full pipeline: identify affected class, find modification, update LoRA.
        
        Args:
            base_costmap: [H, W] current costmap from diffusion model
            user_trajectory: User's edited trajectory
            original_trajectory: Original planned trajectory
            obstacle_positions: {class_id: [(r, c), ...]}
            class_activations: [NUM_CLASSES, H, W]
            goal: Goal position
            n_finetune_steps: Steps to fine-tune LoRA
        
        Returns:
            info dict with results
        """
        H, W = base_costmap.shape
        
        # Step 1: Identify affected class
        if verbose:
            print("Step 1: Identifying affected class...")
        
        affected_class, confidence = identify_affected_class(
            user_trajectory, original_trajectory, obstacle_positions, H, W
        )
        class_name = OBSTACLE_CLASSES[affected_class]['name']
        
        if verbose:
            print(f"  Affected class: {affected_class} ({class_name}), confidence: {confidence:.3f}")
        
        # Step 2: Find minimal costmap modification
        if verbose:
            print("\nStep 2: Finding minimal costmap modification...")
        
        delta_costmap, mod_info = self.find_minimal_modification(
            base_costmap, user_trajectory, class_activations, goal, verbose
        )
        
        if verbose:
            print(f"  Delta magnitude: {mod_info['delta_magnitude']:.4f}")
            print(f"  Delta max: {mod_info['delta_max']:.4f}")
        
        # Step 3: Create target costmap
        target_costmap = base_costmap + delta_costmap
        target_costmap = np.clip(target_costmap, 0, None)
        
        # Step 4: Fine-tune only the affected class's LoRA
        if verbose:
            print(f"\nStep 3: Fine-tuning {class_name} LoRA...")
        
        finetune_info = self.finetune_class_lora(
            affected_class, 
            target_costmap,
            class_activations,
            goal,
            n_steps=n_finetune_steps,
            verbose=verbose
        )
        
        return {
            'affected_class': affected_class,
            'class_name': class_name,
            'confidence': confidence,
            'delta_costmap': delta_costmap,
            'target_costmap': target_costmap,
            'modification_info': mod_info,
            'finetune_info': finetune_info
        }
    
    def finetune_class_lora(
        self,
        class_id: int,
        target_costmap: np.ndarray,
        class_activations: np.ndarray,
        goal: Tuple[int, int],
        n_steps: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Fine-tune only one class's LoRA to produce the target costmap.
        """
        from diffusion_utils import schedule_betas, q_sample
        
        H, W = target_costmap.shape
        device = self.device
        
        # Freeze all except target class
        self.diffusion_model.freeze_all_lora_except_class(class_id)
        
        # Get trainable parameters
        params = self.diffusion_model.get_class_lora_parameters(class_id)
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        
        # Prepare target
        target = torch.from_numpy(target_costmap).float().to(device)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        target = target * 2 - 1  # Normalize to [-1, 1]
        target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Prepare conditioning
        from DataGenerator.dataset_costmap import make_goal_map
        goal_map = make_goal_map(H, W, goal)
        cond = np.concatenate([class_activations, goal_map[None, :, :]], axis=0)
        cond = torch.from_numpy(cond).float().unsqueeze(0).to(device)  # [1, NUM_CLASSES+1, H, W]
        
        # Diffusion schedule
        T = 1000
        betas, alphas, alpha_bar = schedule_betas(T, 1e-4, 0.02, device=device)
        
        losses = []
        
        self.diffusion_model.train()
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Sample random timestep
            t = torch.randint(0, T, (1,), device=device, dtype=torch.long)
            
            # Add noise to target
            xt, noise = q_sample(target, t, alpha_bar)
            
            # Model input
            x_in = torch.cat([cond, xt], dim=1)
            
            # Predict noise
            pred_noise = self.diffusion_model(x_in, t)
            
            # Loss
            loss = F.mse_loss(pred_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and step % 10 == 0:
                print(f"    Step {step}: loss={loss.item():.4f}")
        
        self.diffusion_model.eval()
        
        return {
            'losses': losses,
            'final_loss': losses[-1] if losses else 0.0
        }


# ==============================================================================
# Convenience functions
# ==============================================================================

def create_irl_agent(diffusion_model, device="cuda", **kwargs) -> IRLAgent:
    """Create an IRL agent with default settings."""
    return IRLAgent(diffusion_model, device=device, **kwargs)


# ==============================================================================
# Example usage
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from UNet.UNet_class_lora import create_class_lora_unet
    from DataGenerator.sim import Costmap
    from DataGenerator.dataset_costmap import make_class_occupancy_maps, make_goal_map
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = create_class_lora_unet(lora_rank=8).to(device)
    
    # Create IRL agent
    agent = IRLAgent(model, device=device, n_optim_steps=50)
    
    # Create test scenario
    H, W = 64, 64
    goal = (10, 55)
    
    obstacles_by_class = {
        0: [(30, 25), (35, 30)],   # chairs
        1: [(40, 45)],              # table
        2: [(25, 40)],              # person
        3: [],                      # no walls
    }
    
    # Generate base costmap
    cm = Costmap(H, W)
    cm.goal = np.array(goal, dtype=np.float32)
    base_costmap = cm.calculateCostMapMulticlassVectorized(obstacles_by_class)
    
    # Normalize
    base_costmap = (base_costmap - base_costmap.min()) / (base_costmap.max() - base_costmap.min() + 1e-8)
    
    # Original trajectory (A* on base costmap)
    start = (55, 5)
    original_traj = astar(base_costmap, start, goal)
    print(f"Original trajectory: {len(original_traj)} waypoints")
    
    # Simulated user edit: user goes FARTHER from chairs
    user_traj = [(55, 5), (50, 10), (45, 15), (40, 20), (35, 35), 
                 (30, 45), (25, 50), (20, 53), (15, 55), (10, 55)]
    
    # Class activations
    class_activations = make_class_occupancy_maps(H, W, obstacles_by_class)
    
    # Run IRL
    print("\n" + "="*60)
    print("Running IRL Agent")
    print("="*60)
    
    result = agent.learn_from_trajectory_edit(
        base_costmap=base_costmap,
        user_trajectory=user_traj,
        original_trajectory=original_traj,
        obstacle_positions=obstacles_by_class,
        class_activations=class_activations,
        goal=goal,
        n_finetune_steps=30,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Affected class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Base costmap
    axes[0].imshow(base_costmap, cmap='viridis')
    orig_traj_arr = np.array(original_traj)
    axes[0].plot(orig_traj_arr[:, 1], orig_traj_arr[:, 0], 'w-', linewidth=2, label='Original')
    user_traj_arr = np.array(user_traj)
    axes[0].plot(user_traj_arr[:, 1], user_traj_arr[:, 0], 'r--', linewidth=2, label='User')
    axes[0].scatter([goal[1]], [goal[0]], c='green', s=100, marker='*')
    axes[0].legend()
    axes[0].set_title('Base Costmap + Trajectories')
    
    # Delta
    axes[1].imshow(result['delta_costmap'], cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1].set_title(f'Delta (affected: {result["class_name"]})')
    
    # Target costmap
    axes[2].imshow(result['target_costmap'], cmap='viridis')
    axes[2].set_title('Target Costmap')
    
    # Class activations
    class_vis = np.zeros((H, W, 3))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]  # R, G, B, Y
    for i in range(NUM_CLASSES):
        for c in range(3):
            class_vis[:, :, c] += class_activations[i] * colors[i][c]
    class_vis = np.clip(class_vis, 0, 1)
    axes[3].imshow(class_vis)
    axes[3].set_title('Class Activations (R=chair, G=table, B=person, Y=wall)')
    
    plt.tight_layout()
    plt.savefig('irl_agent_demo.png', dpi=150)
    print("\nSaved visualization to: irl_agent_demo.png")
    plt.show()
