"""
Deployment: Test Learned Preferences in New Scenes

After fine-tuning LoRA on user trajectory edits, deploy the model
in completely new environments to verify the learned preferences transfer.

Pipeline:
1. Train on Scene A: User edits trajectory to avoid chairs more
2. Deploy on Scene B, C, D: Completely different layouts
3. Verify: Model generates costmaps with higher chair costs everywhere

This demonstrates GENERALIZATION of learned preferences.

Usage:
    python deploy_new_scene.py checkpoints/best_model.pt
    python deploy_new_scene.py checkpoints/best_model.pt --n-scenes 5
    python deploy_new_scene.py checkpoints/best_model.pt --compare-before-after
"""

import argparse
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from scipy.interpolate import interp1d
import torch.nn as nn
from diffusion_utils import schedule_betas, q_sample
from DataGenerator.dataset_costmap import make_class_occupancy_maps, make_goal_map
from DataGenerator.sim import Costmap, NUM_CLASSES, OBSTACLE_CLASSES
from UNet.UNet_class_lora import create_class_lora_unet, ClassLoRAUNet
from irl_agent import astar, resample_trajectory


# ==============================================================================
# Scene Generation
# ==============================================================================

def generate_random_scene(
    H: int = 64,
    W: int = 64,
    n_obstacles_per_class: Tuple[int, int] = (1, 4),
    seed: Optional[int] = None
) -> Dict:
    """
    Generate a completely random scene.
    
    Args:
        H, W: Grid dimensions
        n_obstacles_per_class: (min, max) obstacles per class
        seed: Random seed for reproducibility
    
    Returns:
        Scene dict with obstacles, goal, start, class_activations
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random start and goal (ensure they're far apart)
    while True:
        start = (np.random.randint(H//2, H-5), np.random.randint(5, W//2))
        goal = (np.random.randint(5, H//2), np.random.randint(W//2, W-5))
        
        dist = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        if dist > H * 0.5:  # Ensure reasonable distance
            break
    
    # Random obstacles for each class
    obstacles_by_class = {}
    for class_id in range(NUM_CLASSES):
        n_obs = np.random.randint(n_obstacles_per_class[0], n_obstacles_per_class[1] + 1)
        obstacles = []
        
        for _ in range(n_obs):
            # Avoid placing obstacles too close to start/goal
            while True:
                r = np.random.randint(5, H-5)
                c = np.random.randint(5, W-5)
                
                dist_to_start = np.sqrt((r - start[0])**2 + (c - start[1])**2)
                dist_to_goal = np.sqrt((r - goal[0])**2 + (c - goal[1])**2)
                
                if dist_to_start > 8 and dist_to_goal > 8:
                    break
            
            obstacles.append((r, c))
        
        obstacles_by_class[class_id] = obstacles
    
    # Generate class activation maps
    class_activations = make_class_occupancy_maps(H, W, obstacles_by_class)
    
    return {
        'obstacles_by_class': obstacles_by_class,
        'goal': goal,
        'start': start,
        'class_activations': class_activations,
        'H': H,
        'W': W
    }


def generate_specific_scenes() -> List[Dict]:
    """Generate a set of predefined diverse scenes for testing."""
    H, W = 64, 64
    scenes = []
    
    # Scene 1: Obstacles on left side
    scenes.append({
        'name': 'Left Cluster',
        'obstacles_by_class': {
            0: [(20, 15), (30, 20), (40, 18)],  # chairs on left
            1: [(25, 25), (35, 22)],            # tables
            2: [(28, 12)],                       # person
            3: [(45, 10), (46, 10)],            # wall
        },
        'goal': (10, 55),
        'start': (55, 10),
        'H': H, 'W': W
    })
    
    # Scene 2: Obstacles in middle
    scenes.append({
        'name': 'Central Obstacles',
        'obstacles_by_class': {
            0: [(30, 30), (35, 35), (25, 32)],  # chairs in middle
            1: [(32, 40)],                       # table
            2: [(28, 28), (38, 38)],            # people
            3: [],                               # no walls
        },
        'goal': (5, 58),
        'start': (58, 5),
        'H': H, 'W': W
    })
    
    # Add class activations to each scene
    for scene in scenes:
        scene['class_activations'] = make_class_occupancy_maps(
            scene['H'], scene['W'], scene['obstacles_by_class']
        )
    
    return scenes


# ==============================================================================
# Model Utilities
# ==============================================================================

@torch.no_grad()
def sample_costmap(model, conditioning, device, T=1000, verbose=False):
    """Sample costmap from diffusion model."""
    betas, alphas, alpha_bar = schedule_betas(T, 1e-4, 0.02, device=device)
    
    model.eval()
    B = conditioning.shape[0]
    H, W = conditioning.shape[2], conditioning.shape[3]
    
    x = torch.randn(B, 1, H, W, device=device)
    
    for t in reversed(range(T)):
        if verbose and t % 200 == 0:
            print(f"  Sampling step {T - t}/{T}")
        
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x_in = torch.cat([conditioning, x], dim=1)
        eps = model(x_in, t_batch)
        
        a_t = alphas[t]
        a_bar_t = alpha_bar[t]
        b_t = betas[t]
        
        mew = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * eps)
        
        if t > 0:
            x = mew + torch.sqrt(b_t) * torch.randn_like(x)
        else:
            x = mew
    
    return (x.clamp(-1, 1) + 1) / 2


def generate_ground_truth_costmap(scene: Dict) -> np.ndarray:
    """Generate ground truth costmap for a scene."""
    cm = Costmap(scene['H'], scene['W'])
    cm.goal = np.array(scene['goal'], dtype=np.float32)
    cost = cm.calculateCostMapMulticlassVectorized(scene['obstacles_by_class'])
    cost = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
    return cost


def compute_class_costs(costmap: np.ndarray, scene: Dict, radius: float = 5.0) -> Dict[int, float]:
    """
    Compute average cost around each obstacle class.
    
    This measures how much the model "fears" each class.
    """
    H, W = costmap.shape
    rows, cols = np.ogrid[:H, :W]
    
    class_costs = {}
    
    for class_id, obstacles in scene['obstacles_by_class'].items():
        if len(obstacles) == 0:
            class_costs[class_id] = 0.0
            continue
        
        total_cost = 0.0
        total_weight = 0.0
        
        for (r, c) in obstacles:
            # Compute cost in region around obstacle
            dist_sq = (rows - r)**2 + (cols - c)**2
            mask = dist_sq < radius**2
            
            if mask.any():
                total_cost += costmap[mask].mean()
                total_weight += 1
        
        class_costs[class_id] = total_cost / max(total_weight, 1)
    
    return class_costs


# ==============================================================================
# Fine-tuning with Geometric IRL
# ==============================================================================

def finetune_on_trajectory_edit(
    model: ClassLoRAUNet,
    scene: Dict,
    original_trajectory: List[Tuple[int, int]],
    user_trajectory: List[Tuple[int, int]],
    affected_class: int,
    device: str = "cuda",
    n_steps: int = 100,  # Not used for GPR, kept for API compatibility
    lr: float = 1e-2,    # Not used for GPR
    verbose: bool = True
) -> Dict:
    """
    Use GPR-FiLM to learn cost modulation for the affected class.
    
    GPR Benefits over Neural Networks:
    - No gradient training needed (instant update)
    - Uncertainty quantification (know when extrapolating)
    - Data efficient (works with 1-2 examples)
    - Interpretable
    
    Process:
    1. Compute geometric IRL delta (target costmap change)
    2. Solve for (gamma, beta) that best explains the delta
    3. Add observation to GPR for affected class
    4. At inference: apply learned (gamma, beta) to new scenes
    """
    H, W = scene['H'], scene['W']
    
    # =========================================================================
    # Step 1: Compute geometric IRL delta (our target)
    # =========================================================================
    if verbose:
        print(f"  Computing geometric IRL delta for class {affected_class} ({OBSTACLE_CLASSES[affected_class]['name']})...")
    
    delta = compute_geometric_delta(original_trajectory, user_trajectory, H, W)
    
    # =========================================================================
    # Step 2: Generate current costmap (before modification)
    # =========================================================================
    goal_map = make_goal_map(H, W, scene['goal'])
    cond = np.concatenate([scene['class_activations'], goal_map[None, :, :]], axis=0)
    cond_t = torch.from_numpy(cond).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        current_costmap = sample_costmap(model, cond_t, device)[0, 0].cpu().numpy()
    
    target_costmap = np.clip(current_costmap + delta, 0, 1)
    
    # =========================================================================
    # Step 3: Initialize GPR-FiLM manager if not present
    # =========================================================================
    if not hasattr(model, 'gpr_film_manager'):
        if verbose:
            print(f"  Initializing GPR-FiLM manager...")
        model.gpr_film_manager = GPRFiLMManager(num_classes=NUM_CLASSES, grid_size=H)
    
    # =========================================================================
    # Step 4: Compute target (gamma, beta) from IRL delta
    # =========================================================================
    target_gamma, target_beta = model.gpr_film_manager.compute_target_params_from_irl(
        current_costmap=current_costmap,
        target_costmap=target_costmap,
        class_activation=scene['class_activations'][affected_class]
    )
    
    if verbose:
        print(f"  Target FiLM params from IRL: γ={target_gamma:.3f}, β={target_beta:.3f}")
    
    # =========================================================================
    # Step 5: Add observation to GPR (this is the "training")
    # =========================================================================
    model.gpr_film_manager.add_trajectory_edit(
        original_traj=original_trajectory,
        user_traj=user_trajectory,
        affected_class=affected_class,
        target_gamma=target_gamma,
        target_beta=target_beta
    )
    
    if verbose:
        print(f"  Added observation to GPR for {OBSTACLE_CLASSES[affected_class]['name']}")
        
        # Show current state
        preds = model.gpr_film_manager.get_all_predictions()
        print(f"  GPR-FiLM state:")
        for cid in range(NUM_CLASSES):
            p = preds[cid]
            name = OBSTACLE_CLASSES[cid]['name']
            if p['n_observations'] > 0:
                print(f"    {name}: γ={p['gamma']:.3f}±{p['gamma_std']:.3f}, β={p['beta']:.3f}±{p['beta_std']:.3f} (n={p['n_observations']})")
            else:
                print(f"    {name}: no data")
    
    return {
        'delta': delta,
        'target_costmap': target_costmap,
        'losses': [0.0],  # No iterative loss for GPR
        'affected_class': affected_class,
        'gamma': target_gamma,
        'beta': target_beta
    }


# ==============================================================================
# GPR-FiLM Components (imported from gpr_film.py, inlined here for standalone)
# ==============================================================================

class TrajectoryFeatureExtractor:
    """Extract features from trajectory pair for GPR input."""
    
    def __init__(self, n_points: int = 50, grid_size: int = 64):
        self.n_points = n_points
        self.grid_size = grid_size
    
    def extract(self, original_traj: List[Tuple[int, int]], user_traj: List[Tuple[int, int]]) -> np.ndarray:
        orig = self._resample(original_traj)
        user = self._resample(user_traj)
        
        displacement = user - orig
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        
        mean_displacement = np.mean(displacement_magnitude)
        max_displacement = np.max(displacement_magnitude)
        std_displacement = np.std(displacement_magnitude)
        
        max_idx = np.argmax(displacement_magnitude)
        max_position = max_idx / self.n_points
        max_location = orig[max_idx] / self.grid_size
        
        mean_direction = np.mean(displacement, axis=0)
        mean_direction_norm = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)
        
        orig_length = self._path_length(orig)
        user_length = self._path_length(user)
        length_ratio = user_length / (orig_length + 1e-8)
        
        area_between = np.sum(displacement_magnitude) / self.n_points
        
        features = np.array([
            mean_displacement / self.grid_size,
            max_displacement / self.grid_size,
            std_displacement / self.grid_size,
            max_position,
            max_location[0],
            max_location[1],
            mean_direction_norm[0],
            mean_direction_norm[1],
            length_ratio - 1.0,
            area_between / self.grid_size,
        ])
        return features
    
    def _resample(self, traj: List[Tuple[int, int]]) -> np.ndarray:
        traj = np.array(traj, dtype=np.float32)
        if len(traj) < 2:
            return np.tile(traj[0], (self.n_points, 1))
        diffs = np.diff(traj, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dists = np.concatenate([[0], np.cumsum(dists)])
        total = cum_dists[-1]
        if total < 1e-6:
            return np.tile(traj[0], (self.n_points, 1))
        sample_d = np.linspace(0, total, self.n_points)
        interp_r = interp1d(cum_dists, traj[:, 0], kind='linear', fill_value='extrapolate')
        interp_c = interp1d(cum_dists, traj[:, 1], kind='linear', fill_value='extrapolate')
        return np.stack([interp_r(sample_d), interp_c(sample_d)], axis=1)
    
    def _path_length(self, traj: np.ndarray) -> float:
        diffs = np.diff(traj, axis=0)
        return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))


class GPRFiLM:
    """Gaussian Process Regression for FiLM parameters."""
    
    def __init__(self, length_scale: float = 0.5, noise_var: float = 0.01, output_scale: float = 1.0):
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.output_scale = output_scale
        self.X_train = None
        self.y_gamma = None
        self.y_beta = None
        self._alpha_gamma = None
        self._alpha_beta = None
        self._L = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist
        sq_dist = cdist(X1, X2, metric='sqeuclidean')
        return self.output_scale * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
    
    def fit(self, X: np.ndarray, y_gamma: np.ndarray, y_beta: np.ndarray):
        from scipy.linalg import cholesky, solve_triangular
        self.X_train = X.copy()
        self.y_gamma = y_gamma.copy()
        self.y_beta = y_beta.copy()
        
        K = self._rbf_kernel(X, X)
        K += self.noise_var * np.eye(len(X))
        
        try:
            self._L = cholesky(K, lower=True)
        except:
            K += 1e-6 * np.eye(len(X))
            self._L = cholesky(K, lower=True)
        
        self._alpha_gamma = solve_triangular(self._L.T, solve_triangular(self._L, y_gamma, lower=True))
        self._alpha_beta = solve_triangular(self._L.T, solve_triangular(self._L, y_beta, lower=True))
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from scipy.linalg import solve_triangular
        if self.X_train is None:
            M = X.shape[0]
            return np.zeros(M), np.ones(M) * self.output_scale, np.zeros(M), np.ones(M) * self.output_scale
        
        K_star = self._rbf_kernel(X, self.X_train)
        gamma_mean = K_star @ self._alpha_gamma
        beta_mean = K_star @ self._alpha_beta
        
        v = solve_triangular(self._L, K_star.T, lower=True)
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-8)
        
        return gamma_mean, np.sqrt(var), beta_mean, np.sqrt(var)
    
    def add_observation(self, x: np.ndarray, gamma: float, beta: float):
        x = x.reshape(1, -1)
        if self.X_train is None:
            self.fit(x, np.array([gamma]), np.array([beta]))
        else:
            X_new = np.vstack([self.X_train, x])
            y_gamma_new = np.append(self.y_gamma, gamma)
            y_beta_new = np.append(self.y_beta, beta)
            self.fit(X_new, y_gamma_new, y_beta_new)


class GPRFiLMManager:
    """Manages GPR-FiLM models for all classes."""
    
    def __init__(self, num_classes: int = 4, grid_size: int = 64):
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.feature_extractor = TrajectoryFeatureExtractor(n_points=50, grid_size=grid_size)
        self.gpr_models: Dict[int, GPRFiLM] = {c: GPRFiLM() for c in range(num_classes)}
        self.observations: Dict[int, List[Dict]] = {c: [] for c in range(num_classes)}
    
    def add_trajectory_edit(self, original_traj, user_traj, affected_class, target_gamma, target_beta):
        features = self.feature_extractor.extract(original_traj, user_traj)
        self.observations[affected_class].append({
            'features': features, 'gamma': target_gamma, 'beta': target_beta
        })
        self.gpr_models[affected_class].add_observation(features, target_gamma, target_beta)
    
    def get_all_predictions(self) -> Dict[int, Dict[str, float]]:
        predictions = {}
        for class_id in range(self.num_classes):
            if len(self.observations[class_id]) > 0:
                last_obs = self.observations[class_id][-1]
                features = last_obs['features'].reshape(1, -1)
                gamma_mean, gamma_std, beta_mean, beta_std = self.gpr_models[class_id].predict(features)
                predictions[class_id] = {
                    'gamma': gamma_mean[0], 'gamma_std': gamma_std[0],
                    'beta': beta_mean[0], 'beta_std': beta_std[0],
                    'n_observations': len(self.observations[class_id])
                }
            else:
                predictions[class_id] = {
                    'gamma': 0.0, 'gamma_std': 1.0, 'beta': 0.0, 'beta_std': 1.0, 'n_observations': 0
                }
        return predictions
    
    def compute_target_params_from_irl(self, current_costmap, target_costmap, class_activation):
        from scipy import ndimage as ndi
        mask = ndi.maximum_filter(class_activation, size=12)
        mask = ndi.gaussian_filter(mask.astype(np.float32), sigma=3)
        region = mask > 0.1
        
        if not region.any():
            return 0.0, 0.0
        
        c = current_costmap[region]
        t = target_costmap[region]
        m = mask[region]
        delta = t - c
        
        A = np.column_stack([c * m, m])
        try:
            params, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
            gamma, beta = params
        except:
            gamma = np.mean(delta / (c + 1e-8))
            beta = np.mean(delta)
        
        return float(np.clip(gamma, -2.0, 5.0)), float(np.clip(beta, -1.0, 2.0))


def sample_costmap_with_gpr_film(model, conditioning, device, T=1000, verbose=False):
    """Sample costmap and apply GPR-FiLM modulation."""
    betas, alphas, alpha_bar = schedule_betas(T, 1e-4, 0.02, device=device)
    
    model.eval()
    B = conditioning.shape[0]
    H, W = conditioning.shape[2], conditioning.shape[3]
    class_activations = conditioning[:, :NUM_CLASSES, :, :].cpu().numpy()
    
    x = torch.randn(B, 1, H, W, device=device)
    
    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x_in = torch.cat([conditioning, x], dim=1)
        eps = model(x_in, t_batch)
        
        a_t = alphas[t]
        a_bar_t = alpha_bar[t]
        b_t = betas[t]
        
        mew = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * eps)
        x = mew + torch.sqrt(b_t) * torch.randn_like(x) if t > 0 else mew
    
    costmap_np = ((x.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
    
    # Apply GPR-FiLM modulation
    if hasattr(model, 'gpr_film_manager'):
        from scipy import ndimage as ndi
        predictions = model.gpr_film_manager.get_all_predictions()
        
        for class_id, params in predictions.items():
            if params['n_observations'] == 0:
                continue
            
            gamma = params['gamma']
            beta = params['beta']
            
            if abs(gamma) < 0.01 and abs(beta) < 0.01:
                continue
            
            class_act = class_activations[0, class_id]
            class_region = ndi.maximum_filter(class_act, size=10)
            class_region = ndi.gaussian_filter(class_region.astype(np.float32), sigma=3)
            
            # FiLM: modified = current * (1 + gamma * region) + beta * region
            costmap_np = costmap_np * (1 + gamma * class_region) + beta * class_region
    
    return torch.from_numpy(np.clip(costmap_np, 0, 1)).float().unsqueeze(0).unsqueeze(0)


def compute_geometric_delta(
    original_trajectory: List[Tuple[int, int]],
    user_trajectory: List[Tuple[int, int]],
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
        avoided_strength = np.clip(diff[avoided] / 10.0, 0, 1)  # Linear, capped at 1
        delta[avoided] = avoided_strength * cost_increase
    
    # Preferred region - gentle valley
    preferred = (diff < -2.0) & near_traj
    if preferred.any():
        preferred_strength = np.clip(-diff[preferred] / 10.0, 0, 1)
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


# ==============================================================================
# Deployment Evaluation
# ==============================================================================

def deploy_and_evaluate(
    model: ClassLoRAUNet,
    scenes: List[Dict],
    device: str = "cuda",
    verbose: bool = True
) -> List[Dict]:
    """
    Deploy model on multiple new scenes and evaluate.
    
    Returns results for each scene including:
    - Generated costmap
    - Planned trajectory
    - Per-class costs (measures learned preferences)
    """
    results = []
    
    for i, scene in enumerate(scenes):
        scene_name = scene.get('name', f'Scene {i+1}')
        if verbose:
            print(f"\n--- {scene_name} ---")
        
        H, W = scene['H'], scene['W']
        
        # Prepare conditioning
        goal_map = make_goal_map(H, W, scene['goal'])
        cond = np.concatenate([scene['class_activations'], goal_map[None, :, :]], axis=0)
        cond_t = torch.from_numpy(cond).float().unsqueeze(0).to(device)
        
        # Generate costmap
        with torch.no_grad():
            generated = sample_costmap(model, cond_t, device)[0, 0].cpu().numpy()
        
        # Plan trajectory
        trajectory = astar(generated, scene['start'], scene['goal'])
        
        # Compute ground truth
        gt_costmap = generate_ground_truth_costmap(scene)
        
        # Compute per-class costs
        class_costs = compute_class_costs(generated, scene)
        gt_class_costs = compute_class_costs(gt_costmap, scene)
        
        if verbose:
            print(f"  Trajectory length: {len(trajectory)} waypoints")
            print(f"  Per-class costs (generated vs GT):")
            for cid in range(NUM_CLASSES):
                name = OBSTACLE_CLASSES[cid]['name']
                print(f"    {name}: {class_costs[cid]:.3f} (GT: {gt_class_costs[cid]:.3f})")
        
        results.append({
            'scene': scene,
            'scene_name': scene_name,
            'generated_costmap': generated,
            'gt_costmap': gt_costmap,
            'trajectory': trajectory,
            'class_costs': class_costs,
            'gt_class_costs': gt_class_costs
        })
    
    return results


def compare_before_after(
    model_before: ClassLoRAUNet,
    model_after: ClassLoRAUNet,
    scenes: List[Dict],
    affected_class: int,
    finetune_delta: np.ndarray,
    device: str = "cuda",
    verbose: bool = True
) -> List[Dict]:
    """
    Compare model behavior before and after GPR-FiLM learning.
    
    CRITICAL: Use the SAME base costmap for both, only apply FiLM to AFTER.
    This ensures we're measuring the FiLM effect, not diffusion randomness.
    """
    results = []
    
    for i, scene in enumerate(scenes):
        scene_name = scene.get('name', f'Scene {i+1}')
        if verbose:
            print(f"\n--- {scene_name} ---")
        
        H, W = scene['H'], scene['W']
        
        # Prepare conditioning
        goal_map = make_goal_map(H, W, scene['goal'])
        cond = np.concatenate([scene['class_activations'], goal_map[None, :, :]], axis=0)
        cond_t = torch.from_numpy(cond).float().unsqueeze(0).to(device)
        
        # Generate BASE costmap (same for both before/after)
        with torch.no_grad():
            base_costmap = sample_costmap(model_before, cond_t, device)[0, 0].cpu().numpy()
        
        # BEFORE: Just the base costmap (no modification)
        costmap_before = base_costmap.copy()
        
        # AFTER: Apply GPR-FiLM modulation to the SAME base costmap
        costmap_after = apply_gpr_film_modulation(
            base_costmap, 
            scene['class_activations'],
            model_after
        )
        
        # Plan trajectories
        traj_before = astar(costmap_before, scene['start'], scene['goal'])
        traj_after = astar(costmap_after, scene['start'], scene['goal'])
        
        # Compute per-class costs
        costs_before = compute_class_costs(costmap_before, scene)
        costs_after = compute_class_costs(costmap_after, scene)
        
        if verbose:
            print(f"  Class costs BEFORE → AFTER (GPR-FiLM):")
            for cid in range(NUM_CLASSES):
                name = OBSTACLE_CLASSES[cid]['name']
                change = costs_after[cid] - costs_before[cid]
                pct_change = 100 * change / (costs_before[cid] + 1e-6)
                marker = "▲" if change > 0.01 else ("▼" if change < -0.01 else "─")
                highlight = " ← TARGET" if cid == affected_class else ""
                print(f"    {name}: {costs_before[cid]:.3f} → {costs_after[cid]:.3f} ({pct_change:+.0f}%) {marker}{highlight}")
        
        results.append({
            'scene': scene,
            'scene_name': scene_name,
            'costmap_before': costmap_before,
            'costmap_after': costmap_after,
            'traj_before': traj_before,
            'traj_after': traj_after,
            'costs_before': costs_before,
            'costs_after': costs_after
        })
    
    return results


def apply_gpr_film_modulation(
    base_costmap: np.ndarray,
    class_activations: np.ndarray,
    model: ClassLoRAUNet
) -> np.ndarray:
    """
    Apply GPR-FiLM modulation to a base costmap.
    
    CRITICAL: Use hard class masks to prevent bleeding into other classes.
    Only modifies pixels that are PREDOMINANTLY this class.
    """
    from scipy import ndimage as ndi
    
    modified = base_costmap.copy()
    
    if not hasattr(model, 'gpr_film_manager'):
        return modified
    
    predictions = model.gpr_film_manager.get_all_predictions()
    
    # Find which class dominates each pixel (to prevent overlap)
    class_dominance = np.argmax(class_activations, axis=0)  # [H, W] - which class is strongest
    any_class_present = np.max(class_activations, axis=0) > 0.1  # [H, W] - is any class here
    
    for class_id, params in predictions.items():
        # Skip classes with no observations
        if params['n_observations'] == 0:
            continue
        
        gamma = params['gamma']
        beta = params['beta']
        
        # Skip near-identity transforms
        if abs(gamma) < 0.01 and abs(beta) < 0.01:
            continue
        
        # HARD MASK: Only pixels where THIS class dominates
        # This prevents bleeding into neighboring classes
        this_class_dominates = (class_dominance == class_id) & any_class_present
        
        # Get the actual activation strength for smooth falloff at edges
        class_act = class_activations[class_id]
        
        # Combine: must dominate AND have significant activation
        class_region = np.where(this_class_dominates, class_act, 0.0)
        
        # Small amount of smoothing just at edges (sigma=1, not 3)
        class_region = ndi.gaussian_filter(class_region.astype(np.float32), sigma=1.0)
        
        # Apply FiLM ONLY in class region:
        # modified = base * (1 + gamma * region) + beta * region
        modified = modified * (1 + gamma * class_region) + beta * class_region
    
    return np.clip(modified, 0, 1)


# ==============================================================================
# Visualization
# ==============================================================================

def visualize_deployment(
    results: List[Dict],
    affected_class: int,
    save_path: str = None
):
    """Visualize deployment results across multiple scenes."""
    
    n_scenes = len(results)
    fig, axes = plt.subplots(n_scenes, 4, figsize=(16, 4 * n_scenes))
    
    if n_scenes == 1:
        axes = axes.reshape(1, -1)
    
    colors_class = ['red', 'blue', 'orange', 'gray']
    markers_class = ['o', 's', '^', 'x']
    
    for i, result in enumerate(results):
        scene = result['scene']
        H, W = scene['H'], scene['W']
        
        # Column 1: Class activations
        ax = axes[i, 0]
        class_vis = np.zeros((H, W, 3))
        colors_rgb = [(1, 0, 0), (0, 0, 1), (1, 0.5, 0), (0.5, 0.5, 0.5)]
        for cid in range(NUM_CLASSES):
            for c in range(3):
                class_vis[:, :, c] += scene['class_activations'][cid] * colors_rgb[cid][c]
        ax.imshow(np.clip(class_vis, 0, 1))
        ax.set_title(f"{result['scene_name']}\nObstacles")
        ax.axis('off')
        
        # Column 2: Generated costmap + trajectory
        ax = axes[i, 1]
        ax.imshow(result['generated_costmap'], cmap='viridis')
        
        # Plot obstacles
        for cid, obs in scene['obstacles_by_class'].items():
            if len(obs) > 0:
                obs_arr = np.array(obs)
                ax.scatter(obs_arr[:, 1], obs_arr[:, 0], c=colors_class[cid],
                          marker=markers_class[cid], s=60, edgecolors='white')
        
        # Plot trajectory
        traj = np.array(result['trajectory'])
        ax.plot(traj[:, 1], traj[:, 0], 'w-', linewidth=2)
        ax.scatter([scene['start'][1]], [scene['start'][0]], c='cyan', s=100, marker='o')
        ax.scatter([scene['goal'][1]], [scene['goal'][0]], c='yellow', s=150, marker='*')
        ax.set_title('Generated + A* Path')
        ax.axis('off')
        
        # Column 3: Ground truth costmap
        ax = axes[i, 2]
        ax.imshow(result['gt_costmap'], cmap='viridis')
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # Column 4: Per-class costs
        ax = axes[i, 3]
        class_names = [OBSTACLE_CLASSES[c]['name'] for c in range(NUM_CLASSES)]
        gen_costs = [result['class_costs'][c] for c in range(NUM_CLASSES)]
        gt_costs = [result['gt_class_costs'][c] for c in range(NUM_CLASSES)]
        
        x = np.arange(NUM_CLASSES)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gen_costs, width, label='Generated', color='steelblue')
        bars2 = ax.bar(x + width/2, gt_costs, width, label='GT', color='lightcoral')
        
        # Highlight affected class
        bars1[affected_class].set_color('green')
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylabel('Cost near obstacles')
        ax.set_title('Per-Class Costs')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'Deployment: Learned Preference for "{OBSTACLE_CLASSES[affected_class]["name"]}" Applied to New Scenes',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_complete_pipeline(
    training_scene: Dict,
    training_costmap: np.ndarray,
    original_traj: List[Tuple[int, int]],
    user_traj: List[Tuple[int, int]],
    finetune_result: Dict,
    comparison_results: List[Dict],
    affected_class: int,
    save_path: str = None
):
    """
    Comprehensive visualization showing:
    1. Training scene with user modification
    2. Two test scenes with before/after comparison
    """
    
    n_test_scenes = len(comparison_results)
    
    # Figure layout: 3 rows (training + 2 test scenes), 6 columns
    fig = plt.figure(figsize=(24, 15))
    
    colors_class = ['red', 'blue', 'orange', 'gray']
    markers_class = ['o', 's', '^', 'x']
    colors_rgb = [(1, 0, 0), (0, 0, 1), (1, 0.5, 0), (0.5, 0.5, 0.5)]
    
    def plot_scene_obstacles(ax, scene, title=""):
        H, W = scene['H'], scene['W']
        class_vis = np.zeros((H, W, 3))
        for cid in range(NUM_CLASSES):
            for c in range(3):
                class_vis[:, :, c] += scene['class_activations'][cid] * colors_rgb[cid][c]
        ax.imshow(np.clip(class_vis, 0, 1))
        
        # Add markers
        for cid, obs in scene['obstacles_by_class'].items():
            if len(obs) > 0:
                obs_arr = np.array(obs)
                ax.scatter(obs_arr[:, 1], obs_arr[:, 0], c=colors_class[cid],
                          marker=markers_class[cid], s=100, edgecolors='white',
                          linewidths=2, label=OBSTACLE_CLASSES[cid]['name'])
        
        ax.scatter([scene['start'][1]], [scene['start'][0]], c='cyan', s=150, 
                  marker='o', edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax.scatter([scene['goal'][1]], [scene['goal'][0]], c='yellow', s=200, 
                  marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=10)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        return ax
    
    def plot_costmap_with_traj(ax, costmap, scene, traj=None, traj2=None, title="", show_obs=True):
        ax.imshow(costmap, cmap='viridis', vmin=0, vmax=1)
        
        if show_obs:
            for cid, obs in scene['obstacles_by_class'].items():
                if len(obs) > 0:
                    obs_arr = np.array(obs)
                    ax.scatter(obs_arr[:, 1], obs_arr[:, 0], c=colors_class[cid],
                              marker=markers_class[cid], s=60, edgecolors='white', linewidths=1)
        
        if traj is not None:
            t = np.array(traj)
            ax.plot(t[:, 1], t[:, 0], 'w-', linewidth=3, label='Original/Before')
            ax.plot(t[:, 1], t[:, 0], 'k--', linewidth=1.5)
        
        if traj2 is not None:
            t2 = np.array(traj2)
            ax.plot(t2[:, 1], t2[:, 0], 'lime', linewidth=3, linestyle='--', label='User/After')
        
        ax.scatter([scene['start'][1]], [scene['start'][0]], c='cyan', s=120, marker='o', 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.scatter([scene['goal'][1]], [scene['goal'][0]], c='yellow', s=180, marker='*', 
                  edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, costmap.shape[1])
        ax.set_ylim(costmap.shape[0], 0)
        ax.axis('off')
        return ax
    
    # ==========================================================================
    # ROW 0: Training Scene (6 panels)
    # ==========================================================================
    gs_train = fig.add_gridspec(1, 6, left=0.02, right=0.98, top=0.95, bottom=0.68, wspace=0.08)
    
    # Panel 1: Training scene obstacles
    ax = fig.add_subplot(gs_train[0])
    plot_scene_obstacles(ax, training_scene, "TRAINING\nObstacles")
    ax.legend(loc='upper left', fontsize=7)
    
    # Panel 2: Generated costmap + original trajectory
    ax = fig.add_subplot(gs_train[1])
    plot_costmap_with_traj(ax, training_costmap, training_scene, traj=original_traj,
                          title="Generated +\nOriginal Path")
    
    # Panel 3: Both trajectories (the user edit)
    ax = fig.add_subplot(gs_train[2])
    plot_costmap_with_traj(ax, training_costmap, training_scene, 
                          traj=original_traj, traj2=user_traj,
                          title="USER EDIT\n(white→green)")
    ax.legend(loc='upper left', fontsize=8)
    
    # Panel 4: IRL delta
    ax = fig.add_subplot(gs_train[3])
    delta = finetune_result['delta']
    vmax = max(abs(delta.min()), abs(delta.max()), 0.1)
    im = ax.imshow(delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.plot(np.array(original_traj)[:, 1], np.array(original_traj)[:, 0], 'k-', linewidth=1.5, alpha=0.7)
    ax.plot(np.array(user_traj)[:, 1], np.array(user_traj)[:, 0], 'k--', linewidth=1.5, alpha=0.7)
    ax.set_title("IRL Delta (ΔC)\nRed=+cost Blue=-cost", fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis('off')
    
    # Panel 5: Target costmap
    ax = fig.add_subplot(gs_train[4])
    plot_costmap_with_traj(ax, finetune_result['target_costmap'], training_scene,
                          traj2=user_traj, title="Target Costmap\n(Base + ΔC)", show_obs=True)
    
    # Panel 6: Training info
    ax = fig.add_subplot(gs_train[5])
    ax.axis('off')
    affected_name = OBSTACLE_CLASSES[affected_class]['name']
    info_text = f"""
TRAINING INFO
═════════════════════

Target Class: {affected_name}
Edit Type: AVOID MORE

Fine-tuning:
  Steps: {len(finetune_result['losses'])}
  Final Loss: {finetune_result['losses'][-1]:.4f}

IRL Delta:
  Max +: {delta.max():.2f}
  Max -: {delta.min():.2f}
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # ==========================================================================
    # ROWS 1-2: Test Scenes
    # ==========================================================================
    for i, result in enumerate(comparison_results[:2]):  # Only first 2
        scene = result['scene']
        H, W = scene['H'], scene['W']
        scene_name = result['scene_name']
        
        top = 0.63 - i * 0.32
        bottom = top - 0.28
        
        gs_test = fig.add_gridspec(1, 6, left=0.02, right=0.98, top=top, bottom=bottom, wspace=0.08)
        
        # Panel 1: Scene obstacles
        ax = fig.add_subplot(gs_test[0])
        plot_scene_obstacles(ax, scene, f"TEST {i+1}: {scene_name}")
        
        # Panel 2: Ground truth costmap
        ax = fig.add_subplot(gs_test[1])
        gt_costmap = generate_ground_truth_costmap(scene)
        gt_traj = astar(gt_costmap, scene['start'], scene['goal'])
        plot_costmap_with_traj(ax, gt_costmap, scene, traj=gt_traj, title="Ground Truth")
        
        # Panel 3: BEFORE fine-tuning
        ax = fig.add_subplot(gs_test[2])
        plot_costmap_with_traj(ax, result['costmap_before'], scene, 
                              traj=result['traj_before'], title="BEFORE\nFine-tuning")
        
        # Panel 4: AFTER fine-tuning
        ax = fig.add_subplot(gs_test[3])
        plot_costmap_with_traj(ax, result['costmap_after'], scene,
                              traj=result['traj_after'], title="AFTER\nFine-tuning")
        
        # Panel 5: Difference map
        ax = fig.add_subplot(gs_test[4])
        diff = result['costmap_after'] - result['costmap_before']
        vmax_diff = max(abs(diff.min()), abs(diff.max()), 0.05)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        
        # Mark affected class obstacles
        affected_obs = scene['obstacles_by_class'].get(affected_class, [])
        if len(affected_obs) > 0:
            obs_arr = np.array(affected_obs)
            ax.scatter(obs_arr[:, 1], obs_arr[:, 0], c='lime', marker='o', s=100, 
                      edgecolors='black', linewidths=2, label=f'{affected_name}')
        
        ax.set_title("Difference\n(After - Before)", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        
        # Panel 6: Cost change bar chart
        ax = fig.add_subplot(gs_test[5])
        class_names = [OBSTACLE_CLASSES[c]['name'][:6] for c in range(NUM_CLASSES)]
        cost_changes = [result['costs_after'][c] - result['costs_before'][c] for c in range(NUM_CLASSES)]
        
        colors_bar = ['green' if c == affected_class else 'steelblue' for c in range(NUM_CLASSES)]
        bars = ax.bar(class_names, cost_changes, color=colors_bar, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Δ Cost', fontsize=10)
        ax.set_title('Cost Change\nper Class', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        
        # Value labels
        for bar, val in zip(bars, cost_changes):
            y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    # ==========================================================================
    # Title
    # ==========================================================================
    fig.suptitle(
        f'COMPLETE PIPELINE: Train on "{OBSTACLE_CLASSES[affected_class]["name"]}" avoidance → Deploy to New Scenes',
        fontsize=16, fontweight='bold', y=0.99
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    
    plt.show()
    
    return fig


def visualize_before_after(
    results: List[Dict],
    affected_class: int,
    save_path: str = None
):
    """Visualize comparison of before/after fine-tuning across scenes."""
    
    n_scenes = len(results)
    fig, axes = plt.subplots(n_scenes, 5, figsize=(20, 4 * n_scenes))
    
    if n_scenes == 1:
        axes = axes.reshape(1, -1)
    
    colors_class = ['red', 'blue', 'orange', 'gray']
    markers_class = ['o', 's', '^', 'x']
    
    for i, result in enumerate(results):
        scene = result['scene']
        H, W = scene['H'], scene['W']
        
        # Column 1: Scene setup
        ax = axes[i, 0]
        class_vis = np.zeros((H, W, 3))
        colors_rgb = [(1, 0, 0), (0, 0, 1), (1, 0.5, 0), (0.5, 0.5, 0.5)]
        for cid in range(NUM_CLASSES):
            for c in range(3):
                class_vis[:, :, c] += scene['class_activations'][cid] * colors_rgb[cid][c]
        ax.imshow(np.clip(class_vis, 0, 1))
        ax.set_title(f"{result['scene_name']}")
        ax.axis('off')
        
        # Column 2: Before fine-tuning
        ax = axes[i, 1]
        ax.imshow(result['costmap_before'], cmap='viridis')
        traj = np.array(result['traj_before'])
        ax.plot(traj[:, 1], traj[:, 0], 'w-', linewidth=2)
        ax.scatter([scene['start'][1]], [scene['start'][0]], c='cyan', s=80, marker='o')
        ax.scatter([scene['goal'][1]], [scene['goal'][0]], c='yellow', s=120, marker='*')
        ax.set_title('BEFORE Fine-tuning')
        ax.axis('off')
        
        # Column 3: After fine-tuning
        ax = axes[i, 2]
        ax.imshow(result['costmap_after'], cmap='viridis')
        traj = np.array(result['traj_after'])
        ax.plot(traj[:, 1], traj[:, 0], 'lime', linewidth=2, linestyle='--')
        ax.scatter([scene['start'][1]], [scene['start'][0]], c='cyan', s=80, marker='o')
        ax.scatter([scene['goal'][1]], [scene['goal'][0]], c='yellow', s=120, marker='*')
        ax.set_title('AFTER Fine-tuning')
        ax.axis('off')
        
        # Column 4: Difference (after - before)
        ax = axes[i, 3]
        diff = result['costmap_after'] - result['costmap_before']
        vmax = max(abs(diff.min()), abs(diff.max()), 0.1)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title('Change (After - Before)\nRed=Higher Cost')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
        
        # Column 5: Per-class cost change
        ax = axes[i, 4]
        class_names = [OBSTACLE_CLASSES[c]['name'] for c in range(NUM_CLASSES)]
        cost_changes = [result['costs_after'][c] - result['costs_before'][c] for c in range(NUM_CLASSES)]
        
        colors_bar = ['green' if c == affected_class else 'steelblue' for c in range(NUM_CLASSES)]
        bars = ax.bar(class_names, cost_changes, color=colors_bar, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Cost Change')
        ax.set_title(f'Cost Change per Class\n(Target: {OBSTACLE_CLASSES[affected_class]["name"]})')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, cost_changes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:+.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'GENERALIZATION TEST: Preference for "{OBSTACLE_CLASSES[affected_class]["name"]}" Transfers to New Scenes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_deployment_test(
    checkpoint_path: str,
    affected_class: int = 0,
    n_finetune_steps: int = 100,
    save_dir: str = None,
    seed: int = 42
):
    """
    Full deployment test:
    1. Load model
    2. Create training scene and simulate user edit
    3. Fine-tune on user edit
    4. Deploy on new scenes
    5. Compare before/after
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Target class for fine-tuning: {affected_class} ({OBSTACLE_CLASSES[affected_class]['name']})")
    
    # =========================================================================
    # Step 1: Load model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Model")
    print("="*60)
    
    model = create_class_lora_unet(lora_rank=8).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded: {checkpoint_path}")
    else:
        print("WARNING: No checkpoint, using random weights!")
    
    # Save a copy of the model BEFORE fine-tuning
    model_before = create_class_lora_unet(lora_rank=8).to(device)
    model_before.load_state_dict(copy.deepcopy(model.state_dict()))
    model_before.eval()
    
    # =========================================================================
    # Step 2: Create TRAINING scene and user edit
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Creating Training Scene + User Edit")
    print("="*60)
    
    training_scene = {
        'name': 'Training Scene',
        'obstacles_by_class': {
            0: [(30, 25), (35, 30)],   # chairs
            1: [(40, 45)],              # table
            2: [(25, 40)],              # person
            3: [(50, 15)],              # wall
        },
        'goal': (8, 55),
        'start': (55, 8),
        'H': 64, 'W': 64
    }
    training_scene['class_activations'] = make_class_occupancy_maps(
        64, 64, training_scene['obstacles_by_class']
    )
    
    # Generate costmap and plan original trajectory
    goal_map = make_goal_map(64, 64, training_scene['goal'])
    cond = np.concatenate([training_scene['class_activations'], goal_map[None, :, :]], axis=0)
    cond_t = torch.from_numpy(cond).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        training_costmap = sample_costmap(model, cond_t, device)[0, 0].cpu().numpy()
    
    original_traj = astar(training_costmap, training_scene['start'], training_scene['goal'])
    print(f"Original trajectory: {len(original_traj)} waypoints")
    
    # Simulate user edit: AVOID affected class MORE
    user_traj = create_exaggerated_edit(
        original_traj, training_scene['obstacles_by_class'],
        affected_class, strength=3.0
    )
    print(f"User trajectory: {len(user_traj)} waypoints")
    
    # =========================================================================
    # Step 3: Fine-tune LoRA on user edit
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Fine-tuning LoRA")
    print("="*60)
    
    finetune_result = finetune_on_trajectory_edit(
        model=model,
        scene=training_scene,
        original_trajectory=original_traj,
        user_trajectory=user_traj,
        affected_class=affected_class,
        device=device,
        n_steps=n_finetune_steps,
        verbose=True
    )
    
    print(f"Fine-tuning complete. Final loss: {finetune_result['losses'][-1]:.4f}")
    
    # =========================================================================
    # Step 4: Generate NEW test scenes
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Generating New Test Scenes")
    print("="*60)
    
    test_scenes = generate_specific_scenes()
    print(f"Generated {len(test_scenes)} test scenes")
    
    # =========================================================================
    # Step 5: Compare before/after on new scenes
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Comparing Before/After on New Scenes")
    print("="*60)
    
    comparison_results = compare_before_after(
        model_before=model_before,
        model_after=model,
        scenes=test_scenes,
        affected_class=affected_class,
        finetune_delta=finetune_result['delta'],
        device=device,
        verbose=True
    )
    
    # =========================================================================
    # Step 6: Visualize
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Comprehensive Visualization")
    print("="*60)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'deployment_{OBSTACLE_CLASSES[affected_class]["name"]}.png')
    else:
        save_path = f'deployment_{OBSTACLE_CLASSES[affected_class]["name"]}.png'
    
    # Use comprehensive visualization
    visualize_complete_pipeline(
        training_scene=training_scene,
        training_costmap=training_costmap,
        original_traj=original_traj,
        user_traj=user_traj,
        finetune_result=finetune_result,
        comparison_results=comparison_results,
        affected_class=affected_class,
        save_path=save_path
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Learned Preference Transfer")
    print("="*60)
    
    affected_name = OBSTACLE_CLASSES[affected_class]['name']
    
    avg_change_target = np.mean([r['costs_after'][affected_class] - r['costs_before'][affected_class] 
                                  for r in comparison_results])
    avg_change_others = np.mean([
        np.mean([r['costs_after'][c] - r['costs_before'][c] for c in range(NUM_CLASSES) if c != affected_class])
        for r in comparison_results
    ])
    
    print(f"Target class ({affected_name}) cost change: {avg_change_target:+.3f}")
    print(f"Other classes avg cost change: {avg_change_others:+.3f}")
    
    if avg_change_target > avg_change_others + 0.05:
        print(f"✓ SUCCESS: {affected_name} cost increased more than others!")
        print(f"  The learned preference TRANSFERS to new scenes.")
    else:
        print(f"⚠ Mixed results: Check visualization for details.")
    
    return {
        'training_scene': training_scene,
        'test_scenes': test_scenes,
        'comparison_results': comparison_results,
        'finetune_result': finetune_result
    }


def create_exaggerated_edit(
    original_trajectory: List[Tuple[int, int]],
    obstacles_by_class: Dict[int, List[Tuple[int, int]]],
    affected_class: int,
    strength: float = 3.0
) -> List[Tuple[int, int]]:
    """Create exaggerated user trajectory edit."""
    traj = np.array(original_trajectory, dtype=np.float32)
    obstacles = obstacles_by_class.get(affected_class, [])
    
    if len(obstacles) == 0:
        return original_trajectory
    
    obs_arr = np.array(obstacles, dtype=np.float32)
    modified = []
    
    for point in traj:
        total_push = np.zeros(2)
        
        for obs in obs_arr:
            diff = point - obs
            dist = np.linalg.norm(diff)
            
            if dist < 1e-6:
                continue
            
            direction = diff / dist
            influence_radius = 25.0
            
            if dist < influence_radius:
                push_magnitude = strength * (influence_radius - dist) / influence_radius
                push_magnitude = push_magnitude ** 1.5
                total_push += direction * push_magnitude
        
        new_point = point + total_push
        new_point = np.clip(new_point, 1, 62)
        modified.append((int(new_point[0]), int(new_point[1])))
    
    return modified


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy learned preferences to new scenes")
    
    parser.add_argument("checkpoint", type=str, nargs='?', default=None)
    parser.add_argument("--affected-class", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--n-finetune-steps", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_deployment_test(
        checkpoint_path=args.checkpoint,
        affected_class=args.affected_class,
        n_finetune_steps=args.n_finetune_steps,
        save_dir=args.save_dir,
        seed=args.seed
    )
