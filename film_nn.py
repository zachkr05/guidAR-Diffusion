# film_nn.py - Single-sample convergence version

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as ndi
from typing import Dict, List, Tuple


def compute_optimal_film_params(
    current_costmap: np.ndarray,
    target_costmap: np.ndarray,
    class_activation: np.ndarray,
    influence_size: int = 15,
    sigma: float = 4.0
) -> Tuple[float, float]:
    """
    Analytically solve for optimal (gamma, beta) given current and target costmaps.
    
    We want to find gamma, beta such that:
        target ≈ current * (1 + gamma * region) + beta * region
    
    Rearranging:
        target - current ≈ gamma * current * region + beta * region
        delta ≈ gamma * (current * region) + beta * region
    
    This is a simple linear regression: delta = gamma * X1 + beta * X2
    """
    # Compute class region
    region = ndi.maximum_filter(class_activation, size=influence_size)
    region = ndi.gaussian_filter(region.astype(np.float32), sigma=sigma)
    
    if region.max() < 0.01:
        return 0.0, 0.0
    
    region = region / (region.max() + 1e-8)
    
    # Only fit in region where class has influence
    mask = region > 0.1
    if mask.sum() < 10:
        return 0.0, 0.0
    
    delta = (target_costmap - current_costmap)[mask]
    current_masked = current_costmap[mask]
    region_masked = region[mask]
    
    # Linear regression: delta = gamma * (current * region) + beta * region
    X1 = current_masked * region_masked  # gamma coefficient
    X2 = region_masked                    # beta coefficient
    
    # Solve least squares: [X1, X2] @ [gamma, beta]^T = delta
    A = np.column_stack([X1, X2])
    
    try:
        params, residuals, rank, s = np.linalg.lstsq(A, delta, rcond=None)
        gamma, beta = params
    except:
        # Fallback: simple average
        gamma = 0.0
        beta = np.mean(delta)
    
    # Clip to reasonable range
    gamma = float(np.clip(gamma, -3.0, 3.0))
    beta = float(np.clip(beta, -1.5, 1.5))
    
    return gamma, beta


class SingleShotFiLMManager:
    """
    FiLM manager that converges in a single sample.
    
    Strategy:
    1. Analytically compute optimal (gamma, beta) from IRL delta
    2. Store per-class params (accumulate across scenes)
    3. Apply exponential moving average for stability
    """
    def __init__(
        self,
        num_classes: int = 4,
        grid_size: int = 64,
        ema_decay: float = 0.3,  # How much to weight new observations
        device: str = "cuda"
    ):
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.ema_decay = ema_decay
        self.device = device
        
        # Current params per class
        self.gammas = {c: 0.0 for c in range(num_classes)}
        self.betas = {c: 0.0 for c in range(num_classes)}
        
        # Observation counts
        self.observation_counts = {c: 0 for c in range(num_classes)}
        
        # History for debugging
        self.history: List[Dict] = []
    
    def update_from_edit(
        self,
        semantic_map: np.ndarray,      # [num_classes, H, W]
        current_costmap: np.ndarray,   # [H, W]
        target_costmap: np.ndarray,    # [H, W]
        affected_class: int,
        verbose: bool = True
    ) -> Dict:
        """
        Single-shot update: analytically compute optimal params and apply EMA.
        """
        # Compute optimal (gamma, beta) for affected class
        gamma_new, beta_new = compute_optimal_film_params(
            current_costmap=current_costmap,
            target_costmap=target_costmap,
            class_activation=semantic_map[affected_class]
        )
        
        # Apply exponential moving average
        n = self.observation_counts[affected_class]
        if n == 0:
            # First observation: use directly
            self.gammas[affected_class] = gamma_new
            self.betas[affected_class] = beta_new
        else:
            # EMA update
            decay = self.ema_decay
            self.gammas[affected_class] = (1 - decay) * self.gammas[affected_class] + decay * gamma_new
            self.betas[affected_class] = (1 - decay) * self.betas[affected_class] + decay * beta_new
        
        self.observation_counts[affected_class] += 1
        
        # Store history
        self.history.append({
            'class': affected_class,
            'gamma_new': gamma_new,
            'beta_new': beta_new,
            'gamma_ema': self.gammas[affected_class],
            'beta_ema': self.betas[affected_class],
        })
        
        if verbose:
            print(f"  Class {affected_class}: computed γ={gamma_new:.3f}, β={beta_new:.3f}")
            print(f"  Class {affected_class}: EMA γ={self.gammas[affected_class]:.3f}, β={self.betas[affected_class]:.3f}")
            print(f"  Observations for class {affected_class}: {self.observation_counts[affected_class]}")
        
        return {
            'gamma': self.gammas[affected_class],
            'beta': self.betas[affected_class],
            'gamma_raw': gamma_new,
            'beta_raw': beta_new,
        }
    
    def apply_to_costmap(
        self,
        costmap: np.ndarray,
        semantic_map: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """Apply learned FiLM params to costmap."""
        modified = costmap.copy()
        
        for c in range(self.num_classes):
            gamma = self.gammas[c] * strength
            beta = self.betas[c] * strength
            
            if abs(gamma) < 0.001 and abs(beta) < 0.001:
                continue
            
            # Compute class region
            class_act = semantic_map[c]
            region = ndi.maximum_filter(class_act, size=15)
            region = ndi.gaussian_filter(region.astype(np.float32), sigma=4.0)
            if region.max() > 0:
                region = region / region.max()
            
            # Apply FiLM
            modified = modified * (1 + gamma * region) + beta * region
        
        return np.clip(modified, 0, 1)
    
    def get_observation_count(self) -> int:
        return sum(self.observation_counts.values())
    
    def get_params_summary(self) -> str:
        lines = []
        for c in range(self.num_classes):
            n = self.observation_counts[c]
            if n > 0:
                lines.append(f"  Class {c}: γ={self.gammas[c]:.3f}, β={self.betas[c]:.3f} (n={n})")
            else:
                lines.append(f"  Class {c}: no data")
        return "\n".join(lines)
