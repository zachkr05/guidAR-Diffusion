# irl_finetune.py - GPR version

import numpy as np
import torch
from typing import List, Tuple, Dict
from film import GPRFiLMManager
from irl_agent import compute_geometric_delta


def finetune_gpr_film_on_edit(
    gpr_film_manager: GPRFiLMManager,
    current_costmap: np.ndarray,
    class_activations: np.ndarray,
    original_traj: List[Tuple[int, int]],
    user_traj: List[Tuple[int, int]],
    affected_class: int,
    H: int,
    W: int,
    verbose: bool = True
) -> Dict:
    """
    Update GPR-FiLM based on user trajectory edit.
    
    No gradient training - just adds an observation to the GPR.
    """
    # Compute geometric IRL delta
    delta = compute_geometric_delta(original_traj, user_traj, H, W)
    target_costmap = np.clip(current_costmap + delta, 0, 1)
    
    # Compute target (gamma, beta) from IRL delta
    target_gamma, target_beta = gpr_film_manager.compute_target_params_from_irl(
        current_costmap=current_costmap,
        target_costmap=target_costmap,
        class_activation=class_activations[affected_class]
    )
    
    if verbose:
        print(f"  Target FiLM params: γ={target_gamma:.3f}, β={target_beta:.3f}")
    
    # Add observation to GPR (this is the "training" - instant!)
    gpr_film_manager.add_trajectory_edit(
        original_traj=original_traj,
        user_traj=user_traj,
        affected_class=affected_class,
        target_gamma=target_gamma,
        target_beta=target_beta
    )
    
    if verbose:
        print(f"  Added observation to GPR for class {affected_class}")
    
    return {
        'delta': delta,
        'target_costmap': target_costmap,
        'gamma': target_gamma,
        'beta': target_beta
    }
