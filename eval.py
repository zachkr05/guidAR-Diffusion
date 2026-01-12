
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from eval.config import EvalConfig
from skimage.graph import route_through_array
from DataGenerator.dataset_costmap import MultiClassCostmapDataset
from DataGenerator.sim import NUM_CLASSES
from typing import Tuple, Optional
from train import cosine_beta_schedule
from plotter import *


def compute_expert_channels(num_classes: int) -> int:
    """
    Compute the number of input channels for each expert.
    
    Layout: Base(5) + (N-1)*3 Context Inputs + Goal(1)
    
    Note: The context channels should be (num_classes - 1) * 3 because
    each expert doesn't need context about itself.
    
    Args:
        num_classes: Number of obstacle classes
        
    Returns:
        Number of expert input channels
    """
    # Base channels: occupancy(1) + orientation(2) + EDF(1) + noise(1) = 5
    base_channels = 5
    
    # Context: for each OTHER class (num_classes - 1), we have 3 channels
    # (occupancy, orientation_x, orientation_y or similar)
    context_channels = (num_classes - 1) * 3
    
    # Goal channel
    goal_channel = 1
    
    total = base_channels + context_channels + goal_channel
    return total

def load_model(
    checkpoint: str,
    num_classes: int,
    device: torch.device,
    expert_input_channels: Optional[int] = None
) -> nn.Module:
    """
    Load a CompositionalModel from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        num_classes: Number of obstacle classes
        device: Torch device
        expert_input_channels: Override channel count (auto-computed if None)
        
    Returns:
        Loaded model
    """
    # Import here to avoid circular imports
    from MoE.compositional_model import CompositionalModel
    
    if expert_input_channels is None:
        expert_input_channels = compute_expert_channels(num_classes)
    
    print(f"Initializing model with {num_classes} classes, {expert_input_channels} expert channels")
    
    model = CompositionalModel(
        num_classes=num_classes,
        expert_input_channels=expert_input_channels
    ).to(device)
    
    ckpt = torch.load(checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Try to infer channel count from checkpoint if there's a mismatch
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        error_str = str(e)
        if "size mismatch" in error_str:
            print(f"Warning: Size mismatch detected. Error: {e}")
            print("Attempting to infer correct channel count from checkpoint...")
            
            # Try to find the context conv weight to determine correct channels
            for key, value in state_dict.items():
                if 'context' in key and 'weight' in key and len(value.shape) == 4:
                    inferred_channels = value.shape[1]
                    print(f"Found context layer expecting {inferred_channels} input channels")
                    
                    # Recreate model with correct channels
                    model = CompositionalModel(
                        num_classes=num_classes,
                        expert_input_channels=5 + inferred_channels + 1  # base + context + goal
                    ).to(device)
                    model.load_state_dict(state_dict)
                    print("Successfully loaded with inferred channel count")
                    break
            else:
                raise e
    
    return model



def plan_path(costmap, start, goal):
    indices, weight = route_through_array(costmap, start, goal)
    indices = np.stack(indices, axis=-1)  
    return indices

def sample_base(model, cond, device, img_size, num_steps, training_phase):
        batch_size = cond.shape[0]

        #setup the schuedle
        betas = cosine_beta_schedule(num_steps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)

        xt = torch.randn(batch_size, 1, img_size, img_size, device= device)


        with torch.no_grad():
            for t in reversed(range(num_steps)):
                time = torch.full((batch_size,), t, device=device).long()
                costmap, all_costs = model(xt, time, cond, training_phase = training_phase)
                
                beta_t = betas[t]
                alpha_t = alphas[t]
                alpha_bar_t = alphas_cumprod[t]
                alpha_bar_prev = alphas_cumprod_prev[t]


                posterior_mean = (
                    (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * costmap +
                    (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * xt
                )

                if t > 0:
                    noise = torch.randn_like(xt)
                    # Standard posterior variance
                    posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    log_var = torch.log(torch.clamp(posterior_variance, min=1e-20))
                    xt = posterior_mean + torch.exp(0.5 * log_var) * noise
                else:
                    xt = posterior_mean # No noise at the very last step
        return costmap, all_costs


def post_process_costmap(diffused_output):
    # 1. Move to CPU and numpy
    costmap = diffused_output.squeeze().cpu().numpy()
    
    # 2. Denormalize: Transform [-1, 1] back to [0, 1]
    # (Assuming your dataset used x_norm = 2 * x - 1)
    costmap = (costmap + 1.0) / 2.0
    
    # 3. Clip to ensure valid range (removes tiny errors like -0.0001 or 1.0001)
    costmap = np.clip(costmap, 0.0, 1.0)
    
    return costmap

def run_eval(config: EvalConfig):
    device= config.device

    ds = MultiClassCostmapDataset(
            n_samples = config.n_samples,
            H=config.img_size,
            W=config.img_size
            )

    model = load_model(
            checkpoint = config.checkpoint,
            num_classes = NUM_CLASSES,
            device = device,
            expert_input_channels=None)

    for i in range(25):
        #Get the data 
        cond, x0, obstacles, goal = ds.get_sample_with_metadata(i)
        cond_t = cond.unsqueeze(0).to(device)

        #Predict base map
        pred_base_t, all_costs = sample_base(
                model, cond_t, device,
                img_size = config.img_size,
                num_steps = 1000,
                training_phase="base")
        pred_base = pred_base_t.squeeze().cpu().numpy()
        pred_base = post_process_costmap(pred_base_t) 
        print("goal: ",goal)
        print("OBSTACLES \n", obstacles)
        print("Generating image for predicted costmap")
        visualize_costmap(pred_base)
        return 
        #Get initial path
#        path_base = plan_path(pred_base, [np.rand.randint(), np.rand.randint()], goal)
#       path_arr = np.array(path_base)

        #Get user corrections
#        adj_rows, adj_cols = simulate_user_correction(
#            path_arr[:, 0], path_arr[:, 1],
#            obstacles,
#            target_class=config.target_class,
#            avoidance_radius=config.avoidance_radius,
#            push_strength=config.push_strength,
#            img_size=config.img_size
#        )
#        path_user = list(zip(adj_rows, adj_cols))

#        if (path_user == path_arr):
#            continue
        
        #Get target costmap
#        delta_np = compute_geometric_delta(path_base, path_user, config.img_size, config.img_size)
#        delta_t = delta_to_tensor(delta_np, device)

#        target_costmap_np = delta_np + pred_base

#        affected_classes = conformal_prediction(obstacles, delta_np, path_arr, path_user)
        
def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion-based costmap models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Number of diffusion timesteps (for training reference)"
    )
    parser.add_argument(
        "--img_size", type=int, default=64,
        help="Image size (height and width)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--n_samples", type=int, default=3,
        help="Number of samples to evaluate"
    )
   
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        checkpoint=args.checkpoint,
        timesteps=args.timesteps,
        img_size=args.img_size,
        device=args.device,
        n_samples=args.n_samples,
    )
    
    run_eval(config)

if __name__ == "__main__":
    main()
