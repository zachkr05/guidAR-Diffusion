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
from planner import SoftGridPlanner
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


def make_expert_target(user_path, H, W, device):
    """
    Converts a user path (N, 2) [x, y] into a soft target image (1, 1, H, W).
    """
    target = torch.zeros((1, 1, H, W), device=device)
    
    # user_path comes in as (N, 2) numpy array
    if isinstance(user_path, np.ndarray):
        path_tensor = torch.from_numpy(user_path).float().to(device)
    else:
        path_tensor = user_path

    # Extract coordinates. Assuming user_path is [x, y] (col, row)
    # We clip to ensure we don't crash if the path touches the edge
    xs = path_tensor[:, 0].long().clamp(0, W-1)
    ys = path_tensor[:, 1].long().clamp(0, H-1)
    
    # "Draw" the path onto the grid
    target[0, 0, ys, xs] = 1.0
    
    # Smooth the target slightly (Blur) so the loss isn't too harsh
    # This helps the optimizer "find" the path if the prediction is slightly off
    target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    target = target / (target.sum() + 1e-8) # Normalize to sum to 1
    
    return target

def finetune_models(model, batch, user_path, device,lr, target_class,wH,wF,wK,wL,epochs):
   

    features, targets, positions, radii, goal = batch
    expert_model = model.experts[target_class]
    expert_model.set_finetune(active=True)

    optimizer = AdamW(
            [p for p in expert_model.parameters() if p.requires_grad],
            lr = lr
            )
    planner = SoftGridPlanner(iters=80, tau=1.0, step_cost=0.05).to(device)
    
    H, W = 128, 128
    target_visitation = make_expert_target(user_path, H, W, device)
    ddpm = DDPM(timesteps=1000, device = device)

    loss_history = []


#    avg_cost = None
#    alpha_baseline = 0.9

    for epoch in tqdm(range(epochs), desc = "IRL Finetuning"):

        model.train()
        optimizer.zero_grad()

        conditioning = features[target_class].to(device)
        x_0 = targets[target_class].to(device)

        B = x_0.shape[0]
        t = torch.randint(0, ddpm.timesteps, (B,), device=device).long()

        x_t, noise = ddpm.q_sample(x_0,t)

        noise_pred = expert_model(x_t, t, conditioning)

        x_0_pred_target = ddpm.predict_start_from_noise(x_t, t, noise_pred)
        #costmap_dict = {}

        #generated, log_prob = ddpm.sample_with_partial_logprob(
        #            expert_model, 
        #            conditioning, 
        #            shape=x_0.shape,
        #            logprob_steps=50
        #        )
        
        maps_to_fuse = []
        for cls in model.obstacle_classes:
            if cls == target_class:
                maps_to_fuse.append(x_0_pred_target)
            else:
                maps_to_fuse.append(targets[cls].to(device).detach())
       
        stack_maps = torch.stack(maps_to_fuse, dim=1)
        fused_map_raw = torch.logsumexp(stack_maps, dim=1)

        # 6. Planner Loss
        cost_map_fused = F.softplus(fused_map_raw) + 0.1
        pred_visitation = planner(cost_map_fused, goal.to(device))
        
        loss_plan = -(target_visitation * torch.log(pred_visitation + 1e-8)).sum()
        loss_diff = F.mse_loss(noise_pred, noise)

        total_loss = loss_diff + (2.0 * loss_plan)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(expert_model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(total_loss.item())
    print(loss_history)
    expert_model.set_finetune(active=False)
    return loss_history
        

    """
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

            #if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{epochs} - Trajectory Cost: {traj_cost:.4f}") 

        except (ValueError, RuntimeError) as e:
                    print(f"\nWarning: Could not compute path at epoch {epoch + 1}: {e}")
                    continue 
    """
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
    finetune_models(model=model, batch=first_batch, user_path=user_path, device=device, lr=1e-3, epochs=100, target_class="chair", wH=1, wF = 0, wK = 0.01, wL=0.0)


    original_image_tensor = diffused_cm["chair"][0]
    
    visualize_improvement(
        model=model,
        target_class="chair",
        features=features,
        device=device,
        original_generated=original_image_tensor
    )
    #visualize_costmap()
    
def visualize_improvement(model, target_class, features, device, original_generated):
    """
    Generates a new sample from the fine-tuned model, subtracts the old sample,
    and displays the delta.
    """
    print(f"\nGenering 'After' image for {target_class}...")
    
    # 1. Setup Model and Sampling
    expert_model = model.experts[target_class]
    expert_model.eval() # Ensure we are in eval mode for sampling
    ddpm = DDPM(timesteps=1000, device=device)
    
    cond = features[target_class].to(device)
    # We use the shape from the original generation to match dimensions
    shape = original_generated.shape 

    # 2. Generate "After" Image
    with torch.no_grad():
        new_generated = ddpm.sample(expert_model, cond, shape=shape)

    # 3. Convert to Numpy for Plotting
    # Squeeze to remove Batch and Channel dims -> (H, W)
    img_old = original_generated[0, 0].cpu().numpy()
    img_new = new_generated[0, 0].cpu().numpy()
    
    # Calculate Delta
    delta = img_new - img_old

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot: Before
    im1 = axes[0].imshow(img_old, cmap='viridis', origin='lower')
    axes[0].set_title(f"Before Fine-tuning ({target_class})")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot: After
    im2 = axes[1].imshow(img_new, cmap='viridis', origin='lower')
    axes[1].set_title(f"After Fine-tuning ({target_class})")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot: Delta (Difference)
    # Use 'seismic' or 'bwr' (Blue-White-Red) to show negative vs positive change
    # Center the colormap at 0 using vmin/vmax
    max_val = max(abs(np.min(delta)), abs(np.max(delta)))
    im3 = axes[2].imshow(delta, cmap='seismic', origin='lower', vmin=-max_val, vmax=max_val)
    axes[2].set_title("Delta (New - Old)\nRed = Cost Increased | Blue = Cost Decreased")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    
    # Optional: Save it
    plt.savefig(f"delta_improvement_{target_class}.png")
    print(f"Visualization saved to delta_improvement_{target_class}.png")

if __name__ == "__main__":
    evaluate() 
