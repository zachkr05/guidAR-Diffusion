"""
IRL finetuning for MoE diffusion costmap generation.

Simpler approach: Direct planning loss without complex masking.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from .utils import gaussian_blur

#Draw 1's along user path anbd gaussian blur them
def make_path_target(path, H, W, device, sigma=5.0):
    target = torch.zeros(1, 1, H, W, device=device)
    coords = path.astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, W-1)
    coords[:, 1] = np.clip(coords[:, 1], 0, H-1)
    target[0, 0, coords[:, 1], coords[:, 0]] = 1.0
    target = gaussian_blur(target, kernel_size=int(6*sigma+1)|1, sigma=sigma)
    return target / (target.sum() + 1e-8)

def finetune_models(
    model, batch, orig_path, user_path, device, lr, edit_regions, epochs,
    ddpm, planner,
    w_diffusion=1.0,
    w_cost_direct=1.0,
    w_plan_attract=1.0,
    w_plan_repulse=0.5,
    obstacle_classes=None,
):
    features, targets, positions, radii, goal, orientations = batch
    H, W = 128, 128

    loss_history = []

    for region_mask, points, affected_classes, class_contributions in edit_regions:

        expert_models = {}
        x_0_gts = {}
        conditionings = {}

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=True)
            expert_models[cls] = model.experts[cls]
            x_0_gts[cls] = targets[cls].to(device)
            conditionings[cls] = features[cls].to(device)

        frozen_maps = {}
        for cls in model.obstacle_classes:
            if cls not in affected_classes:
                frozen_maps[cls] = targets[cls].to(device).detach()

        optimizer = AdamW(
            [p for expert in expert_models.values()
             for p in expert.parameters() if p.requires_grad],
            lr=lr,
        )


        # --- Pre-compute targets and masks ---
        user_path_target = make_path_target(user_path, H, W, device, sigma=5.0)
        orig_path_target = make_path_target(orig_path, H, W, device, sigma=5.0)


        repulsive_target = torch.clamp(orig_path_target - user_path_target, min=0.0)
        repulsive_target = repulsive_target / (repulsive_target.sum() + 1e-8)

        region_mask_t = torch.from_numpy(region_mask).float().to(device).unsqueeze(0).unsqueeze(0)
        repulsive_target = repulsive_target * region_mask_t
        repulsive_target = repulsive_target / (repulsive_target.sum() + 1e-8)

        deviation = torch.abs(user_path_target - orig_path_target)
        deviation = deviation / (deviation.max() + 1e-8)

        for epoch in tqdm(range(epochs), desc="IRL Finetuning"):
            model.train()
            optimizer.zero_grad()

            B = list(x_0_gts.values())[0].shape[0]
            t = torch.randint(0, ddpm.timesteps // 4, (B,), device=device).long()

            x_0_preds = {}
            noises = {}
            noise_preds = {}

            for cls in affected_classes:
                x_t, noise = ddpm.q_sample(x_0_gts[cls], t)
                noise_pred = expert_models[cls](x_t, t, conditionings[cls])
                x_0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)
                x_0_preds[cls] = x_0_pred
                noises[cls] = noise
                noise_preds[cls] = noise_pred

            # Normalize precomputed contributions to only affected classes
            total = sum(class_contributions[cls] for cls in affected_classes) + 1e-8
            expert_weights = {
                cls: class_contributions[cls] / total for cls in affected_classes
            }

            # Loss 1: Weighted diffusion loss
            loss_diffusion = sum(
                expert_weights[cls] * F.mse_loss(noise_preds[cls], noises[cls])
                for cls in affected_classes
            )

            # Fuse all maps
            maps_to_fuse = []
            for cls in model.obstacle_classes:
                if cls in x_0_preds:
                    maps_to_fuse.append(x_0_preds[cls])
                else:
                    maps_to_fuse.append(frozen_maps[cls])

            stacked = torch.stack(maps_to_fuse, dim=1).squeeze(2)
            fused_map = torch.logsumexp(stacked, dim=1, keepdim=True)
            cost_map = F.softplus(fused_map) + 0.1

            pred_visitation = planner(cost_map, goal.to(device))

            # Loss 2: attract loss — masked by orientation-aware edit region
            #user_target_in_edit = user_path_target * 
            #user_target_in_edit = user_path_target / (user_path_target.sum() + 1e-8)
            


            #deviation = torch.abs(user_path_target - orig_path_target)
            #deviation = deviation / (deviation.max() + 1e-8) 
            #weighted_target = user_path_target * (1.0 + 5.0 * deviation)
            #weighted_target = weighted_target / (weighted_target.sum() + 1e-8)
            #loss_attract = -(weighted_target * torch.log(pred_visitation + 1e-8)).sum()
            #loss_attract = -(user_target_in_edit * torch.log(pred_visitation + 1e-8)).sum()


            #Loss 3: repulsive
            #jloss_repulse = (repulsive_target * torch.log(pred_visitation + 1e-8)).sum()
            # Direct costmap supervision in repulsion region

            #total_loss = (w_diffusion * loss_diffusion
            #              + w_attract * loss_attract) 
            #             # + w_repulse * loss_repulse_direct)

	    # --- Build edit-region masks ---
            user_mask = (user_path_target > user_path_target.max() * 0.1).float()
            orig_mask = (orig_path_target > orig_path_target.max() * 0.1).float()

	    # Repulsion region: on original path but NOT on user path, inside edit region
            repulse_mask = orig_mask * (1.0 - user_mask) * region_mask_t
	    # Attraction region: on user path but NOT on original, inside edit region
            attract_mask = user_mask * (1.0 - orig_mask) * region_mask_t

	    # Loss 1: diffusion regularizer (unchanged — keeps experts grounded)
            loss_diffusion = sum(expert_weights[cls] * F.mse_loss(noise_preds[cls], noises[cls]) for cls in affected_classes)

	    # Loss 2: direct costmap supervision
	    # Push cost UP where user diverged away from
	    # Push cost DOWN where user went toward
            eps = 1e-8
            loss_cost_repulse = -(cost_map * repulse_mask).sum() / (repulse_mask.sum() + eps)
            loss_cost_attract =  (cost_map * attract_mask).sum() / (attract_mask.sum() + eps)
            loss_cost_direct = loss_cost_repulse + loss_cost_attract  # minimize: up on repulse, down on attract

	    # Loss 3: planner-based attract (keeps global path consistency)
            weighted_target = user_path_target * region_mask_t
            weighted_target = weighted_target / (weighted_target.sum() + eps)
            loss_plan_attract = -(weighted_target * torch.log(pred_visitation + eps)).sum()

	    # Loss 4: planner-based repulse (mass should NOT flow through repulse region)
            repulsive_target_planner = repulse_mask / (repulse_mask.sum() + eps)
            loss_plan_repulse = (repulsive_target_planner * torch.log(pred_visitation + eps)).sum()
	    # note: this is positive log(p); minimizing drives p->0 in repulse region
            total_loss = (
		    w_diffusion * loss_diffusion
		    + 1.0 * loss_cost_direct        # the big win — direct, crisp signal
		    + w_plan_attract * loss_plan_attract
		    + 0.5 * loss_plan_repulse
		)
				
            total_loss.backward()
            for cls in affected_classes:
                torch.nn.utils.clip_grad_norm_(expert_models[cls].parameters(), 1.0)
            optimizer.step()

            loss_history.append({
                'total': total_loss.item(),
                'diffusion': loss_diffusion.item(), 
                #'repulse': loss_repulse_direct.item(),
                #'plan': loss_attract.item(),
            })

            if epoch % 100 == 0 or epoch == epochs - 1:
                weight_str = ", ".join(f"{cls}={w:.2f}" for cls, w in expert_weights.items())
                print(f"\n  [Epoch {epoch}] total={total_loss.item():.4f}, "
                        f"diffusion={loss_diffusion.item():.4f} "
                      f"weights=[{weight_str}]")

        for cls in affected_classes:
            model.experts[cls].set_finetune(active=False)

    return loss_history


