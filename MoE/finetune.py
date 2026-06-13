"""
Online IRL-style finetuning for the UNIFIED trajectory-diffusion model.

The model now emits a trajectory heatmap directly. There are no per-class
experts, no logsumexp fusion, and no planner in the loop: the model's own x0
prediction already plays the role that `planner(cost_map)` used to play.

So the old chain
        per-class experts -> LSE -> cost_map -> planner -> visitation
collapses to
        model -> x0_pred (== the path distribution)

and the same three-term loss applies directly to that prediction:

    L = w_diffusion * MSE(eps_pred, eps)          # stay a *valid* path for this scene
      + w_attract   * CE(user_path,   p)          # bend toward the user's edit
      + w_repulse   * push p off (orig minus user)  # leave the old groove behind

where  p = normalized, non-negative version of the predicted trajectory.

Only LoRA + FiLM train (`model.set_finetune(active=True)`). FiLM is *activated*
by passing a film_cond vector; with it set, the FiLM weights receive gradient
and act as a learned global affine on the decoder, alongside the LoRA deltas.
LoRA is the workhorse for cross-scene generalization; FiLM adds fitting capacity
for the single demonstrated scene.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .utils import gaussian_blur


from scipy.ndimage import distance_transform_edt

def make_distance_penalty(path_xy, H, W, device):
    """
    Creates a continuous spatial gradient field pointing toward the path.
    Returns a (1, 1, H, W) tensor where value = distance to nearest path pixel.
    """
    grid = np.ones((H, W), dtype=np.float32)
    coords = np.asarray(path_xy).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
    
    # 0.0 exactly on the path, 1.0 everywhere else
    grid[coords[:, 1], coords[:, 0]] = 0.0

    # Calculate Euclidean distance to the 0.0 pixels
    dist = distance_transform_edt(grid)

    # Normalize so the maximum distance is 1.0 (keeps loss stable)
    dist = dist / (np.max(dist) + 1e-8)
    
    return torch.from_numpy(dist).float().unsqueeze(0).unsqueeze(0).to(device)

def make_path_target(path_xy, H, W, device, sigma=5.0):
    """Rasterize an (x, y) = (col, row) polyline into a normalized heatmap.

    path_xy : (M, 2) array; column 0 = x (col), column 1 = y (row).
    Returns (1, 1, H, W) summing to 1.  (Same convention as the old code.)
    """
    target = torch.zeros(1, 1, H, W, device=device)
    coords = np.asarray(path_xy).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)   # x -> col index
    coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)   # y -> row index
    target[0, 0, coords[:, 1], coords[:, 0]] = 1.0
    k = int(6 * sigma + 1) | 1
    target = gaussian_blur(target, kernel_size=k, sigma=sigma)
    return target / (target.sum() + 1e-8)


def build_edit_masks(user_t, orig_t, region_mask=None, thresh=0.1, device="cuda"):
    """Split the edit into attract / repulse regions, each (1, 1, H, W).

    repulse = on the OLD path but not the new one  -> push mass away
    attract = on the NEW path but not the old one  -> pull mass in
    """
    user_m = (user_t > user_t.max() * thresh).float()
    orig_m = (orig_t > orig_t.max() * thresh).float()

    repulse_m = orig_m * (1.0 - user_m)
    attract_m = user_m * (1.0 - orig_m)

    if region_mask is not None:
        rm = torch.as_tensor(region_mask, dtype=torch.float32, device=device)[None, None]
        repulse_m = repulse_m * rm
        attract_m = attract_m * rm
    return attract_m, repulse_m


def _traj_to_prob(x0_pred):
    """Map a [-1, 1] trajectory heatmap to a per-sample normalized distribution."""
    p = ((x0_pred + 1.0) * 0.5).clamp(min=0.0)                 # [-1,1] -> [0,1]
    denom = p.flatten(1).sum(dim=1).view(-1, 1, 1, 1) + 1e-8
    return p / denom


def _to_batched(v, B):
    """Accept (C,H,W) or (1,C,H,W) and tile to (B,C,H,W)."""
    if v.dim() == 3:
        v = v.unsqueeze(0)
    return v.repeat(B, 1, 1, 1)


def finetune_online(
    model,
    batch,                 # single scene: (features_dict, target, positions, radii, goal, angles)
    orig_path_xy,          # (M, 2) (x, y) polyline the model produced
    user_path_xy,          # (M, 2) (x, y) polyline the user dragged
    device,
    ddpm,
    lr=2e-4,
    epochs=200,
    batch_repeat=8,        # tile the one scene so each step averages several noise draws
    region_mask=None,      # optional extra spatial confinement of the edit
    sigma=5.0,
    w_diffusion=1.0,
    w_attract=1.0,
    w_repulse=0.5,
    max_t_frac=0.25,       # sample low-noise timesteps so x0_pred is trustworthy
    verbose=True,
):
    features, target, positions, radii, goal, angles = batch
    H, W = target.shape[-2], target.shape[-1]
    B = batch_repeat

    # ---- tile the single scene into a small batch (stabilizes online updates) ----
    #cond = {k: _to_batched(v.to(device), B) for k, v in features.items()}
    #x0_dataset = _to_batched(target.to(device), B)        # diffusion-grounding anchor

    # ---- precompute targets / masks once ----
    user_t = make_path_target(user_path_xy, H, W, device, sigma=sigma)
    orig_t = make_path_target(orig_path_xy, H, W, device, sigma=sigma)

    # ---- tile the single scene into a small batch ----
    cond = {k: _to_batched(v.to(device), B) for k, v in features.items()}
    
    # THE FIX: Anchor the diffusion process to the USER'S path, not the original target
    # THE FIX: Scale the user's path to [-1, 1] before giving it to the DDPM
    user_img = user_t / (user_t.max() + 1e-8)          # Scale up to [0, 1]
    x0_dataset = _to_batched(user_img * 2.0 - 1.0, B)  # Scale to [-1, 1]
    #x0_dataset = _to_batched(user_t, B)

    # ---- precompute targets / masks once ----
    
    # ---- precompute targets / masks once ----
    user_dist = make_distance_penalty(user_path_xy, H, W, device)
    orig_dist = make_distance_penalty(orig_path_xy, H, W, device)

    # Build the repulse penalty:
    # 1. High penalty exactly on the old path, fading out as you move away
    repulse_width = 0.05 
    old_path_penalty = torch.exp(-orig_dist / repulse_width)

    # 2. Only repulse where the user actively diverged from the original path!
    # (We don't want to repulse the shared start/end points)
    divergence_mask = (user_dist > 0.05).float()
    repulse_penalty = old_path_penalty * divergence_mask

    if region_mask is not None:
        rm = torch.as_tensor(region_mask, dtype=torch.float32, device=device)[None, None]
        repulse_penalty = repulse_penalty * rm

    # ---- activate adapters: only LoRA + FiLM train ----
    model.set_finetune(active=True)
    film_dim = model.unet.film_dec3.gamma_proj[0].in_features
    film_cond = torch.zeros(B, film_dim, device=device)   # constant -> FiLM = learned affine

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    max_t = max(1, int(ddpm.timesteps * max_t_frac))
    loss_history = []

    for epoch in tqdm(range(epochs), desc="Online IRL finetune"):
        model.train()
        optimizer.zero_grad()

        t = torch.randint(0, max_t, (B,), device=device).long()
        x_t, noise = ddpm.q_sample(x0_dataset, t)
        noise_pred = model(x_t, t, cond, film_cond=film_cond)
        x0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)

        # 1) diffusion grounding -> output stays a valid path for this scene
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # 2) + 3) attract / repulse directly on the predicted trajectory distribution
        #p = _traj_to_prob(x0_pred)
        #loss_attract = -(user_p * torch.log(p + 1e-8)).sum() / B
        #loss_repulse = (repulse_p * torch.log(p + 1e-8)).sum() / B

        
        # 2) + 3) attract / repulse directly on the predicted trajectory distribution
        p = _traj_to_prob(x0_pred) # Shape: (B, 1, H, W), sums to 1.0 per batch

        # Expected Distance Loss: penalize probability mass that falls far from the target curve
        loss_attract = (p * user_dist).sum() / B
        
        # Repulse Loss: penalize probability mass that falls on the abandoned segment of the old path
        loss_repulse = (p * repulse_penalty).sum() / B

        loss = (w_diffusion * loss_diffusion
                + w_attract * loss_attract
                + w_repulse * loss_repulse)

        loss.backward()
        trainable = [pp for pp in model.parameters() if pp.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        loss_history.append({
            "total": loss.item(),
            "diffusion": loss_diffusion.item(),
            "attract": loss_attract.item(),
            #"repulse": loss_repulse.item(),
        })
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[{epoch:4d}] total={loss.item():.4f}  "
                  f"diff={loss_diffusion.item():.4f}  "
                  f"att={loss_attract.item():.4f}") #  rep={loss_repulse.item():.4f}")

    model.set_finetune(active=False)
    return loss_history
