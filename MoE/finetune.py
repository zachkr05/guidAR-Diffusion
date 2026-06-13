"""
Bulletproof online finetuning loss for 2D trajectory DDPM.

Key properties:
  - Keeps architecture unchanged: 2D CNN UNet, image-to-image DDPM.
  - Keeps route_through_array extraction unchanged.
  - Trains only LoRA + FiLM through model.set_finetune(True).
  - Anchors DDPM to the user's edited path, not the old target.
  - Prevents zero-mass collapse with softmax-over-pixels.
  - Prevents fat highways with centerline/off-ridge/background penalties.
  - Applies spatial losses only at low/mid timesteps where x0_pred is meaningful.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import distance_transform_edt


# -------------------------------------------------------------------------
# Geometry utilities
# -------------------------------------------------------------------------

def _as_numpy_xy(path_xy):
    pts = np.asarray(path_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("path_xy must have shape (N, 2), with columns (x=col, y=row).")
    if len(pts) < 2:
        raise ValueError("path_xy must contain at least two points.")
    return pts


def _lock_start_goal_if_needed(path_xy, H, W, goal_rc, lock=True):
    """
    route_through_array still uses fixed start/goal.
    If the user spline accidentally moves endpoints, keep the routing endpoints fixed.
    """
    pts = _as_numpy_xy(path_xy)

    if not lock:
        return pts

    start_xy = np.array([W - 1, H - 1], dtype=np.float32)
    goal_xy = np.array([float(goal_rc[1]), float(goal_rc[0])], dtype=np.float32)

    out = pts
    if np.linalg.norm(out[0] - start_xy) > 1.5:
        out = np.vstack([start_xy[None], out])
    else:
        out[0] = start_xy

    if np.linalg.norm(out[-1] - goal_xy) > 1.5:
        out = np.vstack([out, goal_xy[None]])
    else:
        out[-1] = goal_xy

    return out.astype(np.float32)


def _densify_polyline_xy(path_xy, step_px=0.35):
    """
    Dense anti-aliasing support for a continuous user spline.
    Returns float xy points, not just rounded pixels.
    """
    pts = _as_numpy_xy(path_xy)
    chunks = []

    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        n = max(2, int(math.ceil(length / max(step_px, 1e-6))) + 1)
        ts = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        chunks.append(a[None, :] + ts[:, None] * seg[None, :])

    chunks.append(pts[-1:])

    dense = np.concatenate(chunks, axis=0)
    return dense.astype(np.float32)


def _subsample_xy(xy, max_points):
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) <= max_points:
        return xy
    idx = np.linspace(0, len(xy) - 1, max_points).round().astype(np.int64)
    return xy[idx]


def _xy_to_grid(xy, H, W, device):
    """
    Convert xy=(col,row) points to grid_sample coordinates.
    Output shape: (1, N, 1, 2)
    """
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) == 0:
        return None

    x = np.clip(xy[:, 0], 0, W - 1)
    y = np.clip(xy[:, 1], 0, H - 1)

    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0

    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    return torch.from_numpy(grid).to(device).view(1, -1, 1, 2)


def _sample_at_grid(img_b1hw, grid_1n12):
    """
    img_b1hw: (B,1,H,W)
    grid_1n12: (1,N,1,2)
    Returns: (B,N)
    """
    if grid_1n12 is None:
        return None

    B = img_b1hw.shape[0]
    grid = grid_1n12.expand(B, -1, -1, -1)
    vals = F.grid_sample(
        img_b1hw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return vals[:, 0, :, 0]


def _normal_offset_samples(dense_xy, H, W, offset_px=2.25, max_points=512):
    """
    Build samples a few pixels to either side of the path.
    Penalizing these prevents the route_through_array cost image from becoming a fat highway.
    """
    xy = _subsample_xy(dense_xy, max_points=max_points)

    if len(xy) < 3:
        return xy

    tangent = np.gradient(xy, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    tangent = tangent / norm

    # normal to xy tangent
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    left = xy + offset_px * normal
    right = xy - offset_px * normal

    off = np.concatenate([left, right], axis=0)
    off[:, 0] = np.clip(off[:, 0], 0, W - 1)
    off[:, 1] = np.clip(off[:, 1], 0, H - 1)
    return off.astype(np.float32)


# -------------------------------------------------------------------------
# Heatmap / field construction
# -------------------------------------------------------------------------

def _build_path_fields(
    path_xy,
    H,
    W,
    device,
    goal_rc=None,
    lock_route_endpoints=True,
    densify_step_px=0.35,
    sigma_data=0.90,
    sigma_ce=1.25,
    center_radius_px=0.75,
    corridor_radius_px=3.00,
    negative_radius_px=4.50,
    dist_cap_px=20.0,
    max_pin_points=768,
    max_offset_points=768,
):
    """
    Builds:
      target01:      max-normalized image target in [0,1], not sum-normalized.
      target_pm1:    DDPM data target in [-1,1].
      q:             sum-normalized spatial distribution for CE only.
      dist_norm:     clipped normalized distance-to-user-path.
      masks:         center/corridor/background masks.
      pin_grid:      exact centerline samples.
      off_grid:      normal-offset samples for anti-fat-highway loss.
    """
    if goal_rc is None:
        goal_rc = (0, 0)

    pts = _lock_start_goal_if_needed(
        path_xy=path_xy,
        H=H,
        W=W,
        goal_rc=goal_rc,
        lock=lock_route_endpoints,
    )
    dense = _densify_polyline_xy(pts, step_px=densify_step_px)

    mask = np.zeros((H, W), dtype=bool)
    cols = np.rint(dense[:, 0]).astype(np.int64)
    rows = np.rint(dense[:, 1]).astype(np.int64)
    cols = np.clip(cols, 0, W - 1)
    rows = np.clip(rows, 0, H - 1)
    mask[rows, cols] = True

    if not np.any(mask):
        raise RuntimeError("Failed to rasterize user path into any pixels.")

    # Euclidean pixel distance to centerline.
    dist_px = distance_transform_edt(~mask).astype(np.float32)

    # DDPM image target: max-normalized ridge, not probability-normalized.
    target01 = np.exp(-(dist_px ** 2) / (2.0 * sigma_data ** 2)).astype(np.float32)
    target01[dist_px > max(corridor_radius_px, 4.0 * sigma_data)] = 0.0
    target01 = target01 / (target01.max() + 1e-8)

    # CE target: probability distribution. This is the only sum-normalized target.
    q = np.exp(-(dist_px ** 2) / (2.0 * sigma_ce ** 2)).astype(np.float32)
    q[dist_px > 4.0 * sigma_ce] = 0.0
    q = q / (q.sum() + 1e-8)

    dist_norm = np.minimum(dist_px, dist_cap_px) / max(dist_cap_px, 1e-6)

    center_mask = (dist_px <= center_radius_px).astype(np.float32)
    corridor_mask = (dist_px <= corridor_radius_px).astype(np.float32)
    hard_negative_mask = (dist_px >= negative_radius_px).astype(np.float32)

    pin_xy = _subsample_xy(dense, max_points=max_pin_points)
    off_xy = _normal_offset_samples(
        dense_xy=dense,
        H=H,
        W=W,
        offset_px=max(2.0, corridor_radius_px * 0.75),
        max_points=max_offset_points,
    )

    def t01(a):
        return torch.from_numpy(a).float().to(device).view(1, 1, H, W)

    target01_t = t01(target01)
    fields = {
        "target01": target01_t,
        "target_pm1": target01_t * 2.0 - 1.0,
        "q": t01(q),
        "dist_norm": t01(dist_norm),
        "center_mask": t01(center_mask),
        "corridor_mask": t01(corridor_mask),
        "hard_negative_mask": t01(hard_negative_mask),
        "target_mass": torch.tensor(float(target01.sum()), device=device),
        "pin_grid": _xy_to_grid(pin_xy, H, W, device),
        "off_grid": _xy_to_grid(off_xy, H, W, device),
        "dense_xy": dense,
        "dist_px_np": dist_px,
    }
    return fields


def _build_repulse_field(
    user_fields,
    orig_path_xy,
    H,
    W,
    device,
    goal_rc,
    lock_route_endpoints=True,
    sigma_repulse=1.25,
    shared_radius_px=2.25,
):
    """
    Penalizes abandoned old path only where it is not close to the new path.
    Shared start/end and unchanged segments are not repulsed.
    """
    orig_fields = _build_path_fields(
        orig_path_xy,
        H=H,
        W=W,
        device=device,
        goal_rc=goal_rc,
        lock_route_endpoints=lock_route_endpoints,
        sigma_data=sigma_repulse,
        sigma_ce=sigma_repulse,
    )

    old_d = orig_fields["dist_px_np"]
    user_d = user_fields["dist_px_np"]

    repulse = np.exp(-(old_d ** 2) / (2.0 * sigma_repulse ** 2)).astype(np.float32)
    abandoned = (user_d > shared_radius_px).astype(np.float32)
    repulse = repulse * abandoned

    if repulse.max() > 0:
        repulse = repulse / (repulse.max() + 1e-8)

    return torch.from_numpy(repulse).float().to(device).view(1, 1, H, W)


# -------------------------------------------------------------------------
# Loss helpers
# -------------------------------------------------------------------------

def _to_batched(v, B, device):
    if not torch.is_tensor(v):
        v = torch.as_tensor(v)
    v = v.to(device).float()
    if v.dim() == 3:
        v = v.unsqueeze(0)
    if v.dim() != 4:
        raise ValueError(f"Expected tensor with shape (C,H,W) or (1,C,H,W), got {tuple(v.shape)}")
    return v.repeat(B, 1, 1, 1)


def _repeat_field(x, B):
    return x.repeat(B, 1, 1, 1)


def _weighted_mean(loss_map, weight_map, eps=1e-8):
    return (loss_map * weight_map).sum() / weight_map.sum().clamp_min(eps)


def _spatial_softmax_and_logp(x0_pred, tau=0.08):
    """
    x0_pred: (B,1,H,W), unclamped.
    Returns p and logp, both (B,1,H,W).
    This is the main zero-mass fix.
    """
    B = x0_pred.shape[0]
    logits = x0_pred.flatten(1) / max(tau, 1e-6)
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    return p.view_as(x0_pred), logp.view_as(x0_pred)


def _sobel_xy(x):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 8.0

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.cat([gx, gy], dim=1)


def _snapshot_trainable(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _adapter_delta_l2(model, snapshot):
    vals = []
    for name, p in model.named_parameters():
        if p.requires_grad and name in snapshot:
            vals.append((p - snapshot[name]).pow(2).mean())

    if not vals:
        return torch.zeros((), device=next(model.parameters()).device)

    return torch.stack(vals).mean()


# -------------------------------------------------------------------------
# Main online finetune
# -------------------------------------------------------------------------

def finetune_online(
    model,
    batch,
    orig_path_xy,
    user_path_xy,
    device,
    ddpm,
    lr=3e-4,
    epochs=200,
    batch_repeat=16,

    # timestep policy
    spatial_t_frac=0.28,
    spatial_batch_frac=0.75,

    # target geometry
    sigma_data=0.90,
    sigma_ce=1.25,
    center_radius_px=0.75,
    corridor_radius_px=3.00,
    negative_radius_px=4.50,
    lock_route_endpoints=True,

    # loss temperatures / margins
    softmax_tau=0.08,
    bce_logit_scale=8.0,
    center_min_heat=0.90,
    off_max_heat=0.18,
    bg_max_heat=0.08,

    # weights
    w_eps=1.0,
    w_x0=18.0,
    w_bce=8.0,
    w_ce=1.0,
    w_dist=10.0,
    w_repulse=12.0,
    w_old_heat=8.0,
    w_bg=8.0,
    w_center=80.0,
    w_off=35.0,
    w_grad=2.0,
    w_mass=1.5,
    w_delta=1e-4,

    # optimization
    grad_clip=1.0,
    warmup_epochs=20,
    verbose=True,
):
    """
    Replace your current MoE/finetune.py finetune_online with this.

    batch is:
      (features, target, positions, radii, goal, angles)

    Coordinate convention:
      user_path_xy and orig_path_xy are (x,y) = (col,row), matching your UI.
      goal from dataset is (row,col).
    """
    features, target, positions, radii, goal, angles = batch

    H = int(target.shape[-2])
    W = int(target.shape[-1])
    B = int(batch_repeat)

    if B < 2:
        raise ValueError("batch_repeat should be at least 2; use 8-32 for stable online updates.")

    if torch.is_tensor(goal):
        goal_np = goal.detach().cpu().numpy()
    else:
        goal_np = np.asarray(goal)
    goal_rc = (int(goal_np[0]), int(goal_np[1]))

    # ------------------------------------------------------------------
    # Build exact user target fields.
    # target01 is max-normalized [0,1].
    # target_pm1 is DDPM data target [-1,1].
    # q is sum-normalized only for softmax CE.
    # ------------------------------------------------------------------
    user_fields = _build_path_fields(
        path_xy=user_path_xy,
        H=H,
        W=W,
        device=device,
        goal_rc=goal_rc,
        lock_route_endpoints=lock_route_endpoints,
        sigma_data=sigma_data,
        sigma_ce=sigma_ce,
        center_radius_px=center_radius_px,
        corridor_radius_px=corridor_radius_px,
        negative_radius_px=negative_radius_px,
    )

    repulse_field = _build_repulse_field(
        user_fields=user_fields,
        orig_path_xy=orig_path_xy,
        H=H,
        W=W,
        device=device,
        goal_rc=goal_rc,
        lock_route_endpoints=lock_route_endpoints,
    )

    # ------------------------------------------------------------------
    # Tile one scene into a small batch of independent noise draws.
    # ------------------------------------------------------------------
    cond = {k: _to_batched(v, B, device) for k, v in features.items()}

    x0_user = _repeat_field(user_fields["target_pm1"], B)
    target01_all = _repeat_field(user_fields["target01"], B)
    q_all = _repeat_field(user_fields["q"], B)
    dist_all = _repeat_field(user_fields["dist_norm"], B)
    hard_neg_all = _repeat_field(user_fields["hard_negative_mask"], B)
    center_mask_all = _repeat_field(user_fields["center_mask"], B)
    corridor_all = _repeat_field(user_fields["corridor_mask"], B)
    repulse_all = _repeat_field(repulse_field, B)

    # ------------------------------------------------------------------
    # Activate adapters only.
    # Your sampling wrapper uses zero film_cond, so train with zero film_cond.
    # ------------------------------------------------------------------
    model.set_finetune(active=True)
    model.train()

    if not hasattr(model, "unet") or not hasattr(model.unet, "film_dec3"):
        raise AttributeError("Expected model.unet.film_dec3, matching your SimpleTrajectoryModel.")

    film_dim = model.unet.film_dec3.gamma_proj[0].in_features
    film_cond = torch.zeros(B, film_dim, device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters. Check model.set_finetune(active=True).")

    snapshot = _snapshot_trainable(model)

    optimizer = AdamW(trainable, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.10)

    max_t_spatial = max(1, int(ddpm.timesteps * spatial_t_frac))
    n_spatial = max(1, int(round(B * spatial_batch_frac)))
    n_full = B - n_spatial

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Mixed timestep batch:
        #   first n_spatial samples: low/mid t, used for x0 spatial losses
        #   remaining samples: full t, used only for DDPM noise preservation
        t_sp = torch.randint(
            low=0,
            high=max_t_spatial,
            size=(n_spatial,),
            device=device,
            dtype=torch.long,
        )

        if n_full > 0:
            t_full = torch.randint(
                low=0,
                high=ddpm.timesteps,
                size=(n_full,),
                device=device,
                dtype=torch.long,
            )
            t = torch.cat([t_sp, t_full], dim=0)
        else:
            t = t_sp

        x_t, noise = ddpm.q_sample(x0_user, t)
        noise_pred = model(x_t, t, cond, film_cond=film_cond)

        # Standard DDPM loss over the whole mixed-t batch.
        loss_eps = F.mse_loss(noise_pred, noise)

        # x0 prediction. Do NOT clamp during training.
        x0_pred = ddpm.predict_start_from_noise(x_t, t, noise_pred)

        # Spatial losses only on stable low/mid timesteps.
        x0_sp = x0_pred[:n_spatial]
        target_pm1_sp = x0_user[:n_spatial]
        target01_sp = target01_all[:n_spatial]
        q_sp = q_all[:n_spatial]
        dist_sp = dist_all[:n_spatial]
        hard_neg_sp = hard_neg_all[:n_spatial]
        center_mask_sp = center_mask_all[:n_spatial]
        corridor_sp = corridor_all[:n_spatial]
        repulse_sp = repulse_all[:n_spatial]

        # Inference heatmap uses linear mapping (x0+1)/2, so train that too.
        heat01 = 0.5 * (x0_sp + 1.0)

        # Strong positive weighting along the ridge, plus hard negatives away from it.
        pos_count = center_mask_sp.sum().clamp_min(1.0)
        neg_count = hard_neg_sp.sum().clamp_min(1.0)
        auto_pos_weight = (neg_count / pos_count).detach().clamp(25.0, 600.0)

        ridge_weight = 1.0 + auto_pos_weight * target01_sp + 8.0 * hard_neg_sp

        # 1) Direct x0 image-space loss in DDPM data range [-1,1].
        loss_x0_map = F.smooth_l1_loss(
            x0_sp,
            target_pm1_sp,
            reduction="none",
            beta=0.15,
        )
        loss_x0 = _weighted_mean(loss_x0_map, ridge_weight)

        # 2) Amplitude loss. All-black is heavily penalized on positive ridge pixels.
        bce_logits = bce_logit_scale * x0_sp
        loss_bce_map = F.binary_cross_entropy_with_logits(
            bce_logits,
            target01_sp,
            reduction="none",
        )
        loss_bce = _weighted_mean(loss_bce_map, ridge_weight)

        # 3) Softmax CE. This is the zero-mass-collapse fix.
        p_sp, logp_sp = _spatial_softmax_and_logp(x0_sp, tau=softmax_tau)
        loss_ce = -(q_sp * logp_sp).flatten(1).sum(dim=1).mean()

        # 4) Expected normalized distance to the user path.
        loss_dist = (p_sp * dist_sp).flatten(1).sum(dim=1).mean()

        # 5) Repulse probability mass from abandoned old path.
        loss_repulse = (p_sp * repulse_sp).flatten(1).sum(dim=1).mean()

        # 6) Direct old-path heat penalty. This catches diffuse heat that softmax may understate.
        repulse_den = repulse_sp.sum().clamp_min(1.0)
        loss_old_heat = (F.relu(heat01) * repulse_sp).sum() / repulse_den

        # 7) Background hinge: do not allow a fat highway outside the corridor.
        bg_mask = hard_neg_sp
        bg_den = bg_mask.sum().clamp_min(1.0)
        loss_bg = (F.relu(heat01 - bg_max_heat).pow(2) * bg_mask).sum() / bg_den

        # 8) Pin exact centerline high.
        center_vals = _sample_at_grid(heat01, user_fields["pin_grid"])
        if center_vals is None:
            loss_center = torch.zeros((), device=device)
        else:
            loss_center = F.relu(center_min_heat - center_vals).pow(2).mean()

        # 9) Force normal offsets low. This is the anti-fat-highway term.
        off_vals = _sample_at_grid(heat01, user_fields["off_grid"])
        if off_vals is None:
            loss_off = torch.zeros((), device=device)
        else:
            loss_off = F.relu(off_vals - off_max_heat).pow(2).mean()

        # 10) Gradient/ridge-shape matching.
        grad_pred = _sobel_xy(heat01)
        grad_tgt = _sobel_xy(target01_sp)
        grad_weight = 1.0 + 3.0 * corridor_sp
        loss_grad_map = F.smooth_l1_loss(
            grad_pred,
            grad_tgt,
            reduction="none",
            beta=0.05,
        )
        loss_grad = _weighted_mean(loss_grad_map, grad_weight.repeat(1, 2, 1, 1))

        # 11) Total positive heat mass. Prevents both all-black and giant blobs.
        pred_mass = torch.sigmoid(bce_logits).flatten(1).sum(dim=1)
        target_mass = user_fields["target_mass"].clamp_min(1.0)
        loss_mass = ((pred_mass - target_mass) / target_mass).pow(2).mean()

        # 12) Keep LoRA/FiLM delta small so base generation quality is not destroyed.
        loss_delta = _adapter_delta_l2(model, snapshot)

        # Ramp spatial losses in gently but quickly.
        ramp = min(1.0, float(epoch + 1) / float(max(1, warmup_epochs)))

        loss_spatial = (
            w_x0 * loss_x0
            + w_bce * loss_bce
            + w_ce * loss_ce
            + w_dist * loss_dist
            + w_repulse * loss_repulse
            + w_old_heat * loss_old_heat
            + w_bg * loss_bg
            + w_center * loss_center
            + w_off * loss_off
            + w_grad * loss_grad
            + w_mass * loss_mass
        )

        loss = (
            w_eps * loss_eps
            + ramp * loss_spatial
            + w_delta * loss_delta
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        row = {
            "epoch": epoch,
            "total": float(loss.detach().cpu()),
            "eps": float(loss_eps.detach().cpu()),
            "x0": float(loss_x0.detach().cpu()),
            "bce": float(loss_bce.detach().cpu()),
            "ce": float(loss_ce.detach().cpu()),
            "dist": float(loss_dist.detach().cpu()),
            "repulse": float(loss_repulse.detach().cpu()),
            "old_heat": float(loss_old_heat.detach().cpu()),
            "bg": float(loss_bg.detach().cpu()),
            "center": float(loss_center.detach().cpu()),
            "off": float(loss_off.detach().cpu()),
            "grad": float(loss_grad.detach().cpu()),
            "mass": float(loss_mass.detach().cpu()),
            "delta": float(loss_delta.detach().cpu()),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        loss_history.append(row)

        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1):
            print(
                f"[{epoch + 1:04d}/{epochs}] "
                f"total={row['total']:.4f} "
                f"eps={row['eps']:.4f} "
                f"x0={row['x0']:.4f} "
                f"ce={row['ce']:.4f} "
                f"dist={row['dist']:.4f} "
                f"rep={row['repulse']:.4f} "
                f"center={row['center']:.4f} "
                f"off={row['off']:.4f} "
                f"bg={row['bg']:.4f}"
            )

    # This only changes requires_grad flags; it does not remove learned LoRA/FiLM behavior.
    model.set_finetune(active=False)

    return loss_history
