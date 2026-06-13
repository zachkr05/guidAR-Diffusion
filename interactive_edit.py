"""
Interactive online finetuning demo for the unified trajectory-diffusion model.

Flow:
  1. Sample a scene, run the diffusion model, extract the ORIGINAL path.
  2. Show that path as a draggable B-spline over the obstacles. Drag the
     control points, then close the window.
  3. Finetune the model online (LoRA + FiLM) on that single edit.
  4. Re-sample with the SAME initial noise and overlay
     original vs. user-edit vs. adapted path.

Coordinate convention used throughout the UI / spline layer:
    (x, y) = (col, row)            <- matplotlib + DraggableBSpline + make_path_target
The scene/heatmap arrays are indexed [row, col], and route_through_array /
the dataset goal are (row, col). Conversions are done explicitly at the seams.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("TkAgg")            # interactive backend; needs a display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.graph import route_through_array

from DataGenerator.dataset import CostmapDataset
from MoE.ddpm import DDPM
from MoE.simple_model import SimpleTrajectoryModel
from MoE.spline import generate_clamped_spline, DraggableBSpline
from MoE.finetune import finetune_online


# --------------------------------------------------------------------------- #
#  FiLM-aware sampling
# --------------------------------------------------------------------------- #
# ddpm.sample()/p_sample() call model(x, t, cond) WITHOUT film_cond, which makes
# FiLM fall back to identity. If we finetuned FiLM, we must keep it active at
# inference too, so we wrap the model to inject a fixed film_cond. This reuses
# ddpm's exact reverse process (no duplicated sampler math).
class _FilmWrap(nn.Module):
    def __init__(self, model, film_cond):
        super().__init__()
        self.model = model
        self.film_cond = film_cond

    def forward(self, x_t, t, conditioning):
        fc = self.film_cond[: x_t.shape[0]]
        return self.model(x_t, t, conditioning, film_cond=fc)


@torch.no_grad()
def sample_heatmap(model, ddpm, cond, H, W, device, film_dim, seed=0):
    """Deterministic single-scene sample -> heatmap in [0, 1]."""
    torch.manual_seed(seed)        # fix initial noise so before/after are comparable
    model.eval()
    film = torch.zeros(1, film_dim, device=device)   # zeros == identity pre-finetune
    wrapped = _FilmWrap(model, film)
    x0 = ddpm.sample(wrapped, cond, shape=(1, 1, H, W))    # [-1, 1]
    return ((x0[0, 0] + 1.0) * 0.5).clamp(0, 1).cpu().numpy()


def heatmap_to_path_xy(heat01, start_rc, goal_rc, floor=1e-3):
    """Route through high-probability cells; return path as (x, y) = (col, row)."""
    cost = (1.0 - heat01).astype(np.float64) + floor       # low cost where the path is
    path_rc, _ = route_through_array(
        cost, start_rc, goal_rc, fully_connected=True
    )
    path_rc = np.asarray(path_rc)                          # (M, 2) = (row, col)
    return path_rc[:, ::-1].copy()                         # -> (col, row) = (x, y)


def draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W):
    classes = list(positions.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(classes))))
    for ci, cls in enumerate(classes):
        for (r, c), rad in zip(positions[cls], radii[cls]):
            ax.add_patch(patches.Circle((c, r), float(rad) + 0.5,
                                        color=colors[ci], alpha=0.35))
            ax.add_patch(patches.Circle((c, r), float(rad) + 0.5,
                                        color=colors[ci], fill=False, lw=2))
        ax.plot([], [], "o", color=colors[ci], label=cls)
    ax.plot(goal_rc[1], goal_rc[0], "g*", ms=18, label="goal")
    ax.plot(robot_rc[1], robot_rc[0], "cs", ms=10, label="start")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)              # image convention: row increases downward
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obstacle_classes", nargs="+", default=["chair", "table", "bomb"])
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_ctrl_pts", type=int, default=10)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.H, args.W
    N = len(args.obstacle_classes)

    # ---- model (must match training-time construction) ----
    in_channels_per_class = 4 + 4 * (N - 1) + 1
    model = SimpleTrajectoryModel(
        obstacle_classes=args.obstacle_classes,
        in_channels_per_class=in_channels_per_class,
        base_channels=args.base_channels,
        time_dim=args.base_channels * 4,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    film_dim = model.unet.film_dec3.gamma_proj[0].in_features

    # ---- one scene ----
    ds = CostmapDataset(n_samples=1, H=H, W=W)
    ds.obstacle_classes = args.obstacle_classes
    features, target, positions, radii, goal, angles = ds[0]
    cond = {k: v.unsqueeze(0).to(device) for k, v in features.items()}   # (1,C,H,W)
    goal_rc = (int(goal[0]), int(goal[1]))     # dataset goal is (row, col)
    robot_rc = (H - 1, W - 1)                  # Costmap default start corner

    # ---- ORIGINAL path from the current model ----
    heat = sample_heatmap(model, ddpm, cond, H, W, device, film_dim, seed=args.seed)
    orig_path_xy = heatmap_to_path_xy(heat, robot_rc, goal_rc)

    # ---- fit a clamped B-spline so the user has draggable control points ----
    xs = orig_path_xy[:, 0].astype(float)
    ys = orig_path_xy[:, 1].astype(float)
    ncp = max(args.degree + 1, min(args.num_ctrl_pts, len(xs) - 1))
    _, _, P, U = generate_clamped_spline(xs, ys, k=args.degree, num_ctrl_pts=ncp)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(heat, cmap="hot", origin="upper", alpha=0.35, extent=[0, W, H, 0])
    draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W)
    drag = DraggableBSpline(ax, U, P, args.degree)
    ax.set_title("Drag the control points to edit the path, then CLOSE the window")
    plt.show()                                  # blocks until the window is closed

    # ---- captured USER path (curve samples reflect final control points) ----
    user_path_xy = np.column_stack([drag.x, drag.y])

    # ---- finetune online on this single edit ----
    batch = (features, target, positions, radii, goal, angles)
    finetune_online(
        model, batch, orig_path_xy, user_path_xy,
        device=device, ddpm=ddpm, lr=args.lr, epochs=args.epochs,
    )

    # ---- ADAPTED path (same seed -> only the weights changed) ----
    heat2 = sample_heatmap(model, ddpm, cond, H, W, device, film_dim, seed=args.seed)
    adapted_path_xy = heatmap_to_path_xy(heat2, robot_rc, goal_rc)

    # ---- compare ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(heat2, cmap="hot", origin="upper", alpha=0.35, extent=[0, W, H, 0])
    draw_scene(ax, positions, radii, goal_rc, robot_rc, H, W)
    ax.plot(orig_path_xy[:, 0], orig_path_xy[:, 1], "w--", lw=2, label="original")
    ax.plot(user_path_xy[:, 0], user_path_xy[:, 1], "c-", lw=2, label="user edit")
    ax.plot(adapted_path_xy[:, 0], adapted_path_xy[:, 1],
            color="lime", lw=2.5, label="after finetune")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Before vs. user edit vs. after online finetuning")
    plt.savefig("online_finetune_result.png", dpi=150)
    print("Saved online_finetune_result.png")
    plt.show()


if __name__ == "__main__":
    main()
