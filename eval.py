
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataGenerator.dataGenerator import CostmapDataset
from MoE.ddpm import DDPM
from MoE.UNet import LightweightUNet
from train import ExpertEnsemble
from utils import *
from utils.planner import SoftGridPlanner
from skimage.graph import route_through_array

from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# =========================================================================
# Obstacle marker conventions (matching utils.py get_user_adjustments)
# =========================================================================
OBSTACLE_COLORS = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}
GOAL_COLOR = 'olive'
OBSTACLE_MARKER = 'o'
GOAL_MARKER = '*'



def visualize_simple(before_fused, after_fused, goal, positions, radii, orientation,
                     obstacle_classes, save_path="trajectory_comparison.png"):
    before_np = before_fused[0, 0].detach().cpu().numpy()
    after_np = after_fused[0, 0].detach().cpu().numpy()
    delta = after_np - before_np
    goal_np = goal[0].cpu().numpy() if isinstance(goal, torch.Tensor) else np.array(goal[0])

    def get_path(costmap_np, g):
        mn, mx = costmap_np.min(), costmap_np.max()
        normed = (costmap_np - mn) / (mx - mn + 1e-8) + 1e-8
        gy = int(np.clip(g[0], 0, costmap_np.shape[0]-1))
        gx = int(np.clip(g[1], 0, costmap_np.shape[1]-1))
        path_pts, _ = route_through_array(normed, [0,0], [gy,gx], fully_connected=True, geometric=True)
        path_arr = np.array(path_pts)
        return path_arr[:, 1], path_arr[:, 0]

    def draw_all(ax, positions, radii, orientation, goal_np, obstacle_classes):
        draw_obstacles(ax, positions, radii, goal_np, obstacle_classes)
        colors_map = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}
        for cls in obstacle_classes:
            for i, pos in enumerate(positions[0][cls]):
                angle = orientation[cls][i].item()
                arrow_len = 5
                dx = arrow_len * np.cos(angle)
                dy = arrow_len * np.sin(angle)
                ax.arrow(pos[1], pos[0], dx, dy,
                         head_width=1.5, head_length=1.0,
                         fc=colors_map.get(cls, 'white'),
                         ec=colors_map.get(cls, 'white'), alpha=0.8)

    old_x, old_y = get_path(before_np, goal_np)
    new_x, new_y = get_path(after_np, goal_np)
    vmin = min(before_np.min(), after_np.min())
    vmax = max(before_np.max(), after_np.max())
    dmax = max(abs(delta).max(), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Before
    ax = axes[0]
    im = ax.imshow(before_np, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax.plot(old_x, old_y, 'r-', linewidth=2)
    ax.plot(0, 0, 'go', markersize=8)
    draw_all(ax, positions, radii, orientation, goal_np, obstacle_classes)
    ax.set_title("Before Fine-tuning", fontsize=13, fontweight='bold')
    ax.legend(handles=[Line2D([],[],color='r',lw=2,label='Original Path'),
                       Line2D([],[],color='w',marker='o',markerfacecolor='g',markersize=8,label='Start')]
              + make_obstacle_legend(obstacle_classes), loc='upper left', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: After
    ax = axes[1]
    im = ax.imshow(after_np, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax.plot(new_x, new_y, 'b-', linewidth=2)
    ax.plot(0, 0, 'go', markersize=8)
    draw_all(ax, positions, radii, orientation, goal_np, obstacle_classes)
    ax.set_title("After Fine-tuning", fontsize=13, fontweight='bold')
    ax.legend(handles=[Line2D([],[],color='b',lw=2,label='New Path'),
                       Line2D([],[],color='w',marker='o',markerfacecolor='g',markersize=8,label='Start')]
              + make_obstacle_legend(obstacle_classes), loc='upper left', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Delta
    ax = axes[2]
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-dmax, vmax=dmax, origin="lower")
    ax.plot(old_x, old_y, 'r-', linewidth=2)
    ax.plot(new_x, new_y, 'b-', linewidth=2)
    draw_all(ax, positions, radii, orientation, goal_np, obstacle_classes)
    ax.set_title("Costmap Delta\nRed=Before | Blue=After", fontsize=11, fontweight='bold')
    ax.legend(handles=[Line2D([],[],color='r',lw=2,label='Old Path'),
                       Line2D([],[],color='b',lw=2,label='New Path')]
              + make_obstacle_legend(obstacle_classes), loc='upper left', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


def draw_obstacles(ax, positions, radii, goal, obstacle_classes, marker_size=8,
                   show_radii=True, radii_alpha=0.25):
    """
    Draw obstacle markers and optional influence circles on an axis.
    Uses the same color/marker conventions as utils.get_user_adjustments.

    Args:
        ax: matplotlib axis
        positions: dict or list-of-dict  {cls: [(r,c), ...]}
        radii: dict or list-of-dict      {cls: [rad, ...]}
        goal: (2,) array [y, x]
        obstacle_classes: list of class names
        show_radii: if True, draw translucent circles showing obstacle radius
    """
    pos_dict = positions[0] if isinstance(positions, list) else positions
    rad_dict = radii[0] if isinstance(radii, list) else radii

    for cls in obstacle_classes:
        color = OBSTACLE_COLORS.get(cls, 'white')
        obs_list = pos_dict.get(cls, [])
        rad_list = rad_dict.get(cls, [])

        for idx, pos in enumerate(obs_list):
            r, c = pos[0], pos[1]
            # Plot marker (x=col, y=row to match imshow)
            ax.plot(c, r, color=color, marker=OBSTACLE_MARKER,
                    markersize=marker_size, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=5)

            # Draw influence radius circle
            if show_radii and idx < len(rad_list):
                radius = rad_list[idx]
                circle = Circle((c, r), radius * 3,  # visual scaling
                                edgecolor=color, facecolor=color,
                                alpha=radii_alpha, linewidth=1.2, zorder=4)
                ax.add_patch(circle)

    # Goal
    g = goal[0] if isinstance(goal, (list, np.ndarray)) and np.ndim(goal) > 1 else goal
    if isinstance(g, torch.Tensor):
        g = g.cpu().numpy()
    ax.plot(g[1], g[0], color=GOAL_COLOR, marker=GOAL_MARKER,
            markersize=14, markeredgecolor='white', markeredgewidth=1, zorder=6)


def make_obstacle_legend(obstacle_classes):
    """Create custom legend handles matching the marker conventions."""
    handles = []
    for cls in obstacle_classes:
        color = OBSTACLE_COLORS.get(cls, 'white')
        handles.append(Line2D([0], [0], marker=OBSTACLE_MARKER, color='w',
                              markerfacecolor=color, markeredgecolor='white',
                              markersize=8, label=cls.capitalize()))
    handles.append(Line2D([0], [0], marker=GOAL_MARKER, color='w',
                          markerfacecolor=GOAL_COLOR, markeredgecolor='white',
                          markersize=12, label='Goal'))
    return handles


def compute_spline_path(costmap_np, goal, k=3, num_ctrl_pts=10):
    """
    Compute a shortest path through a costmap and fit a B-spline to it.

    Returns:
        path_xy: (N,2) raw path [x, y]
        spline_xy: (M,2) smooth spline [x, y]
        P: control points
        U: knot vector
    """
    H, W = costmap_np.shape
    # Normalize for route_through_array
    mn, mx = costmap_np.min(), costmap_np.max()
    normed = (costmap_np - mn) / (mx - mn + 1e-8) + 1e-8

    goal_y = int(np.clip(goal[0], 0, H - 1))
    goal_x = int(np.clip(goal[1], 0, W - 1))

    path_result = route_through_array(normed, [0, 0], [goal_y, goal_x],
                                       fully_connected=True, geometric=True)
    path_array = np.array(path_result[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]
    path_xy = np.column_stack([x_np, y_np])

    x_s, y_s, P, U = generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)
    spline_xy = np.column_stack([x_s, y_s])

    return path_xy, spline_xy, P, U


# =========================================================================
# Core evaluation helpers
# =========================================================================

def get_user_input(batch, model, device, ddpm, obstacle_classes):

    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal, orientations = batch

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            diffused_cm[cls].append(generated)

    fused_costmap = fuse_costmaps(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal, orientations)

    return orig_path, user_path, diffused_cm


def generate_fused(model, batch, obstacle_classes, device, ddpm):
    """Generate costmaps from each expert and fuse them."""
    features, targets, positions, radii, goal, _ = batch
    diffused_cm = {cls: [] for cls in obstacle_classes}
    per_class_maps = {}
    with torch.no_grad():
        for cls in obstacle_classes:
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            generated = ddpm.sample(model.experts[cls], cond, shape=gt.shape)
            diffused_cm[cls].append(generated)
            per_class_maps[cls] = generated.cpu().numpy().squeeze()
    fused = fuse_costmaps(diffused_cm)
    return fused, per_class_maps


# =========================================================================
# Visualization
# =========================================================================

def visualize_results(
    model,
    train_batch,
    eval_batch,
    orig_path,
    user_path,
    obstacle_classes,
    device,
    ddpm,
    before_fused_train,
    before_fused_eval,
    before_per_class_train,
    before_per_class_eval,
    planner,
    save_path="finetuning_results.png",
):
    """
    Comprehensive visualization of before/after finetuning.

    Layout (5 rows × 4 cols):
        Row 0: Training scene — Before | After | Delta | Visitation overlay
        Row 1: Eval scene     — Before | After | Delta | Visitation + spline
        Row 2: Per-class BEFORE (train) — one heatmap per class + fused
        Row 3: Per-class AFTER  (train) — one heatmap per class + fused
        Row 4: Per-class delta  (train) — one delta per class + fused delta
    """
    model.eval()

    _, _, train_positions, train_radii, train_goal = train_batch
    _, _, eval_positions, eval_radii, eval_goal = eval_batch

    # Numpy-ify goals
    train_goal_np = train_goal[0].cpu().numpy() if isinstance(train_goal, torch.Tensor) else np.array(train_goal[0]) if isinstance(train_goal, list) else train_goal
    eval_goal_np = eval_goal[0].cpu().numpy() if isinstance(eval_goal, torch.Tensor) else np.array(eval_goal[0]) if isinstance(eval_goal, list) else eval_goal

    # --- Generate AFTER finetuning ---
    after_fused_train, after_per_class_train = generate_fused(
        model, train_batch, obstacle_classes, device, ddpm)
    after_fused_eval, after_per_class_eval = generate_fused(
        model, eval_batch, obstacle_classes, device, ddpm)

    # --- Plan paths ---
    def plan_path(fused_costmap, goal_tensor):
        cost_map = F.softplus(fused_costmap) + 0.1
        with torch.no_grad():
            visitation = planner(cost_map.to(device), goal_tensor.to(device))
        return visitation.cpu().numpy()[0, 0]

    before_vis_train = plan_path(before_fused_train, train_batch[4])
    after_vis_train = plan_path(after_fused_train, train_batch[4])
    before_vis_eval = plan_path(before_fused_eval, eval_batch[4])
    after_vis_eval = plan_path(after_fused_eval, eval_batch[4])

    # --- Compute eval-scene spline on AFTER costmap ---
    after_eval_np = after_fused_eval.detach().cpu().numpy().squeeze()
    _, eval_spline_xy, eval_P, _ = compute_spline_path(
        after_eval_np, eval_goal_np, k=3, num_ctrl_pts=10)

    # Also compute spline for BEFORE eval for comparison
    before_eval_np = before_fused_eval.detach().cpu().numpy().squeeze()
    _, before_eval_spline_xy, _, _ = compute_spline_path(
        before_eval_np, eval_goal_np, k=3, num_ctrl_pts=10)

    # --- Numpy conversion helpers ---
    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().squeeze()
        return np.array(t).squeeze()

    before_train = to_np(before_fused_train)
    after_train = to_np(after_fused_train)
    before_eval = before_eval_np
    after_eval = after_eval_np
    delta_train = after_train - before_train
    delta_eval = after_eval - before_eval

    # --- Paths as numpy ---
    orig_np = orig_path if isinstance(orig_path, np.ndarray) else np.array(orig_path)
    user_np = user_path if isinstance(user_path, np.ndarray) else np.array(user_path)

    # =====================================================================
    # Figure layout: 5 rows × max(4, n_classes+1) cols
    # =====================================================================
    n_cls = len(obstacle_classes)
    n_cols = max(4, n_cls + 1)  # +1 for fused column

    fig, axes = plt.subplots(5, n_cols, figsize=(5.5 * n_cols, 5 * 5))

    # Hide any unused cells
    for row in range(5):
        for col in range(n_cols):
            axes[row, col].axis('off')

    # Shared color scales
    vmin_cost = min(before_train.min(), after_train.min(),
                    before_eval.min(), after_eval.min())
    vmax_cost = max(before_train.max(), after_train.max(),
                    before_eval.max(), after_eval.max())
    delta_abs_max = max(abs(delta_train).max(), abs(delta_eval).max()) + 1e-8

    # -----------------------------------------------------------------
    #  ROW 0 — Training scene: Before | After | Delta | Visitation
    # -----------------------------------------------------------------
    row = 0
    for col, (data, title) in enumerate([
        (before_train, "Before (Train)"),
        (after_train, "After (Train)"),
    ]):
        ax = axes[row, col]
        ax.imshow(data, cmap="hot", vmin=vmin_cost, vmax=vmax_cost, origin="upper")
        ax.plot(orig_np[:, 0], orig_np[:, 1], 'g--', linewidth=1.5, label='Original')
        ax.plot(user_np[:, 0], user_np[:, 1], 'c-', linewidth=1.5, label='User edit')
        draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                       obstacle_classes, show_radii=True)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Delta
    ax = axes[row, 2]
    im_delta = ax.imshow(delta_train, cmap="RdBu_r",
                         vmin=-delta_abs_max, vmax=delta_abs_max, origin="upper")
    draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                   obstacle_classes, show_radii=False)
    ax.plot(orig_np[:, 0], orig_np[:, 1], 'g--', linewidth=1.2)
    ax.plot(user_np[:, 0], user_np[:, 1], 'c-', linewidth=1.2)
    ax.set_title("Δ Cost (Train)", fontsize=11, fontweight='bold')
    fig.colorbar(im_delta, ax=ax, fraction=0.046, pad=0.04, label="Δ Cost")
    ax.axis('off')

    # Visitation overlay
    ax = axes[row, 3]
    ax.imshow(before_vis_train, cmap="Blues", alpha=0.55, origin="upper")
    ax.imshow(after_vis_train, cmap="Reds", alpha=0.55, origin="upper")
    ax.plot(orig_np[:, 0], orig_np[:, 1], 'g--', linewidth=1.5, label='Orig path')
    ax.plot(user_np[:, 0], user_np[:, 1], 'c-', linewidth=1.5, label='User path')
    draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                   obstacle_classes, show_radii=False)
    ax.set_title("Visitation (Train)\nBlue=Before  Red=After", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Legend for row 0
    legend_handles = make_obstacle_legend(obstacle_classes)
    legend_handles += [
        Line2D([0], [0], color='g', linestyle='--', linewidth=1.5, label='Orig path'),
        Line2D([0], [0], color='c', linestyle='-', linewidth=1.5, label='User edit'),
    ]
    axes[row, 0].legend(handles=legend_handles, loc='upper right', fontsize=7,
                        framealpha=0.8, fancybox=True)

    # -----------------------------------------------------------------
    #  ROW 1 — Eval scene: Before | After | Delta | Visitation + spline
    # -----------------------------------------------------------------
    row = 1
    for col, (data, title) in enumerate([
        (before_eval, "Before (Eval)"),
        (after_eval, "After (Eval)"),
    ]):
        ax = axes[row, col]
        ax.imshow(data, cmap="hot", vmin=vmin_cost, vmax=vmax_cost, origin="upper")
        draw_obstacles(ax, eval_positions, eval_radii, eval_goal_np,
                       obstacle_classes, show_radii=True)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Delta
    ax = axes[row, 2]
    im_delta_eval = ax.imshow(delta_eval, cmap="RdBu_r",
                              vmin=-delta_abs_max, vmax=delta_abs_max, origin="upper")
    draw_obstacles(ax, eval_positions, eval_radii, eval_goal_np,
                   obstacle_classes, show_radii=False)
    ax.set_title("Δ Cost (Eval)", fontsize=11, fontweight='bold')
    fig.colorbar(im_delta_eval, ax=ax, fraction=0.046, pad=0.04, label="Δ Cost")
    ax.axis('off')

    # Visitation + spline trajectories
    ax = axes[row, 3]
    ax.imshow(before_vis_eval, cmap="Blues", alpha=0.45, origin="upper")
    ax.imshow(after_vis_eval, cmap="Reds", alpha=0.45, origin="upper")
    # Spline paths
    ax.plot(before_eval_spline_xy[:, 0], before_eval_spline_xy[:, 1],
            color='dodgerblue', linewidth=2.2, linestyle='--', label='Before spline')
    ax.plot(eval_spline_xy[:, 0], eval_spline_xy[:, 1],
            color='orangered', linewidth=2.2, linestyle='-', label='After spline')
    # Control points of after-spline
    ax.plot(eval_P[:, 0], eval_P[:, 1], 'D', color='orangered',
            markersize=5, alpha=0.6, markeredgecolor='white', markeredgewidth=0.5)
    draw_obstacles(ax, eval_positions, eval_radii, eval_goal_np,
                   obstacle_classes, show_radii=False)
    ax.set_title("Eval Trajectories\nBlue=Before  Red=After", fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.axis('off')

    # -----------------------------------------------------------------
    #  ROW 2 — Per-class BEFORE (training scene)
    # -----------------------------------------------------------------
    row = 2
    cls_vmin = min(v.min() for v in before_per_class_train.values())
    cls_vmax = max(v.max() for v in before_per_class_train.values())

    for col, cls in enumerate(obstacle_classes):
        ax = axes[row, col]
        im = ax.imshow(before_per_class_train[cls], cmap="hot",
                       vmin=cls_vmin, vmax=cls_vmax, origin="upper")
        color = OBSTACLE_COLORS.get(cls, 'white')
        draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                       [cls], show_radii=True)  # Only this class
        ax.set_title(f"Before — {cls.capitalize()}", fontsize=10,
                     fontweight='bold', color=color)
        ax.axis('off')

    # Fused
    ax = axes[row, n_cls]
    ax.imshow(before_train, cmap="hot", vmin=vmin_cost, vmax=vmax_cost, origin="upper")
    draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                   obstacle_classes, show_radii=True)
    ax.set_title("Before — Fused", fontsize=10, fontweight='bold')
    ax.axis('off')

    # -----------------------------------------------------------------
    #  ROW 3 — Per-class AFTER (training scene)
    # -----------------------------------------------------------------
    row = 3
    for col, cls in enumerate(obstacle_classes):
        ax = axes[row, col]
        im = ax.imshow(after_per_class_train[cls], cmap="hot",
                       vmin=cls_vmin, vmax=cls_vmax, origin="upper")
        color = OBSTACLE_COLORS.get(cls, 'white')
        draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                       [cls], show_radii=True)
        ax.set_title(f"After — {cls.capitalize()}", fontsize=10,
                     fontweight='bold', color=color)
        ax.axis('off')

    ax = axes[row, n_cls]
    ax.imshow(after_train, cmap="hot", vmin=vmin_cost, vmax=vmax_cost, origin="upper")
    draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                   obstacle_classes, show_radii=True)
    ax.set_title("After — Fused", fontsize=10, fontweight='bold')
    ax.axis('off')

    # -----------------------------------------------------------------
    #  ROW 4 — Per-class DELTA (training scene)
    # -----------------------------------------------------------------
    row = 4
    per_class_delta_max = 1e-8
    per_class_deltas = {}
    for cls in obstacle_classes:
        d = after_per_class_train[cls] - before_per_class_train[cls]
        per_class_deltas[cls] = d
        per_class_delta_max = max(per_class_delta_max, abs(d).max())

    for col, cls in enumerate(obstacle_classes):
        ax = axes[row, col]
        im = ax.imshow(per_class_deltas[cls], cmap="RdBu_r",
                       vmin=-per_class_delta_max, vmax=per_class_delta_max,
                       origin="upper")
        color = OBSTACLE_COLORS.get(cls, 'white')
        draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                       [cls], show_radii=True, radii_alpha=0.15)
        ax.set_title(f"Δ — {cls.capitalize()}", fontsize=10,
                     fontweight='bold', color=color)
        ax.axis('off')

    ax = axes[row, n_cls]
    im_fused_delta = ax.imshow(delta_train, cmap="RdBu_r",
                               vmin=-delta_abs_max, vmax=delta_abs_max, origin="upper")
    draw_obstacles(ax, train_positions, train_radii, train_goal_np,
                   obstacle_classes, show_radii=True, radii_alpha=0.15)
    ax.set_title("Δ — Fused", fontsize=10, fontweight='bold')
    fig.colorbar(im_fused_delta, ax=ax, fraction=0.046, pad=0.04, label="Δ Cost")
    ax.axis('off')

    # -----------------------------------------------------------------
    #  Title & save
    # -----------------------------------------------------------------
    fig.suptitle("Finetuning Results: Training Scene vs Eval Scene\n"
                 "Rows: Train overview | Eval + spline | Per-class before | Per-class after | Per-class delta",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization to {save_path}")


# =========================================================================
# Main evaluate()
# =========================================================================



def get_user_input(batch,model,device,ddpm, obstacle_classes):
    
    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal, orientation = batch 

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)

            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            diffused_cm[cls].append(generated)
    
    fused_costmap = fuse_costmaps(diffused_cm)
    orig_path, user_path = get_user_adjustments(fused_costmap, positions, radii, goal, orientation)
        
    return orig_path, user_path, diffused_cm

def evaluate():
   
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
   
    #Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"
    output_dir = "eval_results"
    obstacle_classes = ["chair", "table", "bomb"]
    H, W = 128, 128
    batch_size = 1
    replay_buffer = []



    # Calculate channels: 2 (curr) + 2*(n-1) (others) + 1 (goal)
    n_classes = len(obstacle_classes)
    conditioning_channels = 4 + 4 * (n_classes - 1) + 1

    # --- Load Data ---
    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)

    #Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch8.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #Setup reverse diffusion process
    ddpm = DDPM(timesteps=1000, device=device)
    train_batch = next(iter(loader))

    #Generate costmaps and get user input
    orig_path, user_path, diffused_cms = get_user_input(obstacle_classes= obstacle_classes,batch=train_batch,model=model,device=device,ddpm=ddpm,)
    
    if np.all(user_path == None):
        print("No modifications made to original path!")
        return None
    
    #Get classes to modify
    #target_classes, area_mask = identify_classes(obstacle_classes=obstacle_classes,batch=batch, orig_path=orig_path, user_path=user_path)
   
    # After get_user_input returns diffused_cms:
    before_fused_train = fuse_costmaps(diffused_cms)
    before_per_class_train = {
        cls: diffused_cms[cls][0].cpu().numpy().squeeze() for cls in obstacle_classes
    }

    # Get edit regions
    edit_regions = get_edit_regions(
        orig_path=orig_path,
        user_path=user_path,
        obstacle_classes=obstacle_classes,
        batch=train_batch,
    )

    print(f"Found {len(edit_regions)} edit region(s)")
    filtered_edit_regions = []
    affected_class_threshold = 0.83
    for i, (class_contributions, points, mask) in enumerate(edit_regions):
    #    dominant_class = max(class_contributions, key=class_contributions.get)
    #    print(f"  Region {i}: {len(points)} pixels, dominant class = {dominant_class} ({class_contributions[dominant_class]*100:.1f}%)")
        curr_prob = 0
        affected_classes = set()
        while(curr_prob<affected_class_threshold):
            remaining_classes = {k: v for k, v in class_contributions.items() if k not in affected_classes}
            best_class = max(remaining_classes, key = remaining_classes.get)
            curr_prob += remaining_classes[best_class]
            affected_classes.add(best_class)
        filtered_edit_regions.append([mask, points, affected_classes])
        print(f" Region {i}: {len(points)} pixels ; Affected Classes: {affected_classes}")

    #Different scene to eval

    eval_batch = next(iter(loader))
   
    
    print(filtered_edit_regions)
    
    #Eval new scene before finetuning the models
    diffused_cm = {cls: [] for cls in obstacle_classes}
    features, targets, positions, radii, goal, _ = eval_batch 

    with torch.no_grad():
        for cls in obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)

            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            
            diffused_cm[cls].append(generated)
    
    before_fused_costmap = fuse_costmaps(diffused_cm)
    
    planner = SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device)
    #Finetune the models
    loss_history = finetune_models(
        model=model,
        batch=train_batch,
        orig_path=orig_path,
        user_path=user_path,
        device=device,
        lr=1e-4,
        edit_regions=filtered_edit_regions,
        epochs=500,
        ddpm=ddpm,
        planner=planner,
        w_directional_reg = 0.5,
    )


    before_fused_eval, before_per_class_eval = generate_fused(model, eval_batch, obstacle_classes, device, ddpm)
   
    after_fused_eval, _ = generate_fused(model, eval_batch, obstacle_classes, device, ddpm)
    _, _, eval_positions, eval_radii, eval_goal, eval_orientations = eval_batch

    visualize_simple(before_fused_costmap, after_fused_eval, eval_goal,
                     eval_positions, eval_radii, eval_orientations, obstacle_classes,
                     save_path="trajectory_comparison_eval.png")

if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate() 
