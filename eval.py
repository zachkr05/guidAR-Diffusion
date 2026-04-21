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

from utils.finetune import finetune_models



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



def plan_path_through_costmap(costmap_np, goal, k=3, num_ctrl_pts=10):
    """Plan a shortest path through a 2D costmap and fit a B-spline.
    Same method as the original path generation in get_user_adjustments.
    Returns (N,2) spline path in [x, y] = [col, row] convention."""
    H, W = costmap_np.shape
    mn, mx = costmap_np.min(), costmap_np.max()
    normed = (costmap_np - mn) / (mx - mn + 1e-8) + 1e-8

    goal_y = int(np.clip(goal[0], 0, H - 1))
    goal_x = int(np.clip(goal[1], 0, W - 1))

    path_result = route_through_array(normed, [0, 0], [goal_y, goal_x],
                                       fully_connected=True, geometric=True)
    path_array = np.array(path_result[0])
    x_np = path_array[:, 1]  # col
    y_np = path_array[:, 0]  # row

    x_s, y_s, P, U = generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)
    return np.column_stack([x_s, y_s])


def visualize_improvement(scene_train, eval_scenes, N):
    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    train_fused_cm, train_orig_path, train_user_path, train_obstacle_classes, train_filtered_edit_regions, train_batch = scene_train

    # Layout: 1 training column + N eval columns, 3 rows (before/after/diff)
    fig, axes = plt.subplots(3, N + 1, figsize=(6 * (N + 1), 18))
    if N + 1 == 1:
        axes = axes[:, np.newaxis]

    # --- Training scene (top row only, hide rows 2 and 3) ---
    for row in range(3):
        ax = axes[row, 0]
        if row > 0:
            ax.axis('off')
            continue

        map_np = train_fused_cm[0, 0].detach().cpu().numpy()
        ax.imshow(map_np, origin='lower')

        _, _, positions, radii, goal, orientations = train_batch
        for cls, obs_list in positions[0].items():
            for i, pos in enumerate(obs_list):
                ax.plot(pos[1], pos[0], color=colors.get(cls, 'white'), marker='o')
                angle = orientations[cls][i].item()
                dx = 5 * np.cos(angle)
                dy = 5 * np.sin(angle)
                ax.arrow(pos[1], pos[0], dx, dy,
                         head_width=1.5, head_length=1.0,
                         fc=colors.get(cls, 'white'), ec=colors.get(cls, 'white'), alpha=0.8)

        goal_val = goal[0]
        ax.plot(goal_val[1], goal_val[0], color='olive', marker='*', markersize=15)
        ax.plot(train_orig_path[:, 0], train_orig_path[:, 1], linewidth=2, label='Original', color='cyan')
        ax.plot(train_user_path[:, 0], train_user_path[:, 1], linewidth=2, label='User', color='magenta')

        for mask, points, affected_classes, _ in train_filtered_edit_regions:
            ax.contour(mask, levels=[0.5], colors='yellow', linewidths=1.5, linestyles='dashed')

        ax.set_title('Training Scene')
        ax.legend(loc='upper right', fontsize=7)

    # --- Eval scenes ---
    for j, (before_fused, after_fused, eval_batch) in enumerate(eval_scenes):
        col = j + 1

        before_np = before_fused[0, 0].detach().cpu().numpy()
        after_np = after_fused[0, 0].detach().cpu().numpy()
        diff_np = after_np - before_np

        _, _, eval_positions, eval_radii, eval_goal, eval_orientations = eval_batch

        # Compute trajectories through before and after costmaps
        before_spline = plan_path_through_costmap(before_np, eval_goal[0])
        after_spline = plan_path_through_costmap(after_np, eval_goal[0])

        # Rows 0 and 1: before / after with trajectories
        for row, (data_np, label, traj) in enumerate([
            (before_np, 'Before', before_spline),
            (after_np, 'After', after_spline),
        ]):
            ax = axes[row, col]
            ax.imshow(data_np, origin='lower')

            for cls, obs_list in eval_positions[0].items():
                for i, pos in enumerate(obs_list):
                    ax.plot(pos[1], pos[0], color=colors.get(cls, 'white'), marker='o')
                    angle = eval_orientations[cls][i].item()
                    dx = 5 * np.cos(angle)
                    dy = 5 * np.sin(angle)
                    ax.arrow(pos[1], pos[0], dx, dy,
                             head_width=1.5, head_length=1.0,
                             fc=colors.get(cls, 'white'), ec=colors.get(cls, 'white'), alpha=0.8)

            eval_goal_val = eval_goal[0]
            ax.plot(eval_goal_val[1], eval_goal_val[0], color='olive', marker='*', markersize=15)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2, color='cyan', label='Path')
            ax.set_title(f'Eval {j+1} ({label})')
            ax.legend(loc='upper right', fontsize=7)

        # Row 2: difference map with both trajectories
        ax = axes[2, col]
        vmax = np.abs(diff_np).max() or 1.0
        im = ax.imshow(diff_np, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)

        for cls, obs_list in eval_positions[0].items():
            for i, pos in enumerate(obs_list):
                ax.plot(pos[1], pos[0], color=colors.get(cls, 'white'), marker='o', markersize=4)
                angle = eval_orientations[cls][i].item()
                dx = 5 * np.cos(angle)
                dy = 5 * np.sin(angle)
                ax.arrow(pos[1], pos[0], dx, dy,
                         head_width=1.5, head_length=1.0,
                         fc=colors.get(cls, 'white'), ec=colors.get(cls, 'white'), alpha=0.8)

        ax.plot(before_spline[:, 0], before_spline[:, 1], linewidth=2, color='cyan', linestyle='--', label='Before path')
        ax.plot(after_spline[:, 0], after_spline[:, 1], linewidth=2, color='magenta', label='After path')
        ax.set_title(f'Eval {j+1} (Diff)')
        ax.legend(loc='upper right', fontsize=7)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=cls, markersize=8)
                       for cls, c in colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(colors), fontsize=10)

    plt.tight_layout()
    plt.savefig("improvement_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def evaluate():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obstacle_classes = ["chair", "table", "bomb"]
    H, W = 128, 128
    batch_size = 1
    N_eval = 4  # number of eval scenes

    n_classes = len(obstacle_classes)
    conditioning_channels = 4 + 4 * (n_classes - 1) + 1

    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)
    data_iter = iter(loader)

    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch7.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ddpm = DDPM(timesteps=1000, device=device)

    # --- Training scene ---
    train_batch = next(data_iter)
    orig_path, user_path, diffused_cms = get_user_input(
        batch=train_batch, model=model, device=device, ddpm=ddpm, obstacle_classes=obstacle_classes)

    if np.all(user_path == None):
        print("No modifications made to original path!")
        return None

    before_fused_train = fuse_costmaps(diffused_cms)

    edit_regions = get_edit_regions(
        orig_path=orig_path, user_path=user_path,
        obstacle_classes=obstacle_classes, batch=train_batch)

    filtered_edit_regions = []
    affected_class_threshold = 0.3
    for i, (class_contributions, points, mask) in enumerate(edit_regions):
        curr_prob = 0
        affected_classes = set()
        while curr_prob < affected_class_threshold:
            remaining_classes = {k: v for k, v in class_contributions.items() if k not in affected_classes}
            best_class = max(remaining_classes, key=remaining_classes.get)
            curr_prob += remaining_classes[best_class]
            affected_classes.add(best_class)
        filtered_edit_regions.append([mask, points, affected_classes, class_contributions])
        print(f" Region {i}: {len(points)} pixels ; Affected Classes: {affected_classes}")

    # --- Collect N eval scenes BEFORE finetuning ---
    eval_batches = []
    eval_before_fused = []
    for _ in range(N_eval):
        eval_batch = next(data_iter)
        eval_batches.append(eval_batch)
        fused, _ = generate_fused(model, eval_batch, obstacle_classes, device, ddpm)
        eval_before_fused.append(fused)

    # --- Finetune on training scene ---
    planner = SoftGridPlanner(iters=256, tau=1.0, step_cost=0.05).to(device)
    finetune_models(
        model=model, batch=train_batch, orig_path=orig_path, user_path=user_path,
        device=device, lr=1e-4, edit_regions=filtered_edit_regions,
        epochs=500, ddpm=ddpm, planner=planner, obstacle_classes=obstacle_classes)

    # --- Regenerate all eval scenes AFTER finetuning ---
    eval_scenes = []
    for i in range(N_eval):
        after_fused, _ = generate_fused(model, eval_batches[i], obstacle_classes, device, ddpm)
        eval_scenes.append((eval_before_fused[i], after_fused, eval_batches[i]))

    # --- Visualize ---
    scene_train = (before_fused_train, orig_path, user_path, obstacle_classes, filtered_edit_regions, train_batch)
    visualize_improvement(scene_train, eval_scenes, N_eval)


if __name__ == "__main__":
    
    torch.manual_seed(42)
    evaluate()
