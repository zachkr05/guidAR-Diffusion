import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import *
from skimage.graph import route_through_array
from spline import generate_clamped_spline

# Import local modules
from DataGenerator.dataGenerator import CostmapDataset
from MoE.ddpm import DDPM
from train import ExpertEnsemble
from torch.utils.data.dataloader import default_collate
from planner import SoftGridPlanner


def collate_ignore_metadata(batch):
    features = default_collate([item[0] for item in batch])
    targets = default_collate([item[1] for item in batch])
    goals = default_collate([item[4] for item in batch])
    positions = [item[2] for item in batch]
    radii = [item[3] for item in batch]
    return features, targets, positions, radii, goals


def generate_costmaps_for_scene(model, batch, ddpm, device, seed=42):
    """Generate costmaps for all obstacle classes for a given scene."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    features, targets, positions, radii, goal = batch
    diffused_cm = {}
    
    with torch.no_grad():
        for cls in model.obstacle_classes:
            expert_model = model.experts[cls]
            cond = features[cls].to(device)
            gt = targets[cls].to(device)
            generated = ddpm.sample(expert_model, cond, shape=gt.shape)
            diffused_cm[cls] = [generated]
    
    return diffused_cm


def compute_path_on_fused(fused_costmap, goal):
    """Compute path using route_through_array on fused costmap."""
    fused_np = fused_costmap[0, 0].detach().cpu().numpy()
    
    # Normalize
    fused_norm = (fused_np - fused_np.min()) / (fused_np.max() - fused_np.min() + 1e-8)
    
    goal_np = goal[0].cpu().numpy()
    goal_y, goal_x = int(goal_np[0]), int(goal_np[1])
    
    path = route_through_array(fused_norm, [2, 2], [goal_y, goal_x], fully_connected=True)
    path_array = np.array(path[0])
    
    return np.column_stack([path_array[:, 1], path_array[:, 0]])  # x, y format


def evaluate():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obstacle_classes = ["chair", "table", "bomb"]
    H, W = 128, 128
    batch_size = 1
    
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes - 1) + 1

    # --- Load Data ---
    print("Generating evaluation dataset...")
    dataset = CostmapDataset(n_samples=50, H=H, W=W)
    dataset.obstacle_classes = obstacle_classes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ignore_metadata)

    # Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load("checkpoints/checkpoint_epoch5.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Setup
    ddpm = DDPM(timesteps=1000, device=device)
    loader_iter = iter(loader)
    
    # =========================================================================
    # Get TWO different scenes
    # =========================================================================
    print("\n" + "="*60)
    print("SCENE SETUP")
    print("="*60)
    
    # Scene 1: Training scene (where user will make edits)
    train_batch = next(loader_iter)
    print("Scene 1 (Training): Loaded")
    
    # Scene 2: Validation scene (never seen during finetuning)
    val_batch = next(loader_iter)
    print("Scene 2 (Validation): Loaded")
    
    # =========================================================================
    # BEFORE FINETUNING: Generate costmaps for BOTH scenes
    # =========================================================================
    print("\n" + "="*60)
    print("BEFORE FINETUNING")
    print("="*60)
    
    # Training scene - before
    print("\nGenerating costmaps for training scene (before)...")
    train_cm_before = generate_costmaps_for_scene(model, train_batch, ddpm, device, seed=42)
    train_fused_before = fuse_costmaps(train_cm_before)
    train_path_before = compute_path_on_fused(train_fused_before, train_batch[4])
    
    # Validation scene - before
    print("Generating costmaps for validation scene (before)...")
    val_cm_before = generate_costmaps_for_scene(model, val_batch, ddpm, device, seed=123)
    val_fused_before = fuse_costmaps(val_cm_before)
    val_path_before = compute_path_on_fused(val_fused_before, val_batch[4])
    
    # =========================================================================
    # USER INTERACTION: Get edits on training scene
    # =========================================================================
    print("\n" + "="*60)
    print("USER INTERACTION (Training Scene Only)")
    print("="*60)
    
    features, targets, positions, radii, goal = train_batch
    orig_path, user_path = get_user_adjustments(train_fused_before, positions, radii, goal)
    
    # =========================================================================
    # FINETUNING: Only on training scene
    # =========================================================================
    print("\n" + "="*60)
    print("FINETUNING (Training Scene Only)")
    print("="*60)
    
    from finetune_focused import finetune_models_focused
    
    loss_history, closest_idx = finetune_models_focused(
        model=model,
        batch=train_batch,
        orig_path=orig_path,
        user_path=user_path,
        device=device,
        lr=1e-4,
        target_class="chair",
        epochs=500,
        ddpm=ddpm,
        planner=SoftGridPlanner(iters=80, tau=1.0, step_cost=0.05).to(device),
    )
    
    # =========================================================================
    # AFTER FINETUNING: Generate costmaps for BOTH scenes
    # =========================================================================
    print("\n" + "="*60)
    print("AFTER FINETUNING")
    print("="*60)
    
    # Training scene - after
    print("\nGenerating costmaps for training scene (after)...")
    train_cm_after = generate_costmaps_for_scene(model, train_batch, ddpm, device, seed=42)
    train_fused_after = fuse_costmaps(train_cm_after)
    train_path_after = compute_path_on_fused(train_fused_after, train_batch[4])
    
    # Validation scene - after
    print("Generating costmaps for validation scene (after)...")
    val_cm_after = generate_costmaps_for_scene(model, val_batch, ddpm, device, seed=123)
    val_fused_after = fuse_costmaps(val_cm_after)
    val_path_after = compute_path_on_fused(val_fused_after, val_batch[4])
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    visualize_train_and_val(
        # Training scene
        train_fused_before, train_fused_after,
        train_path_before, train_path_after,
        orig_path, user_path,
        train_batch[2], train_batch[4],  # positions, goal
        # Validation scene
        val_fused_before, val_fused_after,
        val_path_before, val_path_after,
        val_batch[2], val_batch[4],  # positions, goal
        target_class="chair"
    )


def visualize_train_and_val(
    # Training scene
    train_fused_before, train_fused_after,
    train_path_before, train_path_after,
    orig_path, user_path,
    train_positions, train_goal,
    # Validation scene
    val_fused_before, val_fused_after,
    val_path_before, val_path_after,
    val_positions, val_goal,
    target_class="chair"
):
    """
    Visualize before/after for both training and validation scenes.
    
    Layout:
    Row 1: Training scene (where user made edits)
    Row 2: Validation scene (unseen during finetuning)
    """
    
    # Convert to numpy
    train_before_np = train_fused_before[0, 0].detach().cpu().numpy()
    train_after_np = train_fused_after[0, 0].detach().cpu().numpy()
    val_before_np = val_fused_before[0, 0].detach().cpu().numpy()
    val_after_np = val_fused_after[0, 0].detach().cpu().numpy()
    
    train_goal_np = train_goal[0].cpu().numpy()
    val_goal_np = val_goal[0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    colors = {'chair': 'lime', 'table': 'red', 'bomb': 'cyan'}
    
    def plot_obstacles(ax, pos_dict):
        for cls, obs_list in pos_dict.items():
            for pos in obs_list:
                ax.plot(pos[1], pos[0], color=colors.get(cls, 'white'), 
                       marker='o', markersize=8, markeredgecolor='black')
    
    # =========================================================================
    # Row 1: Training Scene
    # =========================================================================
    
    # Before finetuning
    im1 = axes[0, 0].imshow(train_before_np, cmap='viridis', origin='lower')
    plot_obstacles(axes[0, 0], train_positions[0])
    axes[0, 0].plot(train_path_before[:, 0], train_path_before[:, 1], 'r-', linewidth=2, label='Model Path')
    axes[0, 0].plot(0, 0, 'wo', markersize=10, markeredgecolor='black')
    axes[0, 0].plot(train_goal_np[1], train_goal_np[0], 'r*', markersize=15)
    axes[0, 0].set_title("TRAIN: Before Finetuning", fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # After finetuning
    im2 = axes[0, 1].imshow(train_after_np, cmap='viridis', origin='lower')
    plot_obstacles(axes[0, 1], train_positions[0])
    axes[0, 1].plot(train_path_after[:, 0], train_path_after[:, 1], 'b-', linewidth=2, label='Model Path')
    axes[0, 1].plot(user_path[:, 0], user_path[:, 1], 'g--', linewidth=2, label='User Path')
    axes[0, 1].plot(0, 0, 'wo', markersize=10, markeredgecolor='black')
    axes[0, 1].plot(train_goal_np[1], train_goal_np[0], 'r*', markersize=15)
    axes[0, 1].set_title("TRAIN: After Finetuning", fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Delta
    train_delta = train_after_np - train_before_np
    max_val = max(abs(train_delta.min()), abs(train_delta.max()), 1e-6)
    im3 = axes[0, 2].imshow(train_delta, cmap='seismic', origin='lower', vmin=-max_val, vmax=max_val)
    axes[0, 2].plot(train_path_before[:, 0], train_path_before[:, 1], 'r-', linewidth=2, label='Before')
    axes[0, 2].plot(train_path_after[:, 0], train_path_after[:, 1], 'b-', linewidth=2, label='After')
    axes[0, 2].plot(user_path[:, 0], user_path[:, 1], 'g--', linewidth=2, label='User')
    axes[0, 2].set_title("TRAIN: Costmap Delta", fontsize=12, fontweight='bold')
    axes[0, 2].legend(loc='upper right')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Path comparison
    axes[0, 3].plot(orig_path[:, 0], orig_path[:, 1], 'r-', linewidth=2, label='Original')
    axes[0, 3].plot(train_path_after[:, 0], train_path_after[:, 1], 'b-', linewidth=2, label='After FT')
    axes[0, 3].plot(user_path[:, 0], user_path[:, 1], 'g--', linewidth=2, label='User Target')
    axes[0, 3].set_xlim(0, 128)
    axes[0, 3].set_ylim(0, 128)
    axes[0, 3].set_aspect('equal')
    axes[0, 3].set_title("TRAIN: Path Comparison", fontsize=12, fontweight='bold')
    axes[0, 3].legend(loc='upper right')
    axes[0, 3].grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 2: Validation Scene (UNSEEN during finetuning)
    # =========================================================================
    
    # Before finetuning
    im4 = axes[1, 0].imshow(val_before_np, cmap='viridis', origin='lower')
    plot_obstacles(axes[1, 0], val_positions[0])
    axes[1, 0].plot(val_path_before[:, 0], val_path_before[:, 1], 'r-', linewidth=2, label='Model Path')
    axes[1, 0].plot(0, 0, 'wo', markersize=10, markeredgecolor='black')
    axes[1, 0].plot(val_goal_np[1], val_goal_np[0], 'r*', markersize=15)
    axes[1, 0].set_title("VAL (unseen): Before Finetuning", fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # After finetuning
    im5 = axes[1, 1].imshow(val_after_np, cmap='viridis', origin='lower')
    plot_obstacles(axes[1, 1], val_positions[0])
    axes[1, 1].plot(val_path_after[:, 0], val_path_after[:, 1], 'b-', linewidth=2, label='Model Path')
    axes[1, 1].plot(0, 0, 'wo', markersize=10, markeredgecolor='black')
    axes[1, 1].plot(val_goal_np[1], val_goal_np[0], 'r*', markersize=15)
    axes[1, 1].set_title("VAL (unseen): After Finetuning", fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Delta
    val_delta = val_after_np - val_before_np
    max_val_v = max(abs(val_delta.min()), abs(val_delta.max()), 1e-6)
    im6 = axes[1, 2].imshow(val_delta, cmap='seismic', origin='lower', vmin=-max_val_v, vmax=max_val_v)
    axes[1, 2].plot(val_path_before[:, 0], val_path_before[:, 1], 'r-', linewidth=2, label='Before')
    axes[1, 2].plot(val_path_after[:, 0], val_path_after[:, 1], 'b-', linewidth=2, label='After')
    axes[1, 2].set_title("VAL (unseen): Costmap Delta", fontsize=12, fontweight='bold')
    axes[1, 2].legend(loc='upper right')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Metrics comparison
    ax_text = axes[1, 3]
    ax_text.axis('off')
    
    # Compute some metrics
    train_path_change = np.linalg.norm(train_path_after - train_path_before[:len(train_path_after)], axis=1).mean() if len(train_path_after) == len(train_path_before) else -1
    val_path_change = np.linalg.norm(val_path_after - val_path_before[:len(val_path_after)], axis=1).mean() if len(val_path_after) == len(val_path_before) else -1
    
    train_delta_mean = np.abs(train_delta).mean()
    val_delta_mean = np.abs(val_delta).mean()
    
    # Check if changes are localized to chairs
    train_delta_at_chairs = []
    for pos in train_positions[0].get('chair', []):
        y, x = int(pos[0]), int(pos[1])
        y, x = np.clip(y, 5, 122), np.clip(x, 5, 122)
        train_delta_at_chairs.append(np.abs(train_delta[y-5:y+5, x-5:x+5]).mean())
    train_chair_delta = np.mean(train_delta_at_chairs) if train_delta_at_chairs else 0
    
    val_delta_at_chairs = []
    for pos in val_positions[0].get('chair', []):
        y, x = int(pos[0]), int(pos[1])
        y, x = np.clip(y, 5, 122), np.clip(x, 5, 122)
        val_delta_at_chairs.append(np.abs(val_delta[y-5:y+5, x-5:x+5]).mean())
    val_chair_delta = np.mean(val_delta_at_chairs) if val_delta_at_chairs else 0
    
    metrics_text = f"""
    GENERALIZATION METRICS
    ══════════════════════
    
    Training Scene:
    • Mean |Δ costmap|: {train_delta_mean:.4f}
    • Mean |Δ| at chairs: {train_chair_delta:.4f}
    
    Validation Scene (unseen):
    • Mean |Δ costmap|: {val_delta_mean:.4f}
    • Mean |Δ| at chairs: {val_chair_delta:.4f}
    
    ══════════════════════
    INTERPRETATION:
    
    If VAL delta is:
    • Localized to chairs → ✓ Generalized!
    • Zero everywhere → ✗ No transfer
    • Random/noisy → ✗ Overfitting
    """
    
    ax_text.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
                 verticalalignment='center', transform=ax_text.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Finetuning Generalization Test: '{target_class}' Expert", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"generalization_test_{target_class}.png", dpi=150)
    plt.show()
    
    print(f"\nSaved to generalization_test_{target_class}.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    evaluate()
