# eval.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from DataGenerator.dataGenerator import CostmapDataset
from MoE.UNet import LightweightUNet
from MoE.ddpm import DDPM
from train import ExpertEnsemble
import pickle
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, CheckButtons


def fuse_costmaps_lse(costmaps, obstacle_positions, obstacle_radii, temperature=5.0):
    names = list(costmaps.keys())
    H, W = list(costmaps.values())[0].shape[-2:]
    
    rows, cols = np.ogrid[:H, :W]
    
    fused = np.zeros((H, W))
    responsibilities = {cls: np.zeros((H, W)) for cls in names}
    
    inside_any = np.zeros((H, W), dtype=bool)
    inside_class = {cls: np.zeros((H, W), dtype=bool) for cls in names}
    
    # Find cells inside obstacles
    for cls in names:
        positions = obstacle_positions.get(cls, [])
        radii = obstacle_radii.get(cls, [])
        
        for (r, c), rad in zip(positions, radii):
            dist_sq = (rows - r)**2 + (cols - c)**2
            mask = dist_sq <= rad**2
            inside_class[cls] |= mask
            inside_any |= mask
    
    # Compute distance to nearest obstacle of each class
    distances = {}
    valid_classes = []
    
    for cls in names:
        positions = obstacle_positions.get(cls, [])
        
        if len(positions) == 0:
            distances[cls] = np.full((H, W), np.inf)
        else:
            min_dist = np.full((H, W), np.inf)
            for (r, c) in positions:
                dist = np.sqrt((rows - r)**2 + (cols - c)**2)
                min_dist = np.minimum(min_dist, dist)
            distances[cls] = min_dist
            valid_classes.append(cls)
    
    # Stack distances
    dist_stack = np.stack([distances[k] for k in names], axis=0)  # (N, H, W)
    
    # Minimum distance across all classes at each cell
    min_dist_all = np.min(dist_stack, axis=0, keepdims=True)  # (1, H, W)
    min_dist_all = np.where(np.isinf(min_dist_all), 0, min_dist_all)
    
    # Relative distance: how much farther than the closest?
    relative_dist = dist_stack - min_dist_all  # (N, H, W)
    
    # Exponential decay on relative distance
    closeness_stack = np.exp(-relative_dist / temperature)
    
    # Zero out classes with no obstacles
    mask = np.array([1.0 if cls in valid_classes else 0.0 for cls in names])
    mask = mask[:, None, None]
    closeness_stack = closeness_stack * mask
    
    # Normalize to get responsibilities
    epsilon = 1e-6
    sum_closeness = np.sum(closeness_stack, axis=0, keepdims=True) + epsilon
    resp_stack = closeness_stack / sum_closeness
    
    # Costmap fusion
    costmap_stack = np.stack([np.squeeze(costmaps[k]) for k in names], axis=0)
    max_cost = np.max(costmap_stack)
    
    # Inside obstacles: max cost, 100% responsibility
    for cls in names:
        responsibilities[cls][inside_class[cls]] = 1.0
        fused[inside_class[cls]] = max_cost
    
    # Outside obstacles
    outside = ~inside_any
    
    for i, cls in enumerate(names):
        responsibilities[cls][outside] = resp_stack[i][outside]
    
    fused[outside] = np.sum(resp_stack * costmap_stack, axis=0)[outside]
    
    return fused, responsibilities


def interactive_costmap_plot(
    fused,
    responsibilities,
    obstacle_classes,
    goal,
    pkl_path="rect_probs.pkl",
    label_classes=("chair", "table", "bomb"),  # the tags you want to assign
):
    """
    Hover: shows responsibilities at cursor.
    Drag rectangle: saves probs for every cell in selected region to pkl.
    Checkboxes: choose any combo of chair/table/bomb and it will be saved with the region.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Fused costmap
    im = axes[0].imshow(fused, cmap='hot')
    axes[0].set_title("Fused Costmap (drag rectangle to save; tick labels to tag)")
    plt.colorbar(im, ax=axes[0])

    # Plot goal
    axes[0].plot(goal[1], goal[0], marker='*', color='cyan', markersize=20,
                 markeredgecolor='black', markeredgewidth=1.5, label='Goal')
    axes[0].legend(loc='upper right')

    # Right: Responsibilities bar chart
    n_classes = len(obstacle_classes)
    bars = axes[1].bar(obstacle_classes, [0] * n_classes)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Responsibility")
    axes[1].set_title("Responsibilities at cursor position")

    # Text annotation for hover position
    pos_text = axes[0].text(
        0.02, 0.98, "", transform=axes[0].transAxes,
        va='top', ha='left', color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )

    # ---------- Checkbox UI for labels ----------
    # Create a small axes area for checkboxes (figure coords)
    cb_ax = fig.add_axes([0.74, 0.05, 0.23, 0.20])  # [left, bottom, width, height]
    cb_ax.set_title("Labels for next selection")
    initial = [False] * len(label_classes)
    check = CheckButtons(cb_ax, list(label_classes), initial)

    # Store current label selection
    selected = {name: False for name in label_classes}

    def on_check(label):
        selected[label] = not selected[label]
    check.on_clicked(on_check)

    def current_labels():
        return [k for k, v in selected.items() if v]

    # ---------- PKL logging ----------
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    if pkl_path.exists():
        try:
            with pkl_path.open("rb") as f:
                log = pickle.load(f)
            if not isinstance(log, list):
                log = []
        except Exception:
            log = []
    else:
        log = []

    def save_log():
        tmp = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(log, f)
        tmp.replace(pkl_path)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ---------- Hover updates ----------
    def on_move(event):
        if event.inaxes != axes[0] or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < fused.shape[1] and 0 <= y < fused.shape[0]):
            return

        pos_text.set_text(f"Position: ({x}, {y})\nCost: {fused[y, x]:.3f}")
        for i, cls in enumerate(obstacle_classes):
            bars[i].set_height(float(responsibilities[cls][y, x]))
        axes[1].set_title(f"Responsibilities at ({x}, {y})")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    # ---------- Rectangle selection ----------
    def on_select(eclick, erelease):
        if (eclick.xdata is None or eclick.ydata is None or
            erelease.xdata is None or erelease.ydata is None):
            return

        # Convert to integer cell coords (inclusive range)
        x0, y0 = int(np.floor(eclick.xdata)), int(np.floor(eclick.ydata))
        x1, y1 = int(np.floor(erelease.xdata)), int(np.floor(erelease.ydata))

        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        # Clamp to bounds
        xmin = clamp(xmin, 0, fused.shape[1] - 1)
        xmax = clamp(xmax, 0, fused.shape[1] - 1)
        ymin = clamp(ymin, 0, fused.shape[0] - 1)
        ymax = clamp(ymax, 0, fused.shape[0] - 1)

        labels = current_labels()
        t = time.time()
        n_cells = (xmax - xmin + 1) * (ymax - ymin + 1)

        # Append one record per cell (keeps it simple to train on later)
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                probs = {cls: float(responsibilities[cls][y, x]) for cls in obstacle_classes}
                
                print(probs)
                #log.append({
                #    "timestamp": t,
                #    "labels": labels,  # <-- your multi-label tag here
                #    "selection": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
                #    "x": x,
                #    "y": y,
                #    "fused_cost": float(fused[y, x]),
                #    "probs": probs,
                #})

        save_log()
        print(f"[rect] labels={labels} saved x[{xmin},{xmax}] y[{ymin},{ymax}] ({n_cells} cells) -> {pkl_path}")

    rect_selector = RectangleSelector(
        axes[0], on_select,
        useblit=True,
        button=[1],            # left mouse drag
        interactive=True,
        spancoords='data'
    )

    plt.tight_layout()
    plt.show()



def evaluate(checkpoint_path, num_samples=4, save_dir="eval_results"):
    
    # Config (must match training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obstacle_classes = ["chair", "table", "bomb"]
    n_classes = len(obstacle_classes)
    conditioning_channels = 2 + 2 * (n_classes - 1) + 1
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load model
    model = ExpertEnsemble(obstacle_classes, conditioning_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # DDPM
    ddpm = DDPM(timesteps=1000, device=device)
    
    # Init the sim
    dataset = CostmapDataset(n_samples=num_samples, H=64, W=64)
    dataset.obstacle_classes = obstacle_classes
    
    # Channel names for labeling
    channel_names = ["own_bin", "own_rad", "other_bin_1", "other_bin_2", "other_rad_1", "other_rad_2", "goal"]

    #TODO: Make it for N samples

    #all_generated = []


    features, targets, obstacle_positions, obstacle_radii, goal = dataset[0]

    all_generated = {}
    all_targets = {}
    
    print("Generating costmaps...")
    for cls in obstacle_classes:
        conditioning = features[cls].unsqueeze(0).to(device)
        gt_costmap = targets[cls].squeeze().numpy()
        
        shape = (1, 1, 64, 64)
        generated = ddpm.sample(model.experts[cls], conditioning, shape)
        generated = generated.squeeze().cpu().numpy()
        
        all_generated[cls] = generated
        all_targets[cls] = gt_costmap
        print(f"  {cls}: done")
    
    # Fuse costmaps
    print("Fusing costmaps...")
    fused, responsibilities = fuse_costmaps_lse(all_generated, obstacle_positions, obstacle_radii, temperature=5)
    
    # Interactive plot
    print("Launching interactive plot...")
    interactive_costmap_plot(fused, responsibilities, obstacle_classes, goal)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch5.pt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="eval_results")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.num_samples, args.save_dir)
