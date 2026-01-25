#utils.py
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle
from pathlib import Path
import torch 

from matplotlib.widgets import RectangleSelector, CheckButtons


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



def plot_baseline(num_samples, obstacle_classes, dataset, ddpm, model):
    # Generate and visualize
    for sample_idx in range(num_samples):
        features, targets = dataset[sample_idx]
        
        n_rows = conditioning_channels + 2  # +1 for GT, +1 for generated
        n_cols = n_classes
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        for col, cls in enumerate(obstacle_classes):
            conditioning = features[cls].unsqueeze(0).to(device)
            gt_costmap = targets[cls].squeeze().numpy()
            
            # Generate costmap
            shape = (1, 1, 64, 64)
            generated = ddpm.sample(model.experts[cls], conditioning, shape)
            generated = generated.squeeze().cpu().numpy()
            
            # Plot each conditioning channel (rows 0 to conditioning_channels-1)
            for ch in range(conditioning_channels):
                axes[ch, col].imshow(features[cls][ch].numpy(), cmap='gray')
                if col == 0:
                    axes[ch, col].set_ylabel(channel_names[ch] if ch < len(channel_names) else f"ch_{ch}")
                if ch == 0:
                    axes[ch, col].set_title(cls)
                axes[ch, col].axis('off')
            
            # Plot ground truth (second to last row)
            axes[-2, col].imshow(gt_costmap, cmap='hot')
            if col == 0:
                axes[-2, col].set_ylabel("Ground Truth")
            axes[-2, col].axis('off')
            
            # Plot generated (last row)
            axes[-1, col].imshow(generated, cmap='hot')
            if col == 0:
                axes[-1, col].set_ylabel("Generated")
            axes[-1, col].axis('off')
            
#            all_generated[cls] = generated    
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{sample_idx}.png")
        plt.close()
        print(f"Saved sample_{sample_idx}.png")



def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better for structural learning than linear.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def responsibilities_from_costmaps(costmaps, alpha=8.0, power=1.0, free_space="uniform"):
    """
    costmaps: dict[class_name -> (H,W) cost], each in [-1, 1] (your gaussian outputs)
    alpha: softmax sharpness (bigger = more decisive)
    power: optionally sharpen hazard: hazard^power
    free_space:
        "uniform" -> if all hazards are 0, return uniform distribution
        "zeros"   -> if all hazards are 0, return all zeros
    returns: dict[class_name -> (H,W) responsibility in [0,1], sums to 1 per cell (unless free_space="zeros")
    """
    keys = list(costmaps.keys())

    # Stack costmaps: (K,H,W)
    C = np.stack([costmaps[k].astype(np.float32) for k in keys], axis=0)

    # Hazard signal 
    if power != 1.0:
        C = np.sign(C) * (np.abs(C) ** power)


    # Softmax over classes (numerically stable)
    logits = alpha * C
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    denom = np.sum(exp_logits, axis=0, keepdims=True)

    R = exp_logits / denom  # (K,H,W)
    
    return {k: R[i] for i, k in enumerate(keys)}


