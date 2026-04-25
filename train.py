import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataGenerator.dataset import CostmapDataset
from MoE.ddpm import DDPM
from MoE.simple_model import SimpleTrajectoryModel


def collate_fn(batch):
    features_list, targets_list, positions_list, radii_list, goals_list, angles_list = zip(*batch)
    keys = list(features_list[0].keys())
    features = {k: torch.stack([f[k] for f in features_list]) for k in keys}
    targets = torch.stack(targets_list)
    goals = np.stack(goals_list)
    return features, targets, goals


def visualize_samples(model, ddpm, sample_batch, epoch, out_dir, device, num_samples=4):
    model.eval()
    features, targets, goals = sample_batch

    features_dev = {k: v[:num_samples].to(device) for k, v in features.items()}
    targets_dev = targets[:num_samples].to(device)

    with torch.no_grad():
        shape = (num_samples, 1, targets_dev.shape[2], targets_dev.shape[3])
        generated = ddpm.sample(model, features_dev, shape)

    gen_np = generated.cpu().numpy()
    tgt_np = targets_dev.cpu().numpy()

    gen_np = generated.cpu().numpy()

    print(
    "Target stats:",
    "min=%.3f" % targets_dev.min().item(),
    "max=%.3f" % targets_dev.max().item(),
    "mean=%.3f" % targets_dev.mean().item(),
    "median=%.3f" % targets_dev.median().item(),
    )

    print(
        "Generated stats:",
        "min=%.3f" % generated.min().item(),
        "max=%.3f" % generated.max().item(),
        "mean=%.3f" % generated.mean().item(),
        "median=%.3f" % generated.median().item(),
    )

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    axes = np.array(axes).reshape(2, num_samples)
    for i in range(num_samples):
        axes[0, i].imshow(tgt_np[i, 0], cmap="hot", vmin=-1, vmax=1)
        axes[0, i].set_title("Ground Truth %d" % i)
        axes[0, i].axis("off")

        axes[1, i].imshow(gen_np[i, 0], cmap="hot", vmin=-1, vmax=1)
        axes[1, i].set_title("Generated %d" % i)
        axes[1, i].axis("off")

    fig.suptitle("Epoch %d" % epoch)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "samples_epoch_%04d.png" % epoch), dpi=150)
    plt.close(fig)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train trajectory diffusion model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--obstacle_classes", nargs="+", default=["chair", "table", "bomb"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="runs/traj_diffusion")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    # Dataset
    dataset = CostmapDataset(n_samples=args.n_samples, H=args.H, W=args.W)
    dataset.obstacle_classes = args.obstacle_classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    in_channels_per_class = 4 + 4 * (len(args.obstacle_classes) - 1) + 1

    model = SimpleTrajectoryModel(
        obstacle_classes=args.obstacle_classes,
        in_channels_per_class=in_channels_per_class,
        base_channels=args.base_channels,
        time_dim=args.base_channels * 4,
    ).to(device)

    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    loss_history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])
        print("Resumed from %s at epoch %d" % (args.resume, start_epoch))

    viz_batch = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model: %s params (%s trainable)" % ("{:,}".format(total_params), "{:,}".format(trainable_params)))
    print("Device: %s" % device)
    print("Dataset: %s samples, %d batches/epoch" % ("{:,}".format(len(dataset)), len(loader)))
    print("Obstacle classes: %s" % args.obstacle_classes)
    print("Input channels per class: %d" % in_channels_per_class)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (features, targets, goals) in enumerate(loader):
            features_dev = {k: v.to(device) for k, v in features.items()}
            targets_dev = targets.to(device)

            if viz_batch is None:
                viz_batch = (features, targets, goals)

            loss = ddpm.compute_loss(model, targets_dev, features_dev)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx % 50 == 0:
                print("  epoch %3d | batch %4d/%d | loss %.6f" % (epoch, batch_idx, len(loader), loss.item()))

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        lr_now = scheduler.get_last_lr()[0]
        print("Epoch %3d | avg loss %.6f | lr %.2e" % (epoch, avg_loss, lr_now))

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, "checkpoints", "ckpt_epoch_%04d.pt" % epoch)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss_history": loss_history,
                "args": vars(args),
            }, ckpt_path)
            print("  saved %s" % ckpt_path)

        if (epoch + 1) % args.viz_every == 0 and viz_batch is not None:
            visualize_samples(model, ddpm, viz_batch, epoch, args.out_dir, device)
            print("  saved visualization")

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print("Done. Outputs in %s" % args.out_dir)


if __name__ == "__main__":
    main()
