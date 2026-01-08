import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_utils import schedule_betas, q_sample
from DataGenerator.dataset_costmap import MultiClassCostmapDataset, get_cond_channels
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES, NUM_ORIENTATIONS, ORIENTATIONS
from UNet.UNet import UNet


@torch.no_grad()
def sample_ddpm_with_cond(model, cond, betas, alphas, alpha_bar, device="cuda"):
    """
    Algorithm 2

    cond: [B, NUM_CLASSES*3+1, H, W]  # Updated channel count
    """
    model.eval()
    T = betas.shape[0]
    n_samples = cond.shape[0]
    img_size = cond.shape[2]
    x = torch.randn(n_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        x_in = torch.cat([cond, x], dim=1)
        eps = model(x_in, t_batch)

        a_t = alphas[t]
        a_bar_t = alpha_bar[t]
        b_t = betas[t]
        mew = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * eps)

        if t > 0:
            z = torch.randn_like(x)
            x = mew + torch.sqrt(b_t) * z
        else:
            x = mew

    x = (x.clamp(-1, 1) + 1) / 2
    return x


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Number of obstacle classes: {NUM_CLASSES}")
    for i in range(NUM_CLASSES):
        print(f"  Class {i}: {OBSTACLE_CLASSES[i]['name']} "
              f"(amp={OBSTACLE_CLASSES[i]['amp']}, sigma={OBSTACLE_CLASSES[i]['sigma']})")
    
    print(f"Number of orientations: {NUM_ORIENTATIONS}")
    for i in range(NUM_ORIENTATIONS):
        print(f"  Orientation {i}: {ORIENTATIONS[i]['name']} ({np.degrees(ORIENTATIONS[i]['angle']):.0f}°)")

    # Init variables for diffusion process
    T = args.timesteps
    betas, alphas, alpha_bar = schedule_betas(T, args.beta_start, args.beta_end, device=device)

    ds = MultiClassCostmapDataset(
        n_samples=args.n_samples,
        H=args.img_size,
        W=args.img_size,
        n_obs_per_class=args.n_obs_per_class,
        min_obs_per_class=args.min_obs_per_class,
        min_total_obs=args.min_total_obs
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Calculate input channels: conditioning + noisy costmap
    cond_channels = get_cond_channels()  # NUM_CLASSES * 3 + 1
    in_channels = cond_channels + 1       # + 1 for noisy costmap
    
    model = UNet(
        in_channels=in_channels,
        lora_rank=args.lora_rank
    ).to(device)

    print(f"\nDataset: MultiClassCostmapDataset with Orientations")
    print(f"  Samples: {len(ds)}")
    print(f"  Obstacles per class: {args.min_obs_per_class} to {args.n_obs_per_class}")
    print(f"  Conditioning channels: {cond_channels}")
    print(f"    - {NUM_CLASSES} occupancy channels (one per class)")
    print(f"    - {NUM_CLASSES * 2} orientation channels (sin/cos per class)")
    print(f"    - 1 goal channel")
    print(f"  Input channels to model: {in_channels} (conditioning + noisy costmap)")

    params_to_train = model.parameters()
    lr = args.lr

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {lr}")

    params_to_train = [p for p in params_to_train if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=args.weight_decay)
    
    if args.use_scheduler:
        total_steps = args.epochs * len(dl)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
    else:
        scheduler = None

    os.makedirs(args.ckpt_dir, exist_ok=True)

    model.train()
    step = 0
    best_loss = float('inf')

    print(f"\nStrting training for {args.epochs} epochs...")
    print(f"Steps per epoch: {len(dl)}")
    print("-" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dl:
            # Dataset returns: (cond, x0, obstacles_by_class, goal)
            # We only need cond and x0 for training
            cond, x0 = batch[0], batch[1]
            
            # cond: [B, NUM_CLASSES*3+1, H, W] - occupancy + orientation + goal
            # x0: [B, 1, H, W] - ground truth costmap
            cond = cond.to(device)
            x0 = x0.to(device)

            B, _, H, W = x0.shape

            # Sample random timesteps
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            # Forward diffusion (add noise)
            xt, noise = q_sample(x0, t, alpha_bar)

            # Concatenate: [cond, noisy_costmap]
            # cond is [occupancy_classes, orientation_sin_cos, goal]
            x_in = torch.cat([cond, xt], dim=1)  # [B, NUM_CLASSES*3+2, H, W]

            # Predict noise
            pred_noise = model(x_in, t)

            # Compute loss
            loss = F.mse_loss(pred_noise, noise)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_to_train, args.grad_clip)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            # Logging
            epoch_loss += loss.item()
            epoch_steps += 1

            if step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} lr={current_lr:.2e}")

            # Sample visualization
            if step % args.sample_every == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    sample_cond = cond[:4]
                    sample = sample_ddpm_with_cond(model, sample_cond, betas, alphas, alpha_bar, device)
                model.train()

            step += 1

        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if args.class_id is not None:
            class_name = OBSTACLE_CLASSES[args.class_id]['name']
            ckpt_name = f"ckpt_class_{class_name}_epoch_{epoch}.pt"
        else:
            ckpt_name = f"ckpt_multiclass_epoch_{epoch}.pt"
        
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'args': vars(args)
        }, ckpt_path)
        
        print(f"Saved checkpoint: {ckpt_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args)
            }, best_path)
            print(f"New best model saved: {best_path}")

    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model with Class-Specific LoRA")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
    
    # Diffusion Hyperparameters
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Beta schedule start")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta schedule end")
    
    # Model / LoRA
    parser.add_argument("--lora-rank", type=int, default=8, help="Rank for LoRA adapters")
    
    # Dataset
    parser.add_argument("--n-samples", type=int, default=150000, help="Dataset size per epoch")
    parser.add_argument("--img-size", type=int, default=64, help="Image resolution")
    parser.add_argument("--n-obs-per-class", type=int, default=2, help="Max obstacles per class")
    parser.add_argument("--min-obs-per-class", type=int, default=0, help="Min obstacles per class")
    parser.add_argument("--min-total-obs", type=int, default=1, help="Minimum total obstacles")
    
    # System / Logging
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--log-every", type=int, default=100, help="Log loss every N steps")
    parser.add_argument("--sample-every", type=int, default=500, help="Save sample image every N steps")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use-scheduler", action="store_true", help="Use Cosine LR Scheduler")
    
    # Optional: Train only specific class?
    parser.add_argument("--class-id", type=int, default=None, help="Specific class ID to focus on (optional)")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
