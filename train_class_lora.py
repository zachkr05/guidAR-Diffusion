"""
Training script for Class-Specific LoRA UNet.

Trains a diffusion model where each obstacle class (chair, table, person, wall)
has its own LoRA adapter, allowing independent fine-tuning per class.

Usage:
    # Train all classes from scratch
    python train_class_lora.py
    
    # Fine-tune only chair LoRA (after pretraining)
    python train_class_lora.py --pretrained checkpoints/best_model.pt --class-id 0 --lora-only
    
    # Fine-tune only person LoRA
    python train_class_lora.py --pretrained checkpoints/best_model.pt --class-id 2 --lora-only
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_utils import schedule_betas, q_sample
from DataGenerator.dataset_costmap import MultiClassCostmapDataset
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES
from UNet.UNet_class_lora import ClassLoRAUNet, create_class_lora_unet


# ==============================================================================
# Sampling
# ==============================================================================

@torch.no_grad()
def sample_ddpm(model, cond, betas, alphas, alpha_bar, device="cuda"):
    """Generate samples using DDPM reverse process."""
    model.eval()
    T = betas.shape[0]
    n_samples = cond.shape[0]
    img_size = cond.shape[2]
    x = torch.randn(n_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        x_in = torch.cat([x, cond], dim=1)
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


# ==============================================================================
# Training
# ==============================================================================

def count_parameters(model, trainable_only=False):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of obstacle classes: {NUM_CLASSES}")
    for i in range(NUM_CLASSES):
        print(f"  Class {i}: {OBSTACLE_CLASSES[i]['name']} "
              f"(amp={OBSTACLE_CLASSES[i]['amp']}, sigma={OBSTACLE_CLASSES[i]['sigma']})")

    # -------------------------------------------------------------------------
    # Setup diffusion schedule
    # -------------------------------------------------------------------------
    T = args.timesteps
    betas, alphas, alpha_bar = schedule_betas(T, args.beta_start, args.beta_end, device=device)

    # -------------------------------------------------------------------------
    # Dataset - Multi-class!
    # -------------------------------------------------------------------------
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
    
    print(f"\nDataset: MultiClassCostmapDataset")
    print(f"  Samples: {len(ds)}")
    print(f"  Obstacles per class: {args.min_obs_per_class} to {args.n_obs_per_class}")
    print(f"  Conditioning channels: {NUM_CLASSES + 1} (classes + goal)")
    print(f"  Input channels to model: {NUM_CLASSES + 2} (classes + goal + noisy costmap)")

    # -------------------------------------------------------------------------
    # Model wth Class-Specific LoRA
    # -------------------------------------------------------------------------
    model = create_class_lora_unet(lora_rank=args.lora_rank).to(device)

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"\nLoading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded")

    # -------------------------------------------------------------------------
    # Freeze strategy
    # -------------------------------------------------------------------------
    if args.class_id is not None:
        # Fine-tune only one class
        class_name = OBSTACLE_CLASSES[args.class_id]['name']
        print(f"\n*** Fine-tuning only class {args.class_id} ({class_name}) ***")
        model.freeze_all_lora_except_class(args.class_id)
        params_to_train = model.get_class_lora_parameters(args.class_id)
        lr = args.lora_lr
    elif args.lora_only:
        # Train all LoRA, freeze base
        print("\n*** Training all LoRA adapters (base frozen) ***")
        model.freeze_base_model()
        params_to_train = model.get_lora_parameters()
        lr = args.lora_lr
    else:
        # Train everything
        print("\n*** Training full model ***")
        params_to_train = model.parameters()
        lr = args.lr

    # Print parameter counts
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {lr}")

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    # Filter out parameters that don't require grad
    params_to_train = [p for p in params_to_train if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=args.weight_decay)
    
    if args.use_scheduler:
        total_steps = args.epochs * len(dl)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
    else:
        scheduler = None

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model.train()
    step = 0
    best_loss = float('inf')

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Steps per epoch: {len(dl)}")
    print("-" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for cond, x0 in dl:
            # cond: [B, NUM_CLASSES+1, H, W] - class occupancies + goal
            # x0: [B, 1, H, W] - ground truth costmap
            cond = cond.to(device)
            x0 = x0.to(device)

            B, _, H, W = x0.shape

            # Sample random timesteps
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            # Forward diffusion (add noise)
            xt, noise = q_sample(x0, t, alpha_bar)

            # Concatenate: [noisy_costmap, class_0, class_1, ..., class_N, goal]
            # Model expects: [class_0, ..., class_N, goal, noisy_costmap]
            # So reorder: cond is [classes, goal], we add xt at the end
            x_in = torch.cat([cond, xt], dim=1)  # [B, NUM_CLASSES+2, H, W]

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
                    # For sampling, we need [cond, noisy] format
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


@torch.no_grad()
def sample_ddpm_with_cond(model, cond, betas, alphas, alpha_bar, device):
    """
    Sample with conditioning in correct format.
    cond: [B, NUM_CLASSES+1, H, W]
    """
    model.eval()
    T = betas.shape[0]
    n_samples = cond.shape[0]
    img_size = cond.shape[2]
    x = torch.randn(n_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        # Model expects [classes, goal, noisy_costmap]
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


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Class-Specific LoRA UNet")

    # Training mode
    parser.add_argument("--lora-only", action="store_true",
                        help="Train only LoRA parameters (freeze base model)")
    parser.add_argument("--class-id", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Train only this class's LoRA (0=chair, 1=table, 2=person, 3=wall)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint")

    # LoRA hyperparameters
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for full training")
    parser.add_argument("--lora-lr", type=float, default=1e-3,
                        help="Learning rate for LoRA-only training")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")
    parser.add_argument("--use-scheduler", action="store_true",
                        help="Use cosine annealing scheduler")

    # Diffusion hyperparameters
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta-start", type=float, default=1e-4,
                        help="Beta schedule start")
    parser.add_argument("--beta-end", type=float, default=0.02,
                        help="Beta schedule end")

    # Dataset hyperparameters
    parser.add_argument("--n-samples", type=int, default=100000,
                        help="Number of training samples")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Image size (H=W)")
    parser.add_argument("--n-obs-per-class", type=int, default=3,
                        help="Max obstacles per class")
    parser.add_argument("--min-obs-per-class", type=int, default=0,
                        help="Min obstacles per class")
    parser.add_argument("--min-total-obs", type=int, default=1,
                        help="Minimum total obstacles")

    # System
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--log-every", type=int, default=200,
                        help="Log every N steps")
    parser.add_argument("--sample-every", type=int, default=2000,
                        help="Generate samples every N steps")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
