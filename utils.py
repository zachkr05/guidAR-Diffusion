#utils.py
import numpy as np


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



