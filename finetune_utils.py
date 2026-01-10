import torch
import torch.nn as nn
import torch.nn.functional as F
from masking import SpatialContextMasker, extract_features
from diffusion_utils import schedule_betas, q_sample


def finetune_multiple_classes(
    model, 
    cond_input,      # [B, Channels, H, W]
    base_costmap,    # [B, 1, H, W] Tensor (NOT USED for target, just for reference)
    geometric_delta, # [H, W] Numpy Array
    affected_class_ids,
    learning_rate=1e-3,
    num_steps=20,
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
):
    """
    Finetunes LoRA by training the model to denoise toward the corrected costmap.
    
    Key insight: Diffusion models predict NOISE, not costmaps directly.
    We need to:
    1. Create target costmap (base + delta)
    2. Add noise to target at random timestep t
    3. Train model to predict that noise
    """
    device = cond_input.device
    model.train()
    
    # Setup diffusion schedule
    betas, alphas, alpha_bar = schedule_betas(timesteps, beta_start, beta_end, device=device)
    
    # 1. Freeze base model, unfreeze only affected class LoRAs
    model.freeze_all_lora_except_classes(affected_class_ids)
    
    # 2. Get trainable parameters
    params = model.get_multi_class_lora_parameters(affected_class_ids)
    if not params:
        print("No trainable parameters found!")
        return 0.0
    
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # 3. Create target costmap
    # base_costmap is in [0, 1] range from sampling, convert to [-1, 1] for training
    base_normalized = base_costmap * 2.0 - 1.0
    
    delta_tensor = torch.from_numpy(geometric_delta).float().to(device).unsqueeze(0).unsqueeze(0)
    
    # Scale delta appropriately (it's a cost change, not in [-1,1] space)
    # Adjust this scaling factor based on how strong corrections should be
    delta_scaled = delta_tensor * 0.5  
    
    target_costmap = torch.clamp(base_normalized + delta_scaled, -1.0, 1.0)
    
    # 4. Compute spatial mask for weighted loss
    features = extract_features(cond_input, model.num_classes)
    masker = SpatialContextMasker(feature_dim=features.shape[1])
    spatial_mask = masker.compute_context_mask(
        torch.from_numpy(geometric_delta).to(device), 
        features, 
        device=device
    )
    
    # 5. Training loop - train model to denoise toward target
    final_loss = 0.0
    B = cond_input.shape[0]
    
    for step in range(num_steps):
    
        # Replace the loss computation:
        pred_noise = model(cond_input, t, interaction_mask=spatial_mask)

        # Standard noise prediction loss (maintains model stability)
        base_loss = F.mse_loss(pred_noise, noise)

        # Bias term: REDUCE noise prediction where we want HIGHER output
        # Lower ε → higher x̂₀
        bias_loss = (pred_noise * spatial_mask).mean()  # Minimize this

        total_loss = base_loss + bias_loss * 0.5

        total_loss.backward()
    return total_loss


def finetune_direct_output(
    model, 
    cond_input,
    base_costmap,    # [B, 1, H, W] in [0,1] range
    geometric_delta, # [H, W] numpy
    affected_class_ids,
    learning_rate=5e-4,
    num_steps=50,
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
):
    """
    Alternative: Finetune by running full denoising and comparing output.
    
    This is slower but more direct - we actually see what the model outputs
    after full sampling and compare to target.
    
    Only use for debugging or very small num_steps.
    """
    from train import sample_ddpm_with_cond
    
    device = cond_input.device
    betas, alphas, alpha_bar = schedule_betas(timesteps, beta_start, beta_end, device=device)
    
    model.freeze_all_lora_except_classes(affected_class_ids)
    params = model.get_multi_class_lora_parameters(affected_class_ids)
    
    if not params:
        return 0.0
    
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Target
    delta_tensor = torch.from_numpy(geometric_delta).float().to(device).unsqueeze(0).unsqueeze(0)
    target = torch.clamp(base_costmap + delta_tensor * 0.3, 0.0, 1.0)
    
    # Spatial mask
    features = extract_features(cond_input, model.num_classes)
    masker = SpatialContextMasker(feature_dim=features.shape[1])
    spatial_mask = masker.compute_context_mask(
        torch.from_numpy(geometric_delta).to(device), features, device=device
    )
    
    final_loss = 0.0
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Run abbreviated denoising (fewer steps for speed)
        model.eval()
        with torch.enable_grad():
            # Start from noise
            x = torch.randn_like(base_costmap)
            
            # Run just last 100 steps (where details emerge)
            for t_val in reversed(range(0, 100)):
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                x_in = torch.cat([cond_input, x], dim=1)
                
                eps = model(x_in, t, interaction_mask=spatial_mask)
                
                a_t = alphas[t_val]
                a_bar_t = alpha_bar[t_val]
                b_t = betas[t_val]
                
                mew = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * eps)
                
                if t_val > 0:
                    z = torch.randn_like(x)
                    x = mew + torch.sqrt(b_t) * z
                else:
                    x = mew
            
            output = (x.clamp(-1, 1) + 1) / 2
            
            # Loss comparing to target
            raw_loss = F.mse_loss(output, target, reduction='none')
            loss = (raw_loss * spatial_mask).mean()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        final_loss = loss.item()
        
        if step % 10 == 0:
            print(f"  Step {step}: loss={final_loss:.5f}")
    
    return final_loss


def estimate_uncertainty_mc_dropout(model, cond, n_passes=10, timesteps=1000, 
                                     beta_start=0.0001, beta_end=0.02):
    """
    MC Dropout uncertainty estimation.
    """
    from train import sample_ddpm_with_cond
    
    device = cond.device
    betas, alphas, alpha_bar = schedule_betas(timesteps, beta_start, beta_end, device=device)
    
    # Enable dropout in LoRA layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_passes):
            out = sample_ddpm_with_cond(model, cond, betas, alphas, alpha_bar, device=device)
            predictions.append(out)
    
    stack = torch.stack(predictions)
    mean_pred = torch.mean(stack, dim=0)
    variance = torch.var(stack, dim=0)
    
    return mean_pred, variance
