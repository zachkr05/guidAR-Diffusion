
#diffusion_utils.py


"""
Functions for sampling and diffusion scheduling etc
"""


"""
    Diffusion optimization https://arxiv.org/abs/2102.09672
"""
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

@torch.no_grad()
def sample_x0_prediction(model, cond, args, device):
    """Reverse Diffusion Loop for x0 prediction."""
    model.eval()
    B, _, H, W = cond.shape
    T = args.timesteps
    img = torch.randn((B, 1, H, W), device=device)
    
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) 

    for i in tqdm(reversed(range(0, T)), desc='Sampling', total=T):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        
        # Predict Clean Image
        pred_x0 = model(img, t, cond, training_phase='base')
        
        if i == 0:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        else:
            t_prev = torch.full((B,), i-1, device=device, dtype=torch.long)
            alpha_bar_prev = extract(alphas_cumprod, t_prev, img.shape)
        
        alpha_t = extract(alphas, t, img.shape)
        alpha_bar_t = extract(alphas_cumprod, t, img.shape)
        beta_t = extract(betas, t, img.shape)
        
        coeff1 = beta_t * torch.sqrt(alpha_bar_prev) / (1. - alpha_bar_t)
        coeff2 = (1. - alpha_bar_prev) * torch.sqrt(alpha_t) / (1. - alpha_bar_t)
        
        posterior_mean = coeff1 * pred_x0 + coeff2 * img
        
        if i > 0:
            noise = torch.randn_like(img)
            posterior_variance = beta_t * (1. - alpha_bar_prev) / (1. - alpha_bar_t)
            log_var = torch.log(torch.clamp(posterior_variance, min=1e-20))
            img = posterior_mean + torch.exp(0.5 * log_var) * noise
        else:
            img = posterior_mean

    return img


