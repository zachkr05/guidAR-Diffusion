import torch
import torch.nn.functional as F


def schedule_betas(num_steps, beta_start, beta_end, device="cuda"):
    #For Section 2, eqn 2, We need to schudule betas. 
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas,dim=0)
    return betas, alphas, alpha_bar


def q_sample(x0, t, alpha_bar, noise = None):
    if noise is None:
        noise = torch.randn_like(x0)
    a = alpha_bar[t].view(-1,1,1,1)
    return torch.sqrt(a) * x0 + torch.sqrt(1-a) * noise, noise #eqn (4) via reparameterization E[x] = E[mew + std* epsilon] = mew + std * E[epsilon] = mew + std * 0 (no noise at final x) = mew



