

import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPM:
    def __init__(self, timesteps=1000, s=0.08, device="cuda"):
        self.timesteps = timesteps
        self.device = device


        #Improved DDPM paper
        steps = torch.arange(timesteps+1, device=device)
        f_t = torch.cos(((steps / timesteps) + s) / (1+s) * torch.pi / 2) **2
        alpha_bar = f_t / f_t[0]

        
        self.betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar) 


    def q_sample(self, x_0, t, noise=None):
        """
            Forward diffusion process

            Args:
                x_0: Target costmap (B, 1, H. W)
                t: Timesteps (B, )

            returns:
                x_t: Noisy data
                noise: The noise that was added
        """


        if noise is None:
            noise = torch.randn_like(x_0)
        #noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise #eqn for forward diffusion process

        return x_t, noise

    def p_sample(self, model, x_t, t, conditioning, return_logprob=False):
        """
        Reverse process
        """
        
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        noise_pred = model(x_t, t_tensor, conditioning)
        
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.betas[t]
        
        noise_coef = beta_t / self.sqrt_one_minus_alpha_bar[t]
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
        mean = sqrt_recip_alpha * (x_t - noise_coef * noise_pred)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_prev_unclamped = mean + sigma * noise
            
            if return_logprob:
                log_2pi = math.log(2.0 * math.pi)
                log_prob_elem = -0.5 * ((x_prev_unclamped - mean) / sigma) ** 2 \
                                - torch.log(sigma) - 0.5 * log_2pi
                log_prob = log_prob_elem.flatten(1).sum(dim=1)  # (B,)
            
            x_prev = torch.clamp(x_prev_unclamped, -1.0, 1.0)
        else:
            x_prev = mean
            x_prev = torch.clamp(x_prev, -1.0, 1.0)
            if return_logprob:
                log_prob = torch.zeros((B,), device=x_t.device)
        
        if return_logprob:
            return x_prev, log_prob
        return x_prev

    def compute_loss(self, model, x_0, conditioning):
        B = x_0.shape[0]

        t = torch.randint(0, self.timesteps, (B, ), device=x_0.device, dtype=torch.long)

        x_t, noise = self.q_sample(x_0, t)

        pred_noise = model(x_t, t, conditioning)

        loss = F.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, model, conditioning, shape):
        x_t = torch.randn(shape, device = self.device)

        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(model, x_t, t, conditioning)

        return x_t

    def sample_with_logprob(self, model, conditioning, shape):
        x_t = torch.randn(shape, device=self.device)
        total_logprob = torch.zeros((shape[0],), device=self.device)
        
        for t in reversed(range(self.timesteps)):
            x_t, lp = self.p_sample(model, x_t, t, conditioning, return_logprob=True)
            total_logprob += lp
        
        return x_t, total_logprob
