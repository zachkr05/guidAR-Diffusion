

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


    def predict_start_from_noise(self, x_t, t, noise):
        """
        Reconstructs x_0 from x_t and the predicted noise.
        """
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        return (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    
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
        
        # Predict x_0 and clamp it (not x_{t-1})
        x0_pred = self.predict_start_from_noise(x_t, t_tensor, noise_pred)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Recompute mean from clamped x_0
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.betas[t]
        
        mean = (torch.sqrt(alpha_t) * (1 - alpha_bar_t / alpha_t) / (1 - alpha_bar_t)) * x_t \
             + (torch.sqrt(alpha_bar_t / alpha_t) * beta_t / (1 - alpha_bar_t)) * x0_pred
        
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
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




    # Add to MoE/ddpm.py
    def sample_with_partial_logprob(self, model, conditioning, shape, logprob_steps=5):
        x_t = torch.randn(shape, device=self.device)
        total_logprob = torch.zeros((shape[0],), device=self.device)
        
        num_steps = self.timesteps
        start_logprob_at = num_steps - logprob_steps
        
        for i, t in enumerate(reversed(range(num_steps))):
            if i >= start_logprob_at:
                x_t, lp = self.p_sample(model, x_t, t, conditioning, return_logprob=True)
                total_logprob += lp
            else:
                with torch.no_grad():
                    x_t = self.p_sample(model, x_t, t, conditioning, return_logprob=False)
        
        return x_t, total_logprob
