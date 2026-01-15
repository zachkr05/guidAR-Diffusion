



import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM:
    def __init__(self, timesteps=1000, s=0.08, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.betas = np.zeros(timesteps)
        self.alphas = np.zeros(timesteps)
        self.alpha_bar = np.zeros(timesteps)
        self.sqsrt_alpha_bar = np.zeros(timesteps)
        self.sqrt_one_minus_alpha_bar = np.zeros(timesteps)
        
        #for reverse
        self.sqrt_recip_alpha = np.zeros(timesteps)
        self.sqrt_recip_alpha_bar = np.zeros(timesteps)
        self.posterior_variance = np.zeros(timesteps)
