#utils.py
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle
from pathlib import Path
import torch
from matplotlib.widgets import RectangleSelector, CheckButtons


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def get_user_adjustments(fused_cm):
    map_np = fused_cm[0,0].detach().cpu().numpy()


    H, W = map_np.shape
    x=np.arange(0,W)
    y=np.arange(0,H)
    X, Y = np.meshgrid(x,y)
    Z = map_np

    # 2. Create the 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)

    # 4. Customization
    ax.set_title("3D Fused Costmap")
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Cost Intensity')

    fig.colorbar(surf, shrink=0.5, aspect=5, label='Cost')

    ax.view_init(elev=60, azim=35)

    plt.show()

def fuse_costmaps(diffused_cm):


    """
        IMPORTANT, this only works for 1 CM see line for diffused_stacked
    """
    # stack all the tensor vals


    #TODO: fix for multiple CM
    diffused_stacked = torch.stack([v[0].squeeze(1) for v in diffused_cm.values()], dim=1)

    combined_map = torch.logsumexp(diffused_stacked, dim=1)
        
    #Restore to usual tensor
    return combined_map.unsqueeze(1)

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



