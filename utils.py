#utils.py
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle
from pathlib import Path
import torch
from matplotlib.widgets import RectangleSelector, CheckButtons
from matplotlib import cm
import numpy as np
from skimage.graph import route_through_array
from scipy.interpolate import BSpline
from spline import *


def get_user_adjustments(fused_costmap, obstacle_positions, radii, goal_position):

    map_np = fused_costmap[0,0].detach().cpu().numpy()


    max_val = np.max(map_np)
    min_val = np.min(map_np)

    map_np = ((map_np-min_val)/(max_val - min_val) +1e-8)
    #map_np[map_np > 0.8] = np.inf
    #https://stackoverflow.com/questions/32551536/draw-marker-in-image
    #plt.annotate('25, 50', xy=(25, 50))
    #print(obstacle_positions)
   
    fig, ax = plt.subplots()
    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    #Tech debt but obstacle positions is a list of dictionaries
    for cls, obs_list in obstacle_positions[0].items():
        for pos in obs_list:
            ax.plot(pos[1],pos[0], color=colors[cls], marker='o', label=f'{cls}')
    goal = goal_position[0]

    ax.plot(goal[1], goal[0], color='olive', marker='*', label='goal')

    #Get path
    path = route_through_array(map_np, [0,0], (goal[0],goal[1]), fully_connected=True)
    path_array = np.array(path[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]

    k = 3
    num_ctrl_pts = 10
    x_s, y_s, P, U= generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)

    #Plot B-Spline and control points
    ax.plot(x_s, y_s, linewidth=2, label="Original B-spline curve")
    ax.plot(P[:, 0], P[:, 1], "o--", alpha=0.7, label="original control polygon")
    
    dragger = DraggableBSpline(ax, U, P, k, n_samples=800)

    ax.imshow(map_np, origin='lower')
    #plt.colorbar()
    ax.legend(loc='best')
    plt.show()

def visualize_3d(fused_cm):
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

    combined_map = torch.logsumexp(diffused_stacked, dim=1)# - np.exp(-1)
        
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



