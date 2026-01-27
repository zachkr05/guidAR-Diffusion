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


def find_span(n, p, u, U):
    """
    Find span index for u.
    n = n_ctrl - 1 (highest control index)
    """
    # Special case at the end
    if u >= U[n+1]:
        return n
    if u <= U[p]:
        return p

    low, high = p, n+1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def basis_funs(span, u, p, U):
    """Compute the nonzero B-spline basis functions N_{span-p...span,p}(u)."""
    N = np.zeros(p+1, dtype=float)
    left = np.zeros(p+1, dtype=float)
    right = np.zeros(p+1, dtype=float)
    N[0] = 1.0

    for j in range(1, p+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r+1] + left[j-r]
            # denom should be > 0 for valid knot vectors, but guard anyway
            temp = 0.0 if denom == 0 else N[r] / denom
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        N[j] = saved
    return N


def bspline_design_matrix(u, n_ctrl, degree, U):
    p = degree
    n = n_ctrl - 1
    A = np.zeros((len(u), n_ctrl), dtype=float)

    for j, uj in enumerate(u):
        span = find_span(n, p, uj, U)
        N = basis_funs(span, uj, p, U)  # length p+1
        i0 = span - p
        A[j, i0:i0+p+1] = N

    return A

def clamped_knots(num_ctrl_pts, degree):
    p = degree
    m = num_ctrl_pts + p 
    U = np.zeros(m+1, dtype=float)

    U[:p+1] = 0.0
    U[m-p:] = 1.0

    num_interior = num_ctrl_pts - p - 1
    U[p+1:m-p] = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]

    return U

def reparam_curve(Q):
    d = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
    total = d.sum()
    u = np.concatenate(([0.0], np.cumsum(d) / total))
    u[-1] = 1.0
    return u

def generate_clamped_spline(x_np, y_np, k, num_ctrl_pts):
    Q = np.column_stack([x_np, y_np])
    u = reparam_curve(Q)
    U = clamped_knots(num_ctrl_pts, k)
    A = bspline_design_matrix(u, num_ctrl_pts, k, U)
    
    P, *_ = np.linalg.lstsq(A,Q,rcond=None)
    P[0]  = Q[0]
    P[-1] = Q[-1]
    P = np.asarray(P)

    t0, t1 = U[k], U[-k-1]
    t = np.linspace(t0, t1, 500)

    splx = BSpline(U, P[:, 0], k)
    sply = BSpline(U, P[:, 1], k)

    x_s = splx(t)
    y_s = sply(t)

    return x_s, y_s, P

def get_user_adjustments(fused_costmap, obstacle_positions, radii, goal_position):

    map_np = fused_costmap[0,0].detach().cpu().numpy()


    max_val = np.max(map_np)
    min_val = np.min(map_np)

    map_np = ((map_np-min_val)/(max_val - min_val) +1e-8)
    #map_np[map_np > 0.8] = np.inf
    #https://stackoverflow.com/questions/32551536/draw-marker-in-image
    #plt.annotate('25, 50', xy=(25, 50))
    #print(obstacle_positions)
    
    colors = {'chair': 'green', 'table': 'red', 'bomb': 'blue'}

    #Tech debt but obstacle positions is a list of dictionaries
    for cls, obs_list in obstacle_positions[0].items():
        for pos in obs_list:
            plt.plot(pos[1],pos[0], color=colors[cls], marker='o', label=f'{cls}')
    goal = goal_position[0]

    plt.plot(goal[1], goal[0], color='olive', marker='*', label='goal')

    #Get path
    path = route_through_array(map_np, [0,0], (goal[0],goal[1]), fully_connected=True)
    path_array = np.array(path[0])
    x_np = path_array[:, 1]
    y_np = path_array[:, 0]

    k = 3
    num_ctrl_pts = 10
    x_s, y_s, P = generate_clamped_spline(x_np, y_np, k, num_ctrl_pts)

    #Plot B-Spline and control points
    plt.plot(x_s, y_s, linewidth=2, label="B-spline curve")
    plt.plot(P[:, 0], P[:, 1], "o--", alpha=0.7, label="control polygon")
    
    plt.imshow(map_np, origin='lower')
    plt.colorbar()
    plt.legend(loc='best')
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



