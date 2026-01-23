

import matplotlib.pyplot as plt
import numpy as np
from skimage import graph



from scipy.interpolate import make_interp_spline

def clamped_bspline_path(path_rows, path_cols, n=200, k=3, tangent_scale=0.0):
    """
    Fit a clamped B-spline through (col, row) points and resample to n points.

    path_rows, path_cols: 1D arrays of same length
    n: number of output samples
    k: spline degree (3 = cubic)
    tangent_scale:
        0.0 -> true 'clamped' (zero first-derivative at endpoints)
        >0  -> set endpoint tangents based on first/last segment * tangent_scale
               (often looks better than zero-tangent clamping for paths)
    Returns: (rows_smooth, cols_smooth)
    """
    r = np.asarray(path_rows, dtype=float)
    c = np.asarray(path_cols, dtype=float)
    if r.shape != c.shape or r.ndim != 1:
        raise ValueError("path_rows and path_cols must be 1D arrays of the same length")
    if len(r) < (k + 1):
        raise ValueError(f"Need at least k+1={k+1} points for degree k={k}")

    # Parameterize by chord length for nicer spacing
    d = np.sqrt(np.diff(r)**2 + np.diff(c)**2)
    t = np.concatenate(([0.0], np.cumsum(d)))
    if t[-1] == 0:
        t = np.linspace(0.0, 1.0, len(r))
    else:
        t = t / t[-1]

    # Choose boundary conditions
    if tangent_scale == 0.0:
        bc = "clamped"  # zero first-derivative at both ends
    else:
        # Estimate endpoint tangents from the end segments (in parameter space)
        # and scale them.
        # Use finite differences; guard against degenerate cases.
        dt0 = max(t[1] - t[0], 1e-12)
        dt1 = max(t[-1] - t[-2], 1e-12)
        dc0 = (c[1] - c[0]) / dt0
        dr0 = (r[1] - r[0]) / dt0
        dc1 = (c[-1] - c[-2]) / dt1
        dr1 = (r[-1] - r[-2]) / dt1
        bc_c = ((1, tangent_scale * dc0), (1, tangent_scale * dc1))
        bc_r = ((1, tangent_scale * dr0), (1, tangent_scale * dr1))

    # Fit splines for cols and rows over the same t
    if bc == "clamped":
        spl_c = make_interp_spline(t, c, k=k, bc_type="clamped")
        spl_r = make_interp_spline(t, r, k=k, bc_type="clamped")
    else:
        spl_c = make_interp_spline(t, c, k=k, bc_type=bc_c)
        spl_r = make_interp_spline(t, r, k=k, bc_type=bc_r)

    t_new = np.linspace(t[0], t[-1], n)
    cols_smooth = spl_c(t_new)
    rows_smooth = spl_r(t_new)
    return rows_smooth, cols_smooth

def user_interactive_plot(fused, responsibilities, obstacle_positions, obstacle_radii, goal):

    fig, axes = plt.subplots(1,4,figsize=(14,6)) #initial costmap with user modifications, deployment of original models on eval scene, finetuned costmap on a completely new scene, difference between original on eval scene and finetuned on eval scene

    im = axes[0].imshow(fused, origin='lower', cmap="hot")
    plt.colorbar(im, ax=axes[0])

    print(goal)
    axes[0].plot(goal[1], goal[0], marker='*', color='cyan', markersize=20, markeredgecolor='black', markeredgewidth=1.5, label='Goal')
    #axes[0].legend(loc='upper right')


    #loop thru obstacle_positions

    marker_arr = ['h', 's', 'X']
    color_arr = ['green', 'brown', 'purple']
    for idx, (cls, positions) in enumerate(obstacle_positions.items()):
        print("Class: ", cls, " Positions: ", positions)
        marker = marker_arr[idx]
        color = color_arr[idx]

        for idx, pos in enumerate(positions):
            if (idx == 0):        
                axes[0].plot(pos[1], pos[0], marker=marker, color=color, label=f'{cls}')
            else:
                axes[0].plot(pos[1], pos[0], marker=marker, color=color)
    
    start = np.random.randint(low=0, high = 5,size=(2,))

    axes[0].plot(start[1], start[0], marker='d', label = f'Start')

    #MCP = graph.MCP(fused, fully_connected=False)

    #Obtain path
    path, cost = graph.route_through_array(fused, start, goal, fully_connected=True, geometric=False)
    
    path = np.array(path)
    path_rows = path[:,0]
    path_cols = path[:, 1]

    path_rows, path_cols = clamped_bspline_path(path_rows, path_cols,n=300, tangent_scale=0.0)

    axes[0].plot(path_cols,path_rows, color='red', linewidth=2)

    

    axes[0].legend(loc='upper right')
    plt.show()
