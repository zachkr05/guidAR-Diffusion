from plotly.subplots import make_subplots
from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES, NUM_ORIENTATIONS, orientation_to_vector

import numpy as np
import plotly.graph_objects as go


def add_obstacle_markers(fig, obstacles_by_class, row=None, col=None, show_legend=True):
    """Add obstacle markers with orientation arrows and class labels."""
    symbols = ['circle', 'square', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'orange']
    arrow_length = 4.0
    
    for class_id, obstacles in obstacles_by_class.items():
        if len(obstacles) == 0:
            continue
        color = colors[class_id % len(colors)]
        symbol = symbols[class_id % len(symbols)]
        class_name = OBSTACLE_CLASSES[class_id]['name']
        
        rows_list, cols_list = [], []
        arrow_x, arrow_y = [], []
        
        for obs in obstacles:
            r, c = obs['pos']
            orientation = obs['orientation']
            rows_list.append(r)
            cols_list.append(c)
            dx, dy = orientation_to_vector(orientation)
            arrow_x.extend([c, c + dx * arrow_length, None])
            arrow_y.extend([r, r - dy * arrow_length, None])
        
        marker_kwargs = dict(
            x=cols_list, y=rows_list, mode='markers',
            name=f"{class_name} (class {class_id})",
            showlegend=show_legend,
            legendgroup=f"class_{class_id}",
            marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color='black'))
        )
        arrow_kwargs = dict(
            x=arrow_x, y=arrow_y, mode='lines', showlegend=False,
            legendgroup=f"class_{class_id}",
            line=dict(color=color, width=2)
        )
        
        if row and col:
            fig.add_trace(go.Scatter(**marker_kwargs), row=row, col=col)
            if arrow_x:
                fig.add_trace(go.Scatter(**arrow_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**marker_kwargs))
            if arrow_x:
                fig.add_trace(go.Scatter(**arrow_kwargs))


def visualize_comparison(costmap, orig_path, adj_path, delta_map, interaction_mask,
                         obstacles_by_class, start_pos, goal_pos, title="Base vs User Corrected"):
    """Visualize original trajectory, user correction, delta, and mask."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Base Model + Original Trajectory",
            "Base Model + User Corrected Trajectory", 
            "Geometric Delta (IRL Signal)",
            "Interaction Mask (Learning Zone)"
        )
    )
    
    orig_arr = np.array(orig_path)
    adj_arr = np.array(adj_path)
    
    # Plot 1: Original trajectory
    fig.add_trace(go.Heatmap(z=costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=1, show_legend=True)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        name='Original Path', line=dict(color='white', width=3)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]], mode='markers',
        name='Start', marker=dict(symbol='circle', size=14, color='lime')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]], y=[goal_pos[0]], mode='markers',
        name='Goal', marker=dict(symbol='star', size=16, color='yellow')
    ), row=1, col=1)
    
    # Plot 2: User corrected trajectory
    fig.add_trace(go.Heatmap(z=costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=2, show_legend=False)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        name='Original', line=dict(color='white', width=2, dash='dot'), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        name='User Corrected', line=dict(color='cyan', width=3)
    ), row=1, col=2)
    fig.ad_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]], mode='markers',
        marker=dict(symbol='circle', size=14, color='lime'), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]], y=[goal_pos[0]], mode='markers',
        marker=dict(symbol='star', size=16, color='yellow'), showlegend=False
    ), row=1, col=2)
    
    # Plot 3: Geometric delta
    max_delta = max(abs(delta_map.min()), abs(delta_map.max()), 0.1)
    fig.add_trace(go.Heatmap(
        z=delta_map, colorscale='RdBu_r', zmin=-max_delta, zmax=max_delta, 
        showscale=True, colorbar=dict(title="Delta", x=0.45, len=0.4, y=0.15)
    ), row=2, col=1)
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=1, show_legend=False)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        line=dict(color='black', width=1, dash='dot'), showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        line=dict(color='black', width=2), showlegend=False
    ), row=2, col=1)
    
    # Plot 4: Interaction mask
    if interaction_mask is not None:
        fig.add_trace(go.Heatmap(
            z=interaction_mask, colorscale='Hot', zmin=0, zmax=1,
            showscale=True, colorbar=dict(title="Mask", x=1.0, len=0.4, y=0.15)
        ), row=2, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=2, show_legend=False)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        line=dict(color='cyan', width=2), showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(height=700, width=900, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()
    
    return fig


def visualize_costmap_and_trajectory(costmap, path, obstacles_by_class, start_pos, goal_pos, title="Costmap & Trajectory"):
    """Visualize costmap with trajectory overlay."""
    fig = go.Figure()
    
    # Costmap
    fig.add_trace(go.Heatmap(
        z=costmap, 
        colorscale='Viridis', 
        zmin=0, zmax=1,
        colorbar=dict(title="Cost")
    ))
    
    # Obstacles
    add_obstacle_markers(fig, obstacles_by_class, show_legend=True)
    
    # Trajectory
    path_arr = np.array(path)
    fig.add_trace(go.Scatter(
        x=path_arr[:, 1], y=path_arr[:, 0],
        mode='lines',
        name='Trajectory',
        line=dict(color='cyan', width=3)
    ))
    
    # Start
    fig.add_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]],
        mode='markers',
        name='Start',
        marker=dict(symbol='circle', size=16, color='lime', line=dict(width=2, color='white'))
    ))
    
    # Goal
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]], y=[goal_pos[0]],
        mode='markers',
        name='Goal',
        marker=dict(symbol='star', size=18, color='yellow', line=dict(width=2, color='black'))
    ))
    
    fig.update_layout(height=600, width=700, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()


def visualize_learning_step(pre_map, post_map, orig_path, adj_path, delta, obstacles_by_class, 
                            scene_idx, affected_classes=None):
    """
    Side-by-side comparison of Before vs After Training with improvement metrics.
    
    Shows:
    1. Before training costmap + original path
    2. User correction delta
    3. After training costmap + new path
    4. Difference map (post - pre) showing what the model learned
    """
    from skimage.graph import route_through_array
    
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            "Before Training", 
            "User Correction (Target Delta)", 
            "After Training",
            "Model Change (Post - Pre)"
        )
    )
    
    # Build affected classes string for title
    if affected_classes:
        class_names = [OBSTACLE_CLASSES[c]['name'] for c in affected_classes]
        affected_str = ", ".join(class_names)
    else:
        affected_str = "unknown"
    
    # 1. Before Training
    fig.add_trace(go.Heatmap(z=pre_map, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=1, show_legend=True)
    path_arr = np.array(orig_path)
    fig.add_trace(go.Scatter(
        x=path_arr[:,1], y=path_arr[:,0], mode='lines', 
        name='Original Path', line=dict(color='red', width=3)
    ), row=1, col=1)
    
    # 2. User Correction Delta
    max_delta = max(abs(delta.min()), abs(delta.max()), 0.1)
    fig.add_trace(go.Heatmap(
        z=delta, colorscale='RdBu_r', zmin=-max_delta, zmax=max_delta, 
        showscale=True, colorbar=dict(title="Delta", x=0.45, len=0.4, y=0.78)
    ), row=1, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=2, show_legend=False)
    path_adj = np.array(adj_path)
    fig.add_trace(go.Scatter(
        x=path_arr[:,1], y=path_arr[:,0], mode='lines', 
        line=dict(color='red', width=2, dash='dot'), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=path_adj[:,1], y=path_adj[:,0], mode='lines', 
        name='User Corrected', line=dict(color='green', width=3)
    ), row=1, col=2)

    # 3. After Training
    fig.add_trace(go.Heatmap(z=post_map, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=2, col=1)
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=1, show_legend=False)
    
    # Plan new path on post-training map
    try:
        start_pos = orig_path[0]
        goal_pos = orig_path[-1]
        new_path, _ = route_through_array(post_map, start_pos, goal_pos, fully_connected=True, geometric=True)
        new_path_arr = np.array(new_path)
        fig.add_trace(go.Scatter(
            x=new_path_arr[:,1], y=new_path_arr[:,0], mode='lines', 
            name='New Path (Post-Training)', line=dict(color='cyan', width=3)
        ), row=2, col=1)
    except:
        # If path planning fails, just show the adjusted path
        fig.add_trace(go.Scatter(
            x=path_adj[:,1], y=path_adj[:,0], mode='lines', 
            name='Target Path', line=dict(color='cyan', width=3)
        ), row=2, col=1)
    
    # 4. Difference Map (what the model learned)
    diff_map = post_map - pre_map
    max_diff = max(abs(diff_map.min()), abs(diff_map.max()), 0.01)
    fig.add_trace(go.Heatmap(
        z=diff_map, colorscale='RdBu_r', zmin=-max_diff, zmax=max_diff,
        showscale=True, colorbar=dict(title="Δ Cost", x=1.0, len=0.4, y=0.22)
    ), row=2, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=2, show_legend=False)
    
    # Add stats annotation
    stats_text = (
        f"Pre  - min: {pre_map.min():.3f}, max: {pre_map.max():.3f}, std: {pre_map.std():.3f}<br>"
        f"Post - min: {post_map.min():.3f}, max: {post_map.max():.3f}, std: {post_map.std():.3f}<br>"
        f"Diff - min: {diff_map.min():.3f}, max: {diff_map.max():.3f}, std: {diff_map.std():.3f}"
    )
    
    fig.update_layout(
        title_text=f"Online Learning: Scene {scene_idx} | Affected: {affected_str}",
        height=700, 
        width=900,
        annotations=[
            dict(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.08,
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
        ]
    )
    fig.update_yaxes(autorange='reversed')
    fig.show()
    
    return fig


def visualize_generated_vs_target(generated, target, path, obstacles_by_class, start_pos, goal_pos, title="Generated vs Ground Truth"):
    """Compare model output to ground truth."""
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Generated", "Ground Truth (x0)", "Difference")
    )
    
    # Generated
    fig.add_trace(go.Heatmap(z=generated, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=1)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=1, show_legend=True)
    
    # Target
    fig.add_trace(go.Heatmap(z=target, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=2, show_legend=False)
    
    # Difference
    diff = generated - target
    max_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
    fig.add_trace(go.Heatmap(
        z=diff, colorscale='RdBu_r', zmin=-max_diff, zmax=max_diff,
        showscale=True, colorbar=dict(title="Diff")
    ), row=1, col=3)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=3, show_legend=False)
    
    # Add path to first two
    path_arr = np.array(path)
    for col in [1, 2]:
        fig.add_trace(go.Scatter(
            x=path_arr[:, 1], y=path_arr[:, 0], mode='lines',
            line=dict(color='cyan', width=3), showlegend=False
        ), row=1, col=col)
        
        fig.add_trace(go.Scatter(
            x=[start_pos[1]], y=[start_pos[0]], mode='markers',
            marker=dict(symbol='circle', size=12, color='lime'), showlegend=False
        ), row=1, col=col)
        
        fig.add_trace(go.Scatter(
            x=[goal_pos[1]], y=[goal_pos[0]], mode='markers',
            marker=dict(symbol='star', size=14, color='yellow'), showlegend=False
        ), row=1, col=col)
    
    # Stats
    stats_text = (
        f"Generated - min: {generated.min():.3f}, max: {generated.max():.3f}, std: {generated.std():.3f} | "
        f"Target - min: {target.min():.3f}, max: {target.max():.3f}, std: {target.std():.3f}"
    )
    
    fig.update_layout(
        height=400, width=1000, 
        title_text=title,
        annotations=[
            dict(text=stats_text, xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False, font=dict(size=10))
        ]
    )
    fig.update_yaxes(autorange='reversed')
    fig.show()
    
    return fig
