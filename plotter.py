

from plotly.subplots import make_subplots


from DataGenerator.sim import NUM_CLASSES, OBSTACLE_CLASSES, NUM_ORIENTATIONS, orientation_to_vector

import argparse
import os
import numpy as np
import torch
import plotly.graph_objects as go
from skimage.graph import route_through_array





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
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        name='Original', line=dict(color='white', width=3)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[start_pos[1]], y=[start_pos[0]], mode='markers',
        marker=dict(symbol='circle', size=14, color='lime'), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[goal_pos[1]], y=[goal_pos[0]], mode='markers',
        marker=dict(symbol='star', size=16, color='yellow'), showlegend=False
    ), row=1, col=1)
    
    # Plot 2: User corrected trajectory
    fig.add_trace(go.Heatmap(z=costmap, colorscale='Viridis', zmin=0, zmax=1, showscale=False), row=1, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=1, col=2)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        name='Original', line=dict(color='white', width=2, dash='dot'), showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        name='User Corrected', line=dict(color='cyan', width=3)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
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
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=orig_arr[:, 1], y=orig_arr[:, 0], mode='lines',
        line=dict(color='black', width=1, dash='dot'), showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        line=dict(color='black', width=2), showlegend=False
    ), row=2, col=1)
    
    # Plot 4: Interaction mask
    fig.add_trace(go.Heatmap(
        z=interaction_mask, colorscale='Hot', zmin=0, zmax=1,
        showscale=True, colorbar=dict(title="Mask", x=1.0, len=0.4, y=0.15)
    ), row=2, col=2)
    add_obstacle_markers(fig, obstacles_by_class, row=2, col=2)
    fig.add_trace(go.Scatter(
        x=adj_arr[:, 1], y=adj_arr[:, 0], mode='lines',
        line=dict(color='cyan', width=2), showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(height=700, width=900, title_text=title)
    fig.update_yaxes(autorange='reversed')
    fig.show()
    
    return fig


def add_obstacle_markers(fig, obstacles_by_class, row=None, col=None):
    """Add obstacle markers with orientation arrows."""
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
            name=class_name,
            marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color='black'))
        )
        arrow_kwargs = dict(
            x=arrow_x, y=arrow_y, mode='lines', showlegend=False,
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
    add_obstacle_markers(fig, obstacles_by_class)
    
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


