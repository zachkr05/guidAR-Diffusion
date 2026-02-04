"""
Utilities for IRL-based diffusion costmap finetuning.
"""

from .finetune_focused import (
    finetune_models_focused,
    compute_path_difference_mask,
    make_path_target,
    gaussian_blur,
)

from .planner import SoftGridPlanner

from .spline import (
    find_span,
    basis_funs,
    bspline_design_matrix,
    clamped_knots,
    reparam_curve,
    generate_clamped_spline,
    DraggableBSpline,
)

from .utils import (
    collate_ignore_metadata,
    compute_path_from_costmap,
    path_length,
    curvature_penalty,
    length_ratio_penalty,
    trajectory_cost,
    pairwise_dist,
    hausdorff_distance,
    discrete_frechet_distance,
    get_user_adjustments,
    visualize_3d,
    fuse_costmaps,
    cosine_beta_schedule,
    identify_classes,

    )

__all__ = [
    # IRL finetuning
    'finetune_models_focused',
    'compute_path_difference_mask',
    'make_path_target',
    'gaussian_blur',
    
    # Planner
    'SoftGridPlanner',
    
    # B-spline utilities
    'find_span',
    'basis_funs',
    'bspline_design_matrix',
    'clamped_knots',
    'reparam_curve',
    'generate_clamped_spline',
    'DraggableBSpline',
    
    # General utilities
    'collate_ignore_metadata',
    'compute_path_from_costmap',
    'path_length',
    'curvature_penalty',
    'length_ratio_penalty',
    'trajectory_cost',
    'pairwise_dist',
    'hausdorff_distance',
    'discrete_frechet_distance',
    'get_user_adjustments',
    'visualize_3d',
    'fuse_costmaps',
    'cosine_beta_schedule',
    'identify_classes']
