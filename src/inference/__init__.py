"""
Unified inference module for CellSAM.

Import key functions from submodules for easy access:
    from inference import run_sam_inference, mask_to_rgb, smooth_boundary
"""
from .pipeline import (
    load_model,
    run_sam_inference,
    visualize_results,
    normalize_channel,
)

from .postprocess import (
    smooth_boundary,
    keep_largest_component,
    validate_cell_size,
    postprocess_cell,
    postprocess_instance_mask,
    MIN_CELL_AREA,
    MAX_CELL_AREA,
)

from .visualize import (
    mask_to_rgb,
    create_overlay,
    get_high_contrast_colormap,
    build_adjacency_graph,
    graph_coloring,
)

__all__ = [
    # Pipeline
    'load_model',
    'run_sam_inference',
    'visualize_results',
    'normalize_channel',
    # Postprocess
    'smooth_boundary',
    'keep_largest_component',
    'validate_cell_size',
    'postprocess_cell',
    'postprocess_instance_mask',
    'MIN_CELL_AREA',
    'MAX_CELL_AREA',
    # Visualize
    'mask_to_rgb',
    'create_overlay',
    'get_high_contrast_colormap',
    'build_adjacency_graph',
    'graph_coloring',
]
