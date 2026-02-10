"""
Unified inference module for CellSAM.

Primary API (Phase 0+):
    from inference.core import segment_with_boxes, InferenceConfig, load_cellsam_checkpoint

Legacy API (deprecated, lazy-loaded):
    from inference import run_sam_inference, load_model
"""

# === Primary: unified core (always available) ===
from .core import (
    segment_with_boxes,
    resolve_conflicts,
    InferenceConfig,
    InferenceResult,
    load_cellsam_checkpoint,
)

# === Postprocess ===
from .postprocess import (
    smooth_boundary,
    keep_largest_component,
    validate_cell_size,
    postprocess_cell,
    postprocess_instance_mask,
    MIN_CELL_AREA,
    MAX_CELL_AREA,
)

# === Visualize ===
from .visualize import (
    mask_to_rgb,
    create_overlay,
    get_high_contrast_colormap,
    build_adjacency_graph,
    graph_coloring,
)


def __getattr__(name):
    """Lazy-load legacy pipeline to avoid importing heavy dependencies when unused."""
    _legacy_names = {
        'load_model', 'run_sam_inference', 'visualize_results', 'normalize_channel',
        # these were also in old __init__ from core, keep for compat
        'get_default_config', 'compute_intrusion_rate', 'compute_conflict_rate',
    }
    if name in {'load_model', 'run_sam_inference', 'visualize_results', 'normalize_channel'}:
        import warnings
        warnings.warn(
            f"inference.{name} uses legacy pipeline (first_write conflict). "
            "Use inference.core.segment_with_boxes instead.",
            DeprecationWarning, stacklevel=2
        )
        from . import pipeline
        return getattr(pipeline, name)
    if name in {'get_default_config', 'compute_intrusion_rate', 'compute_conflict_rate'}:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module 'inference' has no attribute {name!r}")


__all__ = [
    # Core (unified, Phase 0+)
    'segment_with_boxes',
    'resolve_conflicts',
    'InferenceConfig',
    'InferenceResult',
    'load_cellsam_checkpoint',
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
