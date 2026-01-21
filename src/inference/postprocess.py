"""
Unified post-processing module for CellSAM inference.

This module contains all post-processing functions used after SAM segmentation,
ensuring consistency across different inference scripts.
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure


# Cell size thresholds based on GT analysis of FULL dataset (478 images, 5173 cells)
# Min: 6240 (may include annotation errors), P1: 40836, Median: 142316, P99: 513928, Max: 1026328
# Using P1/P99 to exclude potential annotation errors (holes, debris)
MIN_CELL_AREA = 40836    # P1 of GT distribution
MAX_CELL_AREA = 513928   # P99 of GT distribution


def smooth_boundary(pred_binary: np.ndarray, sigma_first: int = 7, 
                    sigma_second: int = 5, disk_size: int = 8) -> np.ndarray:
    """
    6-step enhanced boundary smoothing pipeline.
    
    Goal: Match GT-quality smooth curved boundaries, eliminate spike-like artifacts.
    
    Args:
        pred_binary: Binary segmentation mask
        sigma_first: Gaussian sigma for first pass (default 7)
        sigma_second: Gaussian sigma for second pass (default 5)
        disk_size: Morphological disk size for opening/closing (default 8)
    
    Returns:
        Smoothed binary mask
    """
    pred = pred_binary.astype(bool)
    
    # Step 1: Initial morphological cleanup
    pred = morphology.binary_closing(pred, morphology.disk(5))
    pred = ndimage.binary_fill_holes(pred)
    pred = morphology.remove_small_objects(pred, min_size=500)
    
    # Step 2: Strong Gaussian smoothing
    smoothed = gaussian_filter(pred.astype(float), sigma=sigma_first)
    pred = smoothed > 0.5
    
    # Step 3: Remove spike-like protrusions
    pred = morphology.binary_opening(pred, morphology.disk(disk_size))
    
    # Step 4: Restore smooth shape
    pred = morphology.binary_closing(pred, morphology.disk(disk_size))
    
    # Step 5: Final cleanup
    pred = ndimage.binary_fill_holes(pred)
    pred = morphology.remove_small_objects(pred, min_size=500)
    
    # Step 6: Second Gaussian pass for extra smoothness
    smoothed = gaussian_filter(pred.astype(float), sigma=sigma_second)
    pred = smoothed > 0.5
    pred = ndimage.binary_fill_holes(pred)
    
    return pred.astype(bool)


def keep_largest_component(pred_binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    labeled = measure.label(pred_binary)
    if labeled.max() == 0:
        return pred_binary
    
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return (labeled == largest.label).astype(bool)


def validate_cell_size(pred_binary: np.ndarray, 
                       min_area: int = MIN_CELL_AREA,
                       max_area: int = MAX_CELL_AREA) -> bool:
    """
    Validate if cell size is within acceptable range.
    
    Args:
        pred_binary: Binary mask of the cell
        min_area: Minimum acceptable area (default from GT P1)
        max_area: Maximum acceptable area (default from GT P99+)
    
    Returns:
        True if cell is valid, False otherwise
    """
    area = pred_binary.sum()
    return min_area <= area <= max_area


def postprocess_cell(pred_binary: np.ndarray, 
                     validate_size: bool = True) -> np.ndarray | None:
    """
    Complete post-processing pipeline for a single cell.
    
    Args:
        pred_binary: Raw binary prediction from SAM
        validate_size: Whether to validate cell size
    
    Returns:
        Processed binary mask, or None if cell is invalid
    """
    # Step 1: Smooth boundaries
    pred = smooth_boundary(pred_binary)
    
    # Step 2: Keep largest component
    pred = keep_largest_component(pred)
    
    # Step 3: Validate size
    if validate_size and not validate_cell_size(pred):
        return None
    
    return pred.astype(bool)


def postprocess_instance_mask(instance_mask: np.ndarray) -> np.ndarray:
    """
    Post-process an entire instance mask (all cells together).
    
    Note: This applies smoothing to each instance separately.
    
    Args:
        instance_mask: Instance segmentation mask with cell IDs
    
    Returns:
        Processed instance mask
    """
    unique_ids = [i for i in np.unique(instance_mask) if i > 0]
    result = np.zeros_like(instance_mask)
    new_id = 0
    
    for cell_id in unique_ids:
        cell_binary = (instance_mask == cell_id)
        processed = postprocess_cell(cell_binary, validate_size=True)
        
        if processed is not None:
            new_id += 1
            result[processed] = new_id
    
    return result
