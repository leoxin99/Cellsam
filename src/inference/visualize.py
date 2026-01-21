"""
Unified visualization module for CellSAM inference.

This module contains visualization functions with graph coloring
to ensure distinct colors for adjacent cells.
"""
import numpy as np
from scipy import ndimage


def get_high_contrast_colormap():
    """Return a high-contrast color palette for cell visualization."""
    return np.array([
        [0, 0, 0],          # 0: Background (black)
        [255, 50, 50],      # 1: Bright Red
        [50, 255, 50],      # 2: Bright Green
        [50, 50, 255],      # 3: Bright Blue
        [255, 255, 50],     # 4: Yellow
        [255, 50, 255],     # 5: Magenta
        [50, 255, 255],     # 6: Cyan
        [255, 150, 50],     # 7: Orange
        [150, 50, 255],     # 8: Purple
        [50, 200, 150],     # 9: Teal
        [200, 100, 100],    # 10: Salmon
        [100, 200, 100],    # 11: Light Green
        [100, 100, 200],    # 12: Light Blue
    ], dtype=np.uint8)


def build_adjacency_graph(mask: np.ndarray) -> dict:
    """
    Build adjacency graph from instance mask.
    
    Two cells are considered adjacent if their boundaries are within 2 pixels.
    
    Args:
        mask: Instance segmentation mask
    
    Returns:
        Dictionary mapping cell_id to set of adjacent cell_ids
    """
    unique_ids = [i for i in np.unique(mask) if i > 0]
    if len(unique_ids) == 0:
        return {}
    
    adjacency = {i: set() for i in unique_ids}
    
    for cell_id in unique_ids:
        cell_mask = (mask == cell_id)
        # Dilate by 2 pixels to find neighbors
        dilated = ndimage.binary_dilation(cell_mask, iterations=2)
        neighbor_ids = np.unique(mask[dilated & ~cell_mask])
        
        for neighbor in neighbor_ids:
            if neighbor > 0 and neighbor != cell_id:
                adjacency[cell_id].add(neighbor)
                adjacency[neighbor].add(cell_id)
    
    return adjacency


def graph_coloring(adjacency: dict, num_colors: int = 12) -> dict:
    """
    Greedy graph coloring algorithm.
    
    Assigns the smallest available color not used by neighbors.
    Uses 4-color theorem principle: adjacent regions get different colors.
    
    Args:
        adjacency: Adjacency graph (dict of cell_id -> set of neighbors)
        num_colors: Number of available colors
    
    Returns:
        Dictionary mapping cell_id to color index (1-based)
    """
    color_assignment = {}
    
    for cell_id in sorted(adjacency.keys()):
        # Colors used by adjacent cells
        used_colors = {color_assignment.get(neighbor) for neighbor in adjacency[cell_id]}
        used_colors.discard(None)
        
        # Find first available color
        for color in range(1, num_colors + 1):
            if color not in used_colors:
                color_assignment[cell_id] = color
                break
        else:
            # Fallback: use modulo if we run out of colors
            color_assignment[cell_id] = (cell_id % num_colors) + 1
    
    return color_assignment


def mask_to_rgb(mask: np.ndarray, cmap: np.ndarray = None) -> np.ndarray:
    """
    Convert instance mask to RGB using graph coloring.
    
    Ensures adjacent cells have distinct colors using greedy graph coloring
    based on the 4-color theorem.
    
    Args:
        mask: Instance segmentation mask
        cmap: Optional custom colormap (Nx3 uint8 array)
    
    Returns:
        RGB image (H, W, 3) uint8
    """
    if cmap is None:
        cmap = get_high_contrast_colormap()
    
    unique_ids = [i for i in np.unique(mask) if i > 0]
    if len(unique_ids) == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Build adjacency and assign colors
    adjacency = build_adjacency_graph(mask)
    color_assignment = graph_coloring(adjacency, num_colors=len(cmap) - 1)
    
    # Apply colors
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cell_id, color_idx in color_assignment.items():
        rgb[mask == cell_id] = cmap[color_idx]
    
    return rgb


def create_overlay(image: np.ndarray, mask: np.ndarray, 
                   alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of mask on grayscale image.
    
    Args:
        image: Grayscale image (H, W), values 0-1 or 0-255
        mask: Instance mask
        alpha: Transparency of the mask overlay
    
    Returns:
        RGB overlay image
    """
    # Normalize image to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    
    # Get colored mask
    mask_rgb = mask_to_rgb(mask)
    
    # Create overlay
    overlay = image_rgb.astype(float) * (1 - alpha) + mask_rgb.astype(float) * alpha
    return overlay.astype(np.uint8)
