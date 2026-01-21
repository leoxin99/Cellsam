"""
DAPI-based nucleus detection and box generation for CellSAM.

This module consolidates all DAPI detection logic including:
- Nucleus detection
- Smart dual-nuclei merging (size-relative threshold)
- Bounding box generation with anisotropic expansion
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage import morphology, measure, filters


def detect_nuclei(dapi_channel: np.ndarray, 
                  min_area: int = 500, 
                  max_area: int = 30000) -> list:
    """
    Detect nuclei from DAPI channel using Otsu thresholding.
    
    Args:
        dapi_channel: DAPI fluorescence image
        min_area: Minimum nucleus area in pixels
        max_area: Maximum nucleus area in pixels
    
    Returns:
        List of regionprops for detected nuclei
    """
    # Normalize
    p2, p98 = np.percentile(dapi_channel, [2, 98])
    if p98 > p2:
        img_norm = np.clip((dapi_channel - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(dapi_channel, dtype=np.float32)
    
    # Threshold
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    # Label and filter by size
    labels = measure.label(binary)
    regions = [r for r in measure.regionprops(labels) 
               if min_area <= r.area <= max_area]
    
    return regions


def merge_close_nuclei(regions: list, 
                       size_ratio_threshold: float = 3.0,
                       use_relative_distance: bool = True,
                       fixed_merge_distance: int = 100) -> list:
    """
    Merge nearby nuclei that belong to the same binucleated cell.
    
    Uses smart size-relative merging:
    1. Relative distance: 1.5 * average nucleus diameter
    2. Size similarity: Only merge nuclei of similar sizes
    
    Args:
        regions: List of regionprops for detected nuclei
        size_ratio_threshold: Max ratio between nucleus sizes (default 3.0)
        use_relative_distance: Use size-relative threshold vs fixed
        fixed_merge_distance: Fixed distance if not using relative
    
    Returns:
        List of cell groups, each group is a list of regions
    """
    if len(regions) <= 1:
        return [[r] for r in regions]
    
    centroids = np.array([r.centroid for r in regions])
    # Estimate diameter from area
    diameters = np.array([2 * np.sqrt(r.area / np.pi) for r in regions])
    n = len(centroids)
    distances = cdist(centroids, centroids)
    
    # Union-Find for merging
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            # Size similarity check
            size_ratio = max(diameters[i], diameters[j]) / (min(diameters[i], diameters[j]) + 1e-6)
            if size_ratio > size_ratio_threshold:
                continue
            
            # Distance threshold
            if use_relative_distance:
                avg_diameter = (diameters[i] + diameters[j]) / 2
                max_merge_dist = 1.5 * avg_diameter
            else:
                max_merge_dist = fixed_merge_distance
            
            if distances[i, j] < max_merge_dist:
                union(i, j)
    
    # Group by root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return [[regions[i] for i in indices] for indices in groups.values()]


def is_on_edge(region, image_shape: tuple, margin: int = 30) -> bool:
    """Check if a region touches the image edge."""
    y1, x1, y2, x2 = region.bbox
    h, w = image_shape
    return x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin


def create_bounding_boxes(cell_groups: list,
                          image_shape: tuple,
                          expansion_long: float = 5.0,
                          expansion_short: float = 3.0,
                          expansion_isotropic: float = 4.0,
                          round_threshold: float = 1.3,
                          exclude_edges: bool = True,
                          margin: int = 30) -> list:
    """
    Create bounding boxes from nucleus groups with smart anisotropic expansion.
    
    For elongated nuclei: expand more along the major axis (cells are elongated)
    For round nuclei: expand equally in all directions
    
    Args:
        cell_groups: List of cell groups from merge_close_nuclei
        image_shape: (H, W) of the image
        expansion_long: Expansion factor along major axis
        expansion_short: Expansion factor along minor axis
        expansion_isotropic: Expansion for round nuclei
        round_threshold: Aspect ratio threshold for "round" classification
        exclude_edges: Whether to exclude edge-touching nuclei
        margin: Edge margin in pixels
    
    Returns:
        List of boxes [[x1, y1, x2, y2], ...]
    """
    boxes = []
    h, w = image_shape
    
    for group in cell_groups:
        # Skip edge nuclei
        if exclude_edges and any(is_on_edge(r, image_shape, margin) for r in group):
            continue
        
        # Get combined bounding box of group
        all_coords = []
        for r in group:
            coords = r.coords  # (N, 2) array of (row, col)
            all_coords.append(coords)
        all_coords = np.concatenate(all_coords, axis=0)
        
        # Nucleus bounding box
        y_min, x_min = all_coords.min(axis=0)
        y_max, x_max = all_coords.max(axis=0)
        nuc_h = y_max - y_min
        nuc_w = x_max - x_min
        
        # Center
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Determine expansion based on shape
        aspect = max(nuc_w, nuc_h) / (min(nuc_w, nuc_h) + 1e-6)
        
        if aspect < round_threshold:
            # Round nucleus: isotropic expansion
            expand_x = expand_y = expansion_isotropic
        else:
            # Elongated: anisotropic based on orientation
            if nuc_w > nuc_h:
                expand_x = expansion_long
                expand_y = expansion_short
            else:
                expand_x = expansion_short
                expand_y = expansion_long
        
        # Create expanded box
        box_w = nuc_w * expand_x
        box_h = nuc_h * expand_y
        
        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes


def detect_and_create_boxes(dapi_channel: np.ndarray,
                            min_nucleus_area: int = 500,
                            max_nucleus_area: int = 30000,
                            **box_kwargs) -> tuple:
    """
    Complete pipeline: detect nuclei → merge → create boxes.
    
    Args:
        dapi_channel: DAPI fluorescence image
        min_nucleus_area: Min nucleus area for filtering
        max_nucleus_area: Max nucleus area for filtering
        **box_kwargs: Additional arguments for create_bounding_boxes
    
    Returns:
        Tuple of (boxes, cell_groups, all_regions)
    """
    # Step 1: Detect nuclei
    regions = detect_nuclei(dapi_channel, min_nucleus_area, max_nucleus_area)
    
    # Step 2: Merge close nuclei
    cell_groups = merge_close_nuclei(regions)
    
    # Step 3: Create boxes
    image_shape = dapi_channel.shape
    boxes = create_bounding_boxes(cell_groups, image_shape, **box_kwargs)
    
    return boxes, cell_groups, regions
