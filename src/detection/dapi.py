"""
dapi.py

功能: DAPI 核检测与边界框生成
所属: 核心代码 (src/detection/)
创建日期: 2026-01-XX
最后修改: 2026-01-27
版本: v4 (验证完成)

主要函数:
1. detect_nuclei() - DAPI 核检测 (Otsu + 形态学)
2. merge_close_nuclei() - 双核合并 (Union-Find)
3. create_bounding_boxes() - 智能框扩展 (各向异性)
4. filter_by_actn2() - Actn2 心肌细胞过滤
5. detect_cardiomyocytes() - 完整 DAPI+Actn2 检测流程
6. detect_with_adaptive_box() - Z-线引导自适应框
7. detect_zlines_in_region() - blob_log Z-线检测
8. get_cell_zlines() - 距离过滤 Z-线
9. create_adaptive_box() - 自适应框生成

更新日志:
- 2026-01-23 v3: 添加 detect_with_adaptive_box, Z-线检测系列函数
- 2026-01-23 v2: 添加 filter_by_actn2, detect_cardiomyocytes
- 2026-01-XX v1: 初始版本 (detect_nuclei, merge_close_nuclei, create_bounding_boxes)
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage import morphology, measure, filters


def detect_nuclei(dapi_channel: np.ndarray, 
                  min_area: int = 200,   # Updated for 1024px (2026-02-05)
                  max_area: int = 10000) -> list:  # Updated for 1024px
    """
    Detect nuclei from DAPI channel using Otsu thresholding.
    
    Args:
        dapi_channel: DAPI fluorescence image
        min_area: Minimum nucleus area in pixels (P1 at 1024px: 57, using 200 for safety)
        max_area: Maximum nucleus area in pixels (P99 at 1024px: 10026)
    
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
                       fixed_merge_distance: int = 373,
                       merge_coeff: float = 1.2) -> list:
    """
    Merge nearby nuclei that belong to the same binucleated cell.
    
    Uses smart size-relative merging:
    1. Relative distance: 1.2 * average nucleus diameter (default)
    2. Size similarity: Only merge nuclei of similar sizes (ratio < 3.0)
    
    Args:
        regions: List of regionprops for detected nuclei
        size_ratio_threshold: Max ratio between nucleus sizes (default 3.0)
        use_relative_distance: Use size-relative threshold vs fixed (default True)
        fixed_merge_distance: Fixed distance threshold if not using relative
        merge_coeff: Relative-distance multiplier for nucleus diameter (default 1.2)
    
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
                max_merge_dist = merge_coeff * avg_diameter
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


def is_on_edge(region, image_shape: tuple, margin: int = 20) -> bool:  # Updated 2026-02-05: based on P5 edge dist=6
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
                          margin: int = 32) -> list:  # Updated 2026-02-05: scaled from 50 @ 1608->32 @ 1024
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
        margin: Edge margin in pixels (32px @ 1024, verified 2026-02-05)
    
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
                            min_nucleus_area: int = 200,    # Updated 2026-02-05: 1024px
                            max_nucleus_area: int = 10000,  # Updated 2026-02-05: 1024px
                            size_ratio_threshold: float = 3.0,
                            use_relative_distance: bool = True,
                            fixed_merge_distance: int = 373,
                            merge_coeff: float = 1.2,
                            **box_kwargs) -> tuple:
    """
    Complete pipeline: detect nuclei → merge → create boxes.
    
    Args:
        dapi_channel: DAPI fluorescence image
        min_nucleus_area: Min nucleus area for filtering
        max_nucleus_area: Max nucleus area for filtering
        size_ratio_threshold: Max ratio between nucleus sizes for merging
        use_relative_distance: Use size-relative merge threshold
        fixed_merge_distance: Fixed merge distance when relative mode is disabled
        merge_coeff: Relative merge multiplier for average nucleus diameter
        **box_kwargs: Additional arguments for create_bounding_boxes
    
    Returns:
        Tuple of (boxes, cell_groups, all_regions)
    """
    # Step 1: Detect nuclei
    regions = detect_nuclei(dapi_channel, min_nucleus_area, max_nucleus_area)
    
    # Step 2: Merge close nuclei
    cell_groups = merge_close_nuclei(
        regions,
        size_ratio_threshold=size_ratio_threshold,
        use_relative_distance=use_relative_distance,
        fixed_merge_distance=fixed_merge_distance,
        merge_coeff=merge_coeff,
    )
    
    # Step 3: Create boxes
    image_shape = dapi_channel.shape
    boxes = create_bounding_boxes(cell_groups, image_shape, **box_kwargs)
    
    return boxes, cell_groups, regions


def filter_by_actn2(cell_groups: list, 
                    actn2_channel: np.ndarray,
                    coverage_threshold: float = 0.3,
                    intensity_percentile: float = 10.0,
                    expansion_factor: float = 1.5) -> list:
    """
    Filter cell groups to keep only cardiomyocytes based on Actn2 signal.
    
    A cell is considered a cardiomyocyte if the Actn2 signal around its 
    nucleus region exceeds a threshold coverage.
    
    Args:
        cell_groups: List of cell groups from merge_close_nuclei
        actn2_channel: α-actinin fluorescence image
        coverage_threshold: Minimum fraction of region with Actn2 signal (default 0.3)
        intensity_percentile: Percentile threshold for "positive" signal (default 10)
        expansion_factor: Expand nucleus region by this factor when checking Actn2
    
    Returns:
        Filtered list of cell groups (only cardiomyocytes)
    """
    # Compute Actn2 intensity threshold
    actn2_thresh = np.percentile(actn2_channel[actn2_channel > 0], intensity_percentile)
    actn2_positive = actn2_channel > actn2_thresh
    
    h, w = actn2_channel.shape
    filtered_groups = []
    
    for group in cell_groups:
        # Get combined region for the group
        all_coords = []
        for r in group:
            all_coords.append(r.coords)
        all_coords = np.concatenate(all_coords, axis=0)
        
        # Nucleus bounding box
        y_min, x_min = all_coords.min(axis=0)
        y_max, x_max = all_coords.max(axis=0)
        
        # Expand region to check surrounding Actn2
        nuc_h = y_max - y_min
        nuc_w = x_max - x_min
        expand_h = int(nuc_h * expansion_factor / 2)
        expand_w = int(nuc_w * expansion_factor / 2)
        
        check_y1 = max(0, y_min - expand_h)
        check_y2 = min(h, y_max + expand_h)
        check_x1 = max(0, x_min - expand_w)
        check_x2 = min(w, x_max + expand_w)
        
        # Check Actn2 coverage in the region
        region = actn2_positive[check_y1:check_y2, check_x1:check_x2]
        if region.size == 0:
            continue
        
        coverage = region.sum() / region.size
        
        if coverage >= coverage_threshold:
            filtered_groups.append(group)
    
    return filtered_groups


def detect_cardiomyocytes(dapi_channel: np.ndarray,
                          actn2_channel: np.ndarray,
                          min_nucleus_area: int = 200,    # Updated 2026-02-05: 1024px
                          max_nucleus_area: int = 10000,  # Updated 2026-02-05: 1024px
                          actn2_coverage_threshold: float = 0.3,
                          **box_kwargs) -> tuple:
    """
    Hybrid detection: DAPI for localization + Actn2 for cardiomyocyte filtering.
    
    This combines the best of both approaches:
    - DAPI provides reliable nucleus localization
    - Actn2 filtering removes non-cardiomyocyte nuclei (e.g., fibroblasts)
    
    Args:
        dapi_channel: DAPI fluorescence image
        actn2_channel: α-actinin fluorescence image  
        min_nucleus_area: Min nucleus area for filtering
        max_nucleus_area: Max nucleus area for filtering
        actn2_coverage_threshold: Min Actn2 coverage to be considered cardiomyocyte
        **box_kwargs: Additional arguments for create_bounding_boxes
    
    Returns:
        Tuple of (boxes, filtered_groups, all_groups, all_regions)
    """
    # Step 1: Detect nuclei from DAPI
    regions = detect_nuclei(dapi_channel, min_nucleus_area, max_nucleus_area)
    
    # Step 2: Merge close nuclei
    all_groups = merge_close_nuclei(regions)
    
    # Step 3: Filter by Actn2 (keep only cardiomyocytes)
    filtered_groups = filter_by_actn2(
        all_groups, 
        actn2_channel,
        coverage_threshold=actn2_coverage_threshold
    )
    
    # Step 4: Create boxes for filtered cardiomyocytes
    image_shape = dapi_channel.shape
    boxes = create_bounding_boxes(filtered_groups, image_shape, **box_kwargs)
    
    return boxes, filtered_groups, all_groups, regions


def detect_zlines_in_region(actn2_channel: np.ndarray,
                            center_y: int, center_x: int,
                            search_radius: int = 256,     # Updated 2026-02-05: 1024px
                            threshold: float = 0.03,
                            min_sigma: float = 1.0,
                            max_sigma: float = 4.0) -> np.ndarray:
    """
    Detect Z-lines in a circular region around a center point.
    
    Args:
        actn2_channel: α-actinin fluorescence image
        center_y, center_x: Center of search region (nucleus centroid)
        search_radius: Radius of search region in pixels
        threshold: blob_log detection threshold
        min_sigma, max_sigma: Blob size range
    
    Returns:
        np.ndarray: Z-line coordinates (N, 2) in image coordinates
    """
    from skimage.feature import blob_log
    
    h, w = actn2_channel.shape
    
    # Define search region
    y1 = max(0, center_y - search_radius)
    y2 = min(h, center_y + search_radius)
    x1 = max(0, center_x - search_radius)
    x2 = min(w, center_x + search_radius)
    
    # Extract region
    region = actn2_channel[y1:y2, x1:x2].astype(np.float32)
    
    # Normalize
    if region.max() > region.min():
        region = (region - region.min()) / (region.max() - region.min())
    else:
        return np.array([]).reshape(0, 2)
    
    # Detect blobs (Z-lines)
    try:
        blobs = blob_log(region, min_sigma=min_sigma, max_sigma=max_sigma,
                        threshold=threshold, num_sigma=5, overlap=0.5)
    except:
        return np.array([]).reshape(0, 2)
    
    if len(blobs) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to image coordinates
    zlines = blobs[:, :2]  # (y, x) in region
    zlines[:, 0] += y1  # Convert to image y
    zlines[:, 1] += x1  # Convert to image x
    
    return zlines


def get_cell_zlines(zlines: np.ndarray, 
                    nucleus_center: tuple,
                    max_distance: float = 300) -> np.ndarray:
    """
    Filter Z-lines that belong to a specific cell based on distance from nucleus.
    
    Args:
        zlines: All Z-lines in region (N, 2) as (y, x)
        nucleus_center: (cy, cx) nucleus centroid
        max_distance: Maximum distance from nucleus to consider
    
    Returns:
        np.ndarray: Filtered Z-lines belonging to this cell
    """
    if len(zlines) == 0:
        return zlines
    
    cy, cx = nucleus_center
    distances = np.sqrt((zlines[:, 0] - cy)**2 + (zlines[:, 1] - cx)**2)
    
    mask = distances <= max_distance
    return zlines[mask]


def create_adaptive_box(nucleus_group: list,
                        cell_zlines: np.ndarray,
                        image_shape: tuple,
                        min_zlines: int = 15,
                        padding_ratio: float = 0.15,
                        padding_pixels: int = 30,
                        fallback_expansion: float = 4.0) -> list:
    """
    Create a bounding box with adaptive size based on Z-line extent.
    
    If enough Z-lines are found, uses their extent to determine box size.
    Otherwise, falls back to fixed expansion around nucleus.
    
    Args:
        nucleus_group: List of nucleus regions (from merge_close_nuclei)
        cell_zlines: Z-lines belonging to this cell
        image_shape: (H, W) of image
        min_zlines: Minimum Z-lines needed for adaptive sizing
        padding_ratio: Additional padding as ratio of extent
        padding_pixels: Minimum padding in pixels
        fallback_expansion: Expansion factor if not enough Z-lines
    
    Returns:
        [x1, y1, x2, y2] bounding box
    """
    h, w = image_shape
    
    # Get nucleus bounding box
    all_coords = []
    for r in nucleus_group:
        all_coords.append(r.coords)
    all_coords = np.concatenate(all_coords, axis=0)
    
    nuc_y_min, nuc_x_min = all_coords.min(axis=0)
    nuc_y_max, nuc_x_max = all_coords.max(axis=0)
    nuc_cy = (nuc_y_min + nuc_y_max) / 2
    nuc_cx = (nuc_x_min + nuc_x_max) / 2
    nuc_h = nuc_y_max - nuc_y_min
    nuc_w = nuc_x_max - nuc_x_min
    
    # Check if we have enough Z-lines for adaptive sizing
    if len(cell_zlines) >= min_zlines:
        # Use Z-line extent to determine box
        zl_y_min = cell_zlines[:, 0].min()
        zl_y_max = cell_zlines[:, 0].max()
        zl_x_min = cell_zlines[:, 1].min()
        zl_x_max = cell_zlines[:, 1].max()
        
        # Combine nucleus and Z-line extent
        y_min = min(nuc_y_min, zl_y_min)
        y_max = max(nuc_y_max, zl_y_max)
        x_min = min(nuc_x_min, zl_x_min)
        x_max = max(nuc_x_max, zl_x_max)
        
        # Add padding
        extent_h = y_max - y_min
        extent_w = x_max - x_min
        pad_y = max(padding_pixels, int(extent_h * padding_ratio))
        pad_x = max(padding_pixels, int(extent_w * padding_ratio))
        
        box_y1 = max(0, int(y_min - pad_y))
        box_y2 = min(h, int(y_max + pad_y))
        box_x1 = max(0, int(x_min - pad_x))
        box_x2 = min(w, int(x_max + pad_x))
        
    else:
        # Fallback: fixed expansion around nucleus
        box_h = nuc_h * fallback_expansion
        box_w = nuc_w * fallback_expansion
        
        box_y1 = max(0, int(nuc_cy - box_h / 2))
        box_y2 = min(h, int(nuc_cy + box_h / 2))
        box_x1 = max(0, int(nuc_cx - box_w / 2))
        box_x2 = min(w, int(nuc_cx + box_w / 2))
    
    return [box_x1, box_y1, box_x2, box_y2]


def detect_with_adaptive_box(dapi_channel: np.ndarray,
                             actn2_channel: np.ndarray,
                             min_nucleus_area: int = 200,   # Updated 2026-02-05: for 1024px
                             max_nucleus_area: int = 10000, # Updated 2026-02-05: P99 @ 1024
                             search_radius: int = 256,      # Updated 2026-02-05: ~half GT box P99
                             min_zlines: int = 15,
                             zline_threshold: float = 0.03,
                             exclude_edges: bool = True,
                             margin: int = 32,              # Updated 2026-02-05: scaled
                             size_ratio_threshold: float = 3.0,
                             use_relative_distance: bool = True,
                             fixed_merge_distance: int = 373,
                             merge_coeff: float = 1.2) -> tuple:
    """
    DAPI + Actn2 hybrid detection with adaptive box sizing.
    
    Uses DAPI for reliable nucleus localization, then uses Actn2 Z-line
    extent to determine adaptive box size for each cell.
    
    Args:
        dapi_channel: DAPI fluorescence image
        actn2_channel: α-actinin fluorescence image
        min_nucleus_area: Min nucleus area for filtering
        max_nucleus_area: Max nucleus area for filtering
        search_radius: Radius around nucleus to search for Z-lines
        min_zlines: Min Z-lines for adaptive sizing (else fallback)
        zline_threshold: blob_log detection threshold
        exclude_edges: Skip nuclei on image edges
        margin: Edge margin in pixels
        size_ratio_threshold: Max ratio between nucleus sizes for merging
        use_relative_distance: Use size-relative merge threshold
        fixed_merge_distance: Fixed merge distance when relative mode is disabled
        merge_coeff: Relative merge multiplier for average nucleus diameter
    
    Returns:
        Tuple of (boxes, cell_groups, debug_info)
        debug_info contains per-cell statistics for visualization
    """
    # Step 1: Detect nuclei from DAPI
    regions = detect_nuclei(dapi_channel, min_nucleus_area, max_nucleus_area)
    
    # Step 2: Merge close nuclei
    cell_groups = merge_close_nuclei(
        regions,
        size_ratio_threshold=size_ratio_threshold,
        use_relative_distance=use_relative_distance,
        fixed_merge_distance=fixed_merge_distance,
        merge_coeff=merge_coeff,
    )
    
    boxes = []
    debug_info = []
    image_shape = dapi_channel.shape
    
    for group in cell_groups:
        # Skip edge nuclei
        if exclude_edges and any(is_on_edge(r, image_shape, margin) for r in group):
            continue
        
        # Get nucleus centroid
        all_coords = []
        for r in group:
            all_coords.append(r.coords)
        all_coords = np.concatenate(all_coords, axis=0)
        cy = int(all_coords[:, 0].mean())
        cx = int(all_coords[:, 1].mean())
        
        # Step 3: Detect Z-lines around nucleus
        zlines = detect_zlines_in_region(
            actn2_channel, cy, cx, 
            search_radius=search_radius,
            threshold=zline_threshold
        )
        
        # Step 4: Filter Z-lines belonging to this cell
        cell_zlines = get_cell_zlines(zlines, (cy, cx), max_distance=search_radius * 0.8)
        
        # Step 5: Create adaptive box
        box = create_adaptive_box(
            group, cell_zlines, image_shape,
            min_zlines=min_zlines
        )
        
        boxes.append(box)
        debug_info.append({
            'nucleus_center': (cy, cx),
            'num_zlines': len(cell_zlines),
            'adaptive': len(cell_zlines) >= min_zlines,
            'zlines': cell_zlines
        })
    
    return boxes, cell_groups, debug_info
