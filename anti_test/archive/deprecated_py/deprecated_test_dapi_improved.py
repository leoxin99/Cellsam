# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Improved DAPI Detection with Watershed Separation

Author: Bioimage Analysis Evaluation Architect
Date: 2026-01-11

Improvements over original test_dapi_detection.py:
1. Watershed separation for touching nuclei (instead of merge_close_nuclei)
2. Circularity filtering (remove non-nuclear objects)
3. Illumination correction (Gaussian background subtraction)

This addresses the over-merging issue where binucleated cells were incorrectly
merged with neighboring cells.
"""

import os
import sys
import numpy as np
from pathlib import Path
from skimage import measure, filters, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
RAW_TIFF_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
CH_DAPI = 4
CH_MASK = 9


def illumination_correction(img, sigma=50):
    """
    Gaussian background subtraction for illumination correction.
    Estimates background using large-kernel Gaussian blur and subtracts it.
    """
    from scipy.ndimage import gaussian_filter
    
    background = gaussian_filter(img.astype(np.float32), sigma=sigma)
    corrected = img.astype(np.float32) - background
    corrected = np.clip(corrected, 0, None)
    
    # Normalize to 0-1
    if corrected.max() > 0:
        corrected = corrected / corrected.max()
    
    return corrected.astype(np.float32)


def normalize_channel(img):
    """Normalize a channel to [0, 1] using percentile."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
    return img_norm.astype(np.float32)


def watershed_separate_nuclei(binary_mask, min_distance=20):
    """
    Use distance transform + watershed to separate touching nuclei.
    
    This is the key improvement: instead of simply merging close nuclei,
    we use watershed to properly separate touching nuclei while still
    allowing binucleated cells to be identified later.
    
    Args:
        binary_mask: Binary mask of nuclei
        min_distance: Minimum distance between peaks for watershed seeds
    
    Returns:
        Labeled mask where each nucleus has a unique ID
    """
    # Distance transform
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find local maxima as watershed seeds
    # peak_local_max finds pixels that are local maxima in the distance map
    coords = peak_local_max(
        distance, 
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=False
    )
    
    # Create marker image
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    
    # Expand markers to fill the binary mask
    markers = morphology.dilation(markers, morphology.disk(3))
    
    # Watershed segmentation
    labels = segmentation.watershed(-distance, markers, mask=binary_mask)
    
    return labels


def compute_circularity(region):
    """
    Compute circularity of a region.
    Circularity = 4π × area / perimeter²
    Perfect circle = 1, elongated = lower value
    """
    area = region.area
    perimeter = region.perimeter
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0
    return circularity


def detect_nuclei_improved(dapi_channel, 
                            min_nucleus_area=500, 
                            max_nucleus_area=15000,
                            min_circularity=0.2,
                            use_illumination_correction=False,
                            watershed_min_distance=40):
    """
    Improved nucleus detection with:
    1. Illumination correction
    2. Watershed separation
    3. Circularity filtering
    
    Returns:
        labels: Labeled mask of nuclei
        regions: List of valid region properties
        stats: Dictionary with detection statistics
    """
    stats = {
        'original_nuclei': 0,
        'after_watershed': 0,
        'filtered_by_area': 0,
        'filtered_by_circularity': 0,
        'final_nuclei': 0
    }
    
    # Step 1: Illumination correction (optional)
    if use_illumination_correction:
        img_norm = illumination_correction(dapi_channel)
    else:
        img_norm = normalize_channel(dapi_channel)
    
    # Step 2: Otsu thresholding
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    
    # Step 3: Morphological cleanup
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=min_nucleus_area)
    
    # Count original connected components
    original_labels = measure.label(binary)
    stats['original_nuclei'] = original_labels.max()
    
    # Step 4: Watershed separation (KEY IMPROVEMENT)
    labels = watershed_separate_nuclei(binary, min_distance=watershed_min_distance)
    stats['after_watershed'] = labels.max()
    
    # Step 5: Filter by area and circularity
    valid_regions = []
    new_labels = np.zeros_like(labels)
    new_id = 0
    
    for region in measure.regionprops(labels):
        # Area filter
        if region.area < min_nucleus_area or region.area > max_nucleus_area:
            stats['filtered_by_area'] += 1
            continue
        
        # Circularity filter
        circularity = compute_circularity(region)
        if circularity < min_circularity:
            stats['filtered_by_circularity'] += 1
            continue
        
        # Valid nucleus
        new_id += 1
        new_labels[labels == region.label] = new_id
        valid_regions.append(region)
    
    stats['final_nuclei'] = len(valid_regions)
    
    return new_labels, valid_regions, stats


def smart_merge_binucleated(regions, labels, merge_distance=80):
    """
    Smart merging for binucleated cells.
    
    Unlike the original merge_close_nuclei, this function:
    1. Only merges nuclei that are VERY close (true binucleation)
    2. Limits merging to pairs (max 2 nuclei per cell)
    3. Checks size similarity (binucleated nuclei should be similar size)
    
    Args:
        regions: List of region properties
        labels: Labeled mask
        merge_distance: Maximum distance for merging (default: 80px, tighter than before)
    
    Returns:
        cell_groups: List of lists, each sublist contains indices of nuclei in same cell
    """
    if len(regions) <= 1:
        return [[i] for i in range(len(regions))]
    
    centroids = np.array([r.centroid for r in regions])
    areas = np.array([r.area for r in regions])
    n = len(regions)
    
    # Calculate distances
    distances = cdist(centroids, centroids)
    
    # Union-Find with constraints
    parent = list(range(n))
    group_size = [1] * n  # Track group size to limit to 2
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Check if merged group would exceed 2
            if group_size[px] + group_size[py] <= 2:
                parent[px] = py
                group_size[py] += group_size[px]
                return True
        return False
    
    # Only merge pairs that are:
    # 1. Close enough (< merge_distance)
    # 2. Similar in size (ratio between 0.5 and 2.0)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < merge_distance:
                # Check size similarity
                size_ratio = areas[i] / areas[j] if areas[j] > 0 else 0
                if 0.5 <= size_ratio <= 2.0:
                    union(i, j)
    
    # Build groups
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return list(groups.values())


def is_on_edge(centroid, image_shape, margin=30):
    """Check if centroid is near image edge."""
    y, x = centroid
    h, w = image_shape
    return x < margin or y < margin or x > w - margin or y > h - margin


def create_cell_boxes(cell_groups, regions, image_shape, expansion_factor=6.0, 
                       exclude_edges=True, margin=30):
    """
    Create bounding boxes for cells (potentially with merged nuclei).
    """
    boxes = []
    valid_groups = []
    h, w = image_shape
    
    for group_indices in cell_groups:
        group_regions = [regions[i] for i in group_indices]
        
        # Check edge exclusion
        if exclude_edges:
            centroids = [r.centroid for r in group_regions]
            if any(is_on_edge(c, image_shape, margin) for c in centroids):
                continue
        
        # Compute merged bounding box
        y_min = min(r.bbox[0] for r in group_regions)
        x_min = min(r.bbox[1] for r in group_regions)
        y_max = max(r.bbox[2] for r in group_regions)
        x_max = max(r.bbox[3] for r in group_regions)
        
        # Expand
        box_h, box_w = y_max - y_min, x_max - x_min
        center_y, center_x = (y_min + y_max) / 2, (x_min + x_max) / 2
        new_h, new_w = box_h * expansion_factor, box_w * expansion_factor
        
        x1 = int(max(0, center_x - new_w / 2))
        y1 = int(max(0, center_y - new_h / 2))
        x2 = int(min(w, center_x + new_w / 2))
        y2 = int(min(h, center_y + new_h / 2))
        
        boxes.append([x1, y1, x2, y2])
        valid_groups.append(group_indices)
    
    return boxes, valid_groups


def detect_cells_improved(dapi_channel, image_shape):
    """
    Full improved detection pipeline.
    
    Returns:
        boxes: List of [x1, y1, x2, y2] bounding boxes
        stats: Detection statistics
    """
    # Step 1: Detect nuclei with watershed
    labels, regions, stats = detect_nuclei_improved(dapi_channel)
    
    # Step 2: Smart merge for binucleated cells
    cell_groups = smart_merge_binucleated(regions, labels)
    stats['cell_groups'] = len(cell_groups)
    stats['binucleated_cells'] = sum(1 for g in cell_groups if len(g) == 2)
    
    # Step 3: Create boxes
    boxes, valid_groups = create_cell_boxes(cell_groups, regions, image_shape)
    stats['final_cells'] = len(boxes)
    
    return boxes, stats


# ============== Testing ==============

def compute_detection_metrics(pred_boxes, gt_mask, iou_threshold=0.5):
    """Compute detection F1 score."""
    gt_regions = measure.regionprops(gt_mask)
    gt_boxes = []
    for r in gt_regions:
        y1, x1, y2, x2 = r.bbox
        gt_boxes.append([x1, y1, x2, y2])
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    # Compute IoU matrix
    def box_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    # Match predictions to GT
    matched_gt = set()
    tp = 0
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt:
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    precision = tp / len(pred_boxes) if pred_boxes else 0
    recall = tp / len(gt_boxes) if gt_boxes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def main():
    """Test improved detection on sample images."""
    import tifffile
    
    print("="*70)
    print("IMPROVED DAPI DETECTION WITH WATERSHED SEPARATION")
    print("="*70)
    print("\nImprovements:")
    print("  1. Illumination correction (Gaussian background subtraction)")
    print("  2. Watershed separation (distance transform + peak detection)")
    print("  3. Circularity filtering (remove non-nuclear objects)")
    print("  4. Smart binucleated merging (size similarity + distance limit)")
    print()
    
    # Get sample files
    tiff_files = sorted(list(RAW_TIFF_DIR.glob("*.tiff")))[:5]
    
    all_results = []
    
    for i, tiff_path in enumerate(tiff_files):
        sample_id = tiff_path.stem[:40]
        print(f"\n[{i+1}/5] {sample_id}")
        
        with tifffile.TiffFile(tiff_path) as tif:
            data = np.squeeze(tif.asarray())
        
        dapi = data[CH_DAPI]
        gt_mask = data[CH_MASK]
        image_shape = dapi.shape
        
        # Run improved detection
        boxes, stats = detect_cells_improved(dapi, image_shape)
        
        # Compute metrics
        metrics = compute_detection_metrics(boxes, gt_mask)
        
        print(f"  Stats: {stats}")
        print(f"  Detection: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        all_results.append({
            'sample': sample_id,
            **stats,
            **metrics
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    mean_f1 = np.mean([r['f1'] for r in all_results])
    mean_precision = np.mean([r['precision'] for r in all_results])
    mean_recall = np.mean([r['recall'] for r in all_results])
    
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    
    # Compare with original (expected ~0.75)
    print(f"\nOriginal DAPI Detection F1: ~0.75")
    print(f"Improved DAPI Detection F1: {mean_f1:.4f}")


if __name__ == "__main__":
    main()
