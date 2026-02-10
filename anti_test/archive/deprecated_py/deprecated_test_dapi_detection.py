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
DAPI-based cardiomyocyte detection with nucleus merging.

Strategy based on user observations:
1. Single nucleus = one cardiomyocyte
2. Two close nuclei = one binucleated cardiomyocyte (merge them)
3. Edge cells (touching image border) without GT = ignore

This is more biologically accurate for hiPSC-CM detection.
"""

import os
import sys
import numpy as np
from pathlib import Path
from skimage import measure, filters, morphology
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

import tifffile

# Configuration
RAW_TIFF_DIR = "d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full"
OUTPUT_DIR = "d:/AI/paper/CellSam/anti_test"

# Channel mapping (from allen_channel_defs.json)
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1       # 488nm - Actn2-GFP
CH_DAPI = 4        # nuc - DAPI nuclei
CH_MASK = 9        # cell instance mask


def normalize_channel(img):
    """Normalize a channel to [0, 1] using percentile."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
    return img_norm.astype(np.float32)


def detect_nuclei(dapi_channel, min_nucleus_area=500, max_nucleus_area=15000, 
                   relative_size_threshold=0.2):
    """
    Detect individual nuclei from DAPI channel.
    
    Args:
        dapi_channel: 2D array of DAPI intensity
        min_nucleus_area: minimum nucleus area (filter out debris)
        max_nucleus_area: maximum nucleus area (filter out merged artifacts)
        relative_size_threshold: exclude nuclei smaller than median_area * this value
            e.g., 0.2 means exclude nuclei smaller than 20% of median size (5x smaller)
    
    Returns:
        nuclei_centroids: list of (y, x) centroids
        nuclei_labels: labeled mask
        nuclei_regions: list of region properties
        stats: dict with filtering statistics
    """
    # Normalize
    img_norm = normalize_channel(dapi_channel)
    
    # Threshold
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    
    # Morphological cleanup
    binary = morphology.binary_opening(binary, morphology.disk(3))  # Larger disk to remove noise
    binary = morphology.remove_small_objects(binary, min_size=min_nucleus_area)
    binary = ndimage.binary_fill_holes(binary)  # Fill holes in nuclei
    
    # Label nuclei
    labels = measure.label(binary)
    
    # First pass: collect all regions and their areas
    all_regions = []
    all_areas = []
    
    for region in measure.regionprops(labels):
        area = region.area
        if area <= max_nucleus_area:  # Only filter by max area first
            all_regions.append(region)
            all_areas.append(area)
    
    # Calculate median area for relative filtering
    if len(all_areas) > 0:
        median_area = np.median(all_areas)
        min_relative_area = median_area * relative_size_threshold
    else:
        median_area = 0
        min_relative_area = min_nucleus_area
    
    # Second pass: filter by relative size
    centroids = []
    valid_regions = []
    n_too_small_absolute = 0
    n_too_small_relative = 0
    
    for region in all_regions:
        area = region.area
        
        # Check absolute minimum
        if area < min_nucleus_area:
            n_too_small_absolute += 1
            continue
        
        # Check relative size (compared to median)
        if area < min_relative_area:
            n_too_small_relative += 1
            continue
        
        centroids.append(region.centroid)  # (y, x)
        valid_regions.append(region)
    
    stats = {
        'total_detected': len(all_regions),
        'median_area': median_area,
        'min_relative_area': min_relative_area,
        'filtered_absolute': n_too_small_absolute,
        'filtered_relative': n_too_small_relative,
        'valid_nuclei': len(valid_regions)
    }
    
    return centroids, labels, valid_regions, stats


def merge_close_nuclei(centroids, regions, merge_distance=100):
    """
    Merge nuclei that are close together (binucleated cardiomyocytes).
    
    Args:
        centroids: list of (y, x) centroids
        regions: list of region properties
        merge_distance: maximum distance to consider two nuclei as belonging to same cell
    
    Returns:
        cell_centroids: merged centroids (one per cell)
        cell_regions: list of regions belonging to each cell (for bbox calculation)
    """
    if len(centroids) == 0:
        return [], []
    
    if len(centroids) == 1:
        return centroids, [[regions[0]]]
    
    centroids = np.array(centroids)
    n = len(centroids)
    
    # Compute pairwise distances
    distances = cdist(centroids, centroids)
    
    # Union-Find to group close nuclei
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Merge nuclei within merge_distance
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < merge_distance:
                union(i, j)
    
    # Group nuclei by their root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Calculate merged centroids and collect regions
    cell_centroids = []
    cell_regions = []
    
    for indices in groups.values():
        # Merged centroid is the mean of all nuclei centroids
        group_centroids = centroids[indices]
        merged_centroid = tuple(group_centroids.mean(axis=0))
        cell_centroids.append(merged_centroid)
        
        # Collect all regions for this cell
        cell_regions.append([regions[i] for i in indices])
    
    return cell_centroids, cell_regions


def is_on_edge(region, image_shape, margin=20):
    """Check if a region touches the image edge."""
    y1, x1, y2, x2 = region.bbox
    h, w = image_shape
    
    return (x1 < margin or y1 < margin or 
            x2 > w - margin or y2 > h - margin)


def create_cell_boxes(cell_centroids, cell_regions, image_shape, 
                      expansion_factor=5.0, exclude_edges=True, margin=20):
    """
    Create bounding boxes for detected cells.
    
    Args:
        cell_centroids: list of (y, x) centroids
        cell_regions: list of regions for each cell
        image_shape: (height, width)
        expansion_factor: how much to expand from nucleus size
        exclude_edges: whether to exclude cells touching image edge
        margin: edge margin in pixels
    
    Returns:
        boxes: list of [x1, y1, x2, y2] bounding boxes
        filtered_centroids: centroids of non-edge cells
    """
    boxes = []
    filtered_centroids = []
    h, w = image_shape
    
    for centroid, regions in zip(cell_centroids, cell_regions):
        # Check if any nucleus touches edge
        if exclude_edges:
            on_edge = any(is_on_edge(r, image_shape, margin) for r in regions)
            if on_edge:
                continue
        
        # Calculate bounding box from all nuclei of this cell
        y_min = min(r.bbox[0] for r in regions)
        x_min = min(r.bbox[1] for r in regions)
        y_max = max(r.bbox[2] for r in regions)
        x_max = max(r.bbox[3] for r in regions)
        
        # Expand box to estimate cell body
        box_h = y_max - y_min
        box_w = x_max - x_min
        center_y = (y_min + y_max) / 2
        center_x = (x_min + x_max) / 2
        
        new_h = box_h * expansion_factor
        new_w = box_w * expansion_factor
        
        x1 = int(max(0, center_x - new_w / 2))
        y1 = int(max(0, center_y - new_h / 2))
        x2 = int(min(w, center_x + new_w / 2))
        y2 = int(min(h, center_y + new_h / 2))
        
        boxes.append([x1, y1, x2, y2])
        filtered_centroids.append(centroid)
    
    return boxes, filtered_centroids


def get_gt_boxes(mask):
    """Extract ground truth boxes from instance mask."""
    boxes = []
    for region in measure.regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3):
    """Match predicted boxes to ground truth."""
    matched_pairs = []
    matched_gt = set()
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matched_pairs.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
    
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in [m[0] for m in matched_pairs]]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return matched_pairs, unmatched_pred, unmatched_gt


def main():
    print('='*70)
    print('DAPI-BASED CARDIOMYOCYTE DETECTION')
    print('Strategy: Nucleus counting with binucleated cell merging')
    print('='*70)
    
    # Parameters
    MERGE_DISTANCE = 100    # Pixels - distance to merge two nuclei as one cell
    EXPANSION_FACTOR = 6.0  # How much to expand from nucleus to cell box
    EXCLUDE_EDGES = True    # Exclude cells touching image edge
    EDGE_MARGIN = 30        # Edge margin in pixels
    
    print(f'\nParameters:')
    print(f'  Merge distance: {MERGE_DISTANCE} px (nuclei closer than this = same cell)')
    print(f'  Box expansion: {EXPANSION_FACTOR}x from nucleus size')
    print(f'  Exclude edge cells: {EXCLUDE_EDGES} (margin: {EDGE_MARGIN} px)')
    print('-'*70)
    
    # Get raw TIFF files
    raw_dir = Path(RAW_TIFF_DIR)
    tiff_files = sorted(list(raw_dir.glob("*.tiff")))[:5]
    
    if not tiff_files:
        print(f"No TIFF files found in {RAW_TIFF_DIR}")
        return
    
    print(f"Testing on {len(tiff_files)} files")
    
    all_results = []
    results_for_napari = []
    
    for tiff_path in tiff_files:
        sample_id = tiff_path.stem
        print(f'\n{"="*70}')
        print(f'Processing: {sample_id[:50]}...')
        
        # Load TIFF
        with tifffile.TiffFile(tiff_path) as tif:
            data = np.squeeze(tif.asarray())
        
        if len(data.shape) != 3 or data.shape[0] < 10:
            print(f'  [SKIP] Unexpected shape: {data.shape}')
            continue
        
        # Extract channels
        brightfield = data[CH_BRIGHTFIELD]
        dapi = data[CH_DAPI]
        gt_mask = data[CH_MASK]
        
        image_shape = brightfield.shape
        gt_boxes = get_gt_boxes(gt_mask)
        n_gt = len(gt_boxes)
        
        print(f'  Image shape: {image_shape}')
        print(f'  GT cells: {n_gt}')
        
        # Step 1: Detect nuclei with size filtering
        nuclei_centroids, nuclei_labels, nuclei_regions, nuc_stats = detect_nuclei(dapi)
        n_nuclei = len(nuclei_centroids)
        print(f'  Detected nuclei: {nuc_stats["total_detected"]} initial, {n_nuclei} valid')
        print(f'    Median nucleus area: {nuc_stats["median_area"]:.0f} px²')
        print(f'    Filtered (too small): {nuc_stats["filtered_absolute"]} absolute, {nuc_stats["filtered_relative"]} relative')
        
        # Step 2: Merge close nuclei (binucleated cells)
        cell_centroids, cell_regions = merge_close_nuclei(
            nuclei_centroids, nuclei_regions, merge_distance=MERGE_DISTANCE
        )
        n_cells_before_edge = len(cell_centroids)
        
        # Count binucleated cells
        binucleated = sum(1 for regs in cell_regions if len(regs) > 1)
        print(f'  After merging: {n_cells_before_edge} cells ({binucleated} binucleated)')
        
        # Step 3: Create boxes and exclude edge cells
        pred_boxes, filtered_centroids = create_cell_boxes(
            cell_centroids, cell_regions, image_shape,
            expansion_factor=EXPANSION_FACTOR,
            exclude_edges=EXCLUDE_EDGES,
            margin=EDGE_MARGIN
        )
        n_edge_excluded = n_cells_before_edge - len(pred_boxes)
        print(f'  After edge exclusion: {len(pred_boxes)} cells ({n_edge_excluded} excluded)')
        
        # Step 4: Evaluate
        if pred_boxes and gt_boxes:
            matched, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes)
            tp = len(matched)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_iou = np.mean([m[2] for m in matched]) if matched else 0
            
            print(f'  Results: TP={tp}, FP={fp}, FN={fn}')
            print(f'  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
            print(f'  Avg IoU (matched): {avg_iou:.3f}')
        else:
            tp, fp, fn = 0, len(pred_boxes), len(gt_boxes)
            precision, recall, f1, avg_iou = 0, 0, 0, 0
        
        all_results.append({
            'sample_id': sample_id,
            'gt_count': n_gt,
            'nuclei_count': n_nuclei,
            'cells_after_merge': n_cells_before_edge,
            'pred_count': len(pred_boxes),
            'binucleated': binucleated,
            'edge_excluded': n_edge_excluded,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou
        })
        
        # Store for napari
        results_for_napari.append({
            'sample_id': sample_id,
            'brightfield': normalize_channel(brightfield),
            'dapi': normalize_channel(dapi),
            'nuclei_labels': nuclei_labels,
            'gt_mask': gt_mask,
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes,
            'centroids': filtered_centroids,
        })
    
    # Summary
    print('\n' + '='*70)
    print('OVERALL SUMMARY')
    print('='*70)
    
    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    total_gt = sum(r['gt_count'] for r in all_results)
    total_pred = sum(r['pred_count'] for r in all_results)
    total_binuc = sum(r['binucleated'] for r in all_results)
    
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print(f'Total GT cells: {total_gt}')
    print(f'Total detected: {total_pred}')
    print(f'Total binucleated: {total_binuc}')
    print(f'TP: {total_tp}, FP: {total_fp}, FN: {total_fn}')
    print(f'Overall Precision: {overall_p:.3f}')
    print(f'Overall Recall: {overall_r:.3f}')
    print(f'Overall F1: {overall_f1:.3f}')
    
    print('\n' + '-'*70)
    print('COMPARISON')
    print('-'*70)
    print(f'CellFinder:        P=0.009, R=0.016, F1=0.012')
    print(f'DAPI-based:        P={overall_p:.3f}, R={overall_r:.3f}, F1={overall_f1:.3f}')
    
    # Save results
    results_file = Path(OUTPUT_DIR) / 'dapi_detection_results.txt'
    with open(results_file, 'w') as f:
        f.write('DAPI-Based Cardiomyocyte Detection Results\n')
        f.write('='*70 + '\n\n')
        f.write(f'Parameters: merge_dist={MERGE_DISTANCE}, expansion={EXPANSION_FACTOR}, edge_margin={EDGE_MARGIN}\n\n')
        
        for r in all_results:
            f.write(f"Sample: {r['sample_id'][:45]}\n")
            f.write(f"  GT: {r['gt_count']}, Nuclei: {r['nuclei_count']}, Cells: {r['pred_count']}\n")
            f.write(f"  Binucleated: {r['binucleated']}, Edge excluded: {r['edge_excluded']}\n")
            f.write(f"  TP={r['tp']}, FP={r['fp']}, FN={r['fn']}\n")
            f.write(f"  P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f}\n\n")
        
        f.write('='*70 + '\n')
        f.write(f'OVERALL: P={overall_p:.3f}, R={overall_r:.3f}, F1={overall_f1:.3f}\n')
    
    print(f'\nResults saved to: {results_file}')
    
    # Launch napari
    print('\n' + '='*70)
    print('LAUNCHING NAPARI VISUALIZATION')
    print('='*70)
    
    try:
        import napari
        
        viewer = napari.Viewer()
        
        for i, result in enumerate(results_for_napari):
            # Add images
            viewer.add_image(
                result['brightfield'],
                name=f"Brightfield_{i+1}",
                visible=(i == 0)
            )
            viewer.add_image(
                result['dapi'],
                name=f"DAPI_{i+1}",
                visible=(i == 0),
                colormap='blue',
                blending='additive'
            )
            
            # Add nuclei detection
            viewer.add_labels(
                result['nuclei_labels'].astype(np.int32),
                name=f"Nuclei_Detection_{i+1}",
                visible=False,
                opacity=0.5
            )
            
            # Add GT mask
            viewer.add_labels(
                result['gt_mask'].astype(np.int32),
                name=f"GT_Mask_{i+1}",
                visible=False,
                opacity=0.4
            )
            
            # Add GT boxes (green)
            if result['gt_boxes']:
                gt_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['gt_boxes']]
                viewer.add_shapes(gt_rects, shape_type='polygon', edge_color='green', edge_width=3, face_color='transparent', name=f"GT_Boxes_{i+1}", visible=(i == 0))
            
            # Add predicted boxes (red)
            if result['pred_boxes']:
                pred_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['pred_boxes']]
                viewer.add_shapes(pred_rects, shape_type='polygon', edge_color='red', edge_width=3, face_color='transparent', name=f"Pred_Boxes_{i+1}", visible=(i == 0))
            
            # Add centroids as points
            if result['centroids']:
                points = np.array(result['centroids'])
                viewer.add_points(points, name=f"Cell_Centers_{i+1}", size=15, face_color='yellow', visible=False)
        
        print('\nNapari viewer opened!')
        print('Legend:')
        print('  Green boxes = Ground Truth')
        print('  Red boxes = DAPI-based Detection')
        print('  Yellow points = Cell centers (merged nuclei)')
        print('  Blue = DAPI channel')
        
        napari.run()
        
    except ImportError:
        print('\nNapari not installed. Results summary above.')


if __name__ == "__main__":
    main()
