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
DAPI + Actn2 validated cardiomyocyte detection.

Improved strategy based on user observations:
1. DAPI detects ALL nuclei (cardiomyocytes + other cells)
2. Actn2 (α-Actinin 2) is cardiomyocyte-specific sarcomere marker
3. Use Actn2 signal to validate: only nuclei with nearby Actn2 signal = cardiomyocytes

This filters out non-cardiomyocytes (e.g., fibroblasts, undifferentiated cells).
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
CH_ACTN2 = 1       # 488nm - Actn2-GFP (cardiomyocyte-specific sarcomere marker)
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


def detect_nuclei(dapi_channel, min_nucleus_area=500, max_nucleus_area=15000):
    """
    Detect individual nuclei from DAPI channel.
    Returns all nuclei without filtering by relative size (will filter by Actn2 instead).
    """
    img_norm = normalize_channel(dapi_channel)
    
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=min_nucleus_area)
    binary = ndimage.binary_fill_holes(binary)
    
    labels = measure.label(binary)
    
    centroids = []
    valid_regions = []
    
    for region in measure.regionprops(labels):
        area = region.area
        if min_nucleus_area <= area <= max_nucleus_area:
            centroids.append(region.centroid)
            valid_regions.append(region)
    
    return centroids, labels, valid_regions


def create_actn2_mask(actn2_channel, threshold_percentile=50):
    """
    Create a binary mask of regions with Actn2 signal.
    Cardiomyocytes have Actn2 signal, other cells don't.
    
    Args:
        actn2_channel: 2D array of Actn2 fluorescence
        threshold_percentile: percentile for thresholding (lower = more permissive)
    
    Returns:
        actn2_mask: binary mask of Actn2-positive regions
    """
    img_norm = normalize_channel(actn2_channel)
    
    # Use percentile-based threshold (Actn2 signal is variable)
    thresh = np.percentile(img_norm, threshold_percentile)
    
    # More permissive threshold for Actn2 (sarcomere structure)
    binary = img_norm > thresh
    
    # Dilate to connect sarcomere stripes into cell regions
    binary = morphology.binary_dilation(binary, morphology.disk(10))
    binary = ndimage.binary_fill_holes(binary)
    
    return binary


def filter_nuclei_by_actn2(nuclei_regions, actn2_mask, overlap_threshold=0.3):
    """
    Filter nuclei: keep only those that overlap with Actn2-positive regions.
    
    Args:
        nuclei_regions: list of region properties
        actn2_mask: binary mask of Actn2-positive regions
        overlap_threshold: minimum fraction of nucleus that must overlap with Actn2
    
    Returns:
        cm_regions: cardiomyocyte nuclei (Actn2-positive)
        other_regions: non-cardiomyocyte nuclei (Actn2-negative)
    """
    cm_regions = []
    other_regions = []
    
    for region in nuclei_regions:
        # Get the nucleus region mask
        y1, x1, y2, x2 = region.bbox
        
        # Calculate overlap with Actn2 mask
        nucleus_pixels = region.area
        actn2_overlap = 0
        
        for coord in region.coords:
            y, x = coord
            if actn2_mask[y, x]:
                actn2_overlap += 1
        
        overlap_ratio = actn2_overlap / nucleus_pixels if nucleus_pixels > 0 else 0
        
        if overlap_ratio >= overlap_threshold:
            cm_regions.append(region)
        else:
            other_regions.append(region)
    
    return cm_regions, other_regions


def merge_close_nuclei(regions, merge_distance=100):
    """Merge nuclei that are close together (binucleated cardiomyocytes)."""
    if len(regions) == 0:
        return [], []
    
    if len(regions) == 1:
        return [regions[0].centroid], [[regions[0]]]
    
    centroids = np.array([r.centroid for r in regions])
    n = len(centroids)
    
    distances = cdist(centroids, centroids)
    
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
            if distances[i, j] < merge_distance:
                union(i, j)
    
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    cell_centroids = []
    cell_regions = []
    
    for indices in groups.values():
        group_centroids = centroids[indices]
        merged_centroid = tuple(group_centroids.mean(axis=0))
        cell_centroids.append(merged_centroid)
        cell_regions.append([regions[i] for i in indices])
    
    return cell_centroids, cell_regions


def is_on_edge(region, image_shape, margin=30):
    """Check if a region touches the image edge."""
    y1, x1, y2, x2 = region.bbox
    h, w = image_shape
    return (x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin)


def create_cell_boxes(cell_centroids, cell_regions, image_shape, 
                      expansion_factor=6.0, exclude_edges=True, margin=30):
    """Create bounding boxes for detected cells."""
    boxes = []
    filtered_centroids = []
    h, w = image_shape
    
    for centroid, regions in zip(cell_centroids, cell_regions):
        if exclude_edges:
            on_edge = any(is_on_edge(r, image_shape, margin) for r in regions)
            if on_edge:
                continue
        
        y_min = min(r.bbox[0] for r in regions)
        x_min = min(r.bbox[1] for r in regions)
        y_max = max(r.bbox[2] for r in regions)
        x_max = max(r.bbox[3] for r in regions)
        
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
    print('DAPI + Actn2 VALIDATED CARDIOMYOCYTE DETECTION')
    print('Strategy: DAPI for all nuclei + Actn2 to filter cardiomyocytes')
    print('='*70)
    
    # Parameters
    ACTN2_THRESHOLD_PERCENTILE = 40  # Lower = more permissive Actn2 detection
    ACTN2_OVERLAP_THRESHOLD = 0.2    # Min fraction of nucleus overlapping with Actn2
    MERGE_DISTANCE = 100             # Distance to merge binucleated cells
    EXPANSION_FACTOR = 6.0
    EDGE_MARGIN = 30
    
    print(f'\nParameters:')
    print(f'  Actn2 threshold percentile: {ACTN2_THRESHOLD_PERCENTILE}')
    print(f'  Actn2 overlap threshold: {ACTN2_OVERLAP_THRESHOLD}')
    print(f'  Merge distance: {MERGE_DISTANCE} px')
    print('-'*70)
    
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
        
        with tifffile.TiffFile(tiff_path) as tif:
            data = np.squeeze(tif.asarray())
        
        if len(data.shape) != 3 or data.shape[0] < 10:
            print(f'  [SKIP] Unexpected shape: {data.shape}')
            continue
        
        brightfield = data[CH_BRIGHTFIELD]
        actn2 = data[CH_ACTN2]
        dapi = data[CH_DAPI]
        gt_mask = data[CH_MASK]
        
        image_shape = brightfield.shape
        gt_boxes = get_gt_boxes(gt_mask)
        n_gt = len(gt_boxes)
        
        print(f'  Image shape: {image_shape}')
        print(f'  GT cells: {n_gt}')
        
        # Step 1: Detect ALL nuclei from DAPI
        all_centroids, nuclei_labels, all_regions = detect_nuclei(dapi)
        n_all_nuclei = len(all_regions)
        print(f'  All nuclei (DAPI): {n_all_nuclei}')
        
        # Step 2: Create Actn2 mask (cardiomyocyte regions)
        actn2_mask = create_actn2_mask(actn2, threshold_percentile=ACTN2_THRESHOLD_PERCENTILE)
        actn2_coverage = actn2_mask.sum() / actn2_mask.size * 100
        print(f'  Actn2+ coverage: {actn2_coverage:.1f}%')
        
        # Step 3: Filter nuclei by Actn2 overlap
        cm_regions, other_regions = filter_nuclei_by_actn2(
            all_regions, actn2_mask, overlap_threshold=ACTN2_OVERLAP_THRESHOLD
        )
        n_cm = len(cm_regions)
        n_other = len(other_regions)
        print(f'  Cardiomyocyte nuclei (Actn2+): {n_cm}')
        print(f'  Non-cardiomyocyte nuclei: {n_other}')
        
        # Step 4: Merge close nuclei (binucleated)
        cell_centroids, cell_regions = merge_close_nuclei(cm_regions, merge_distance=MERGE_DISTANCE)
        n_cells_before_edge = len(cell_centroids)
        binucleated = sum(1 for regs in cell_regions if len(regs) > 1)
        print(f'  After merging: {n_cells_before_edge} cells ({binucleated} binucleated)')
        
        # Step 5: Create boxes and exclude edge cells
        pred_boxes, filtered_centroids = create_cell_boxes(
            cell_centroids, cell_regions, image_shape,
            expansion_factor=EXPANSION_FACTOR,
            exclude_edges=True,
            margin=EDGE_MARGIN
        )
        n_edge_excluded = n_cells_before_edge - len(pred_boxes)
        print(f'  After edge exclusion: {len(pred_boxes)} cells ({n_edge_excluded} excluded)')
        
        # Step 6: Evaluate
        if pred_boxes and gt_boxes:
            matched, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes)
            tp = len(matched)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f'  Results: TP={tp}, FP={fp}, FN={fn}')
            print(f'  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        else:
            tp, fp, fn = 0, len(pred_boxes), len(gt_boxes)
            precision, recall, f1 = 0, 0, 0
        
        all_results.append({
            'sample_id': sample_id,
            'gt_count': n_gt,
            'all_nuclei': n_all_nuclei,
            'cm_nuclei': n_cm,
            'other_nuclei': n_other,
            'pred_count': len(pred_boxes),
            'binucleated': binucleated,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Store for napari
        results_for_napari.append({
            'sample_id': sample_id,
            'brightfield': normalize_channel(brightfield),
            'dapi': normalize_channel(dapi),
            'actn2': normalize_channel(actn2),
            'actn2_mask': actn2_mask,
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
    total_filtered = sum(r['other_nuclei'] for r in all_results)
    
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print(f'Total GT cells: {total_gt}')
    print(f'Total detected: {total_pred}')
    print(f'Non-CM filtered out: {total_filtered}')
    print(f'TP: {total_tp}, FP: {total_fp}, FN: {total_fn}')
    print(f'Overall Precision: {overall_p:.3f}')
    print(f'Overall Recall: {overall_r:.3f}')
    print(f'Overall F1: {overall_f1:.3f}')
    
    print('\n' + '-'*70)
    print('COMPARISON')
    print('-'*70)
    print(f'CellFinder:           P=0.009, R=0.016, F1=0.012')
    print(f'DAPI only:            P=0.708, R=0.797, F1=0.750')
    print(f'DAPI + Actn2:         P={overall_p:.3f}, R={overall_r:.3f}, F1={overall_f1:.3f}')
    
    # Save results
    results_file = Path(OUTPUT_DIR) / 'dapi_actn2_detection_results.txt'
    with open(results_file, 'w') as f:
        f.write('DAPI + Actn2 Validated Cardiomyocyte Detection Results\n')
        f.write('='*70 + '\n\n')
        f.write(f'Actn2 threshold: {ACTN2_THRESHOLD_PERCENTILE}th percentile\n')
        f.write(f'Actn2 overlap threshold: {ACTN2_OVERLAP_THRESHOLD}\n\n')
        
        for r in all_results:
            f.write(f"Sample: {r['sample_id'][:45]}\n")
            f.write(f"  All nuclei: {r['all_nuclei']}, CM: {r['cm_nuclei']}, Other: {r['other_nuclei']}\n")
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
            viewer.add_image(
                result['actn2'],
                name=f"Actn2_{i+1}",
                visible=False,
                colormap='green',
                blending='additive'
            )
            
            # Add Actn2 mask
            viewer.add_labels(
                result['actn2_mask'].astype(np.int32),
                name=f"Actn2_Mask_{i+1}",
                visible=False,
                opacity=0.3
            )
            
            viewer.add_labels(
                result['nuclei_labels'].astype(np.int32),
                name=f"All_Nuclei_{i+1}",
                visible=False,
                opacity=0.5
            )
            
            viewer.add_labels(
                result['gt_mask'].astype(np.int32),
                name=f"GT_Mask_{i+1}",
                visible=False,
                opacity=0.4
            )
            
            if result['gt_boxes']:
                gt_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['gt_boxes']]
                viewer.add_shapes(gt_rects, shape_type='polygon', edge_color='green', edge_width=3, 
                                  face_color='transparent', name=f"GT_Boxes_{i+1}", visible=(i == 0))
            
            if result['pred_boxes']:
                pred_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['pred_boxes']]
                viewer.add_shapes(pred_rects, shape_type='polygon', edge_color='red', edge_width=3, 
                                  face_color='transparent', name=f"Pred_Boxes_{i+1}", visible=(i == 0))
        
        print('\nNapari viewer opened!')
        print('Legend:')
        print('  Green boxes = Ground Truth')
        print('  Red boxes = DAPI+Actn2 Detection')
        print('  Blue = DAPI, Green = Actn2')
        
        napari.run()
        
    except ImportError:
        print('\nNapari not installed. Results summary above.')


if __name__ == "__main__":
    main()
