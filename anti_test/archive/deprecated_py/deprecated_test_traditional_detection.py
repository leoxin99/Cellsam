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
Traditional cell detection using ACT2N (sarcomere) and DAPI (nuclei) channels.
Alternative to CellFinder for cardiomyocyte detection.

Strategy:
1. Load original TIFF with all channels
2. Use ACT2N (Ch2 or Ch4) for cell body detection
3. Use DAPI (Ch1) for nucleus detection as reference
4. Compare detected boxes with GT
5. Visualize in napari
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import measure, filters, morphology
from skimage import transform as skt
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))

# Configuration
RAW_TIFF_DIR = "d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full"
PROCESSED_DIR = "d:/AI/paper/CellSam/data/processed"
OUTPUT_DIR = "d:/AI/paper/CellSam/anti_test"

# Channel mapping (based on Allen Cell dataset - allen_channel_defs.json)
# Ch0 = bf (Brightfield)
# Ch1 = 488 (488nm fluorescence - Actn2-GFP for sarcomere structure)
# Ch2 = 561 (561nm fluorescence)
# Ch3 = 638 (638nm fluorescence)
# Ch4 = nuc (DAPI - nuclei)
# Ch5 = seg488 (segmentation mask for 488)
# Ch6 = seg561 (segmentation mask for 561)
# Ch7 = seg638 (segmentation mask for 638)
# Ch8 = backmask (background mask)
# Ch9 = cell (instance segmentation mask)
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1       # 488nm - Actn2-GFP (sarcomere marker) - CORRECT CHANNEL!
CH_561 = 2
CH_638 = 3
CH_DAPI = 4        # nuc - DAPI nuclei staining
CH_SEG488 = 5
CH_SEG561 = 6
CH_SEG638 = 7
CH_BACKMASK = 8
CH_MASK = 9


def normalize_channel(img):
    """Normalize a channel to [0, 1] using percentile."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
    return img_norm.astype(np.float32)


def detect_cells_from_fluorescence(fluor_channel, min_area=500, max_area=500000):
    """
    Detect cells from a fluorescence channel using traditional methods.
    
    Args:
        fluor_channel: 2D array of fluorescence intensity
        min_area: minimum cell area in pixels
        max_area: maximum cell area in pixels
    
    Returns:
        boxes: list of [x1, y1, x2, y2] bounding boxes
        labels: labeled mask
    """
    # Normalize
    img_norm = normalize_channel(fluor_channel)
    
    # Apply Otsu thresholding
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.5
    
    binary = img_norm > thresh
    
    # Morphological operations to clean up
    # Close small holes
    binary = morphology.binary_closing(binary, morphology.disk(5))
    # Remove small objects
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    # Fill holes
    binary = ndimage.binary_fill_holes(binary)
    
    # Label connected components
    labels = measure.label(binary)
    
    # Extract bounding boxes
    boxes = []
    valid_labels = []
    
    for region in measure.regionprops(labels):
        area = region.area
        if min_area <= area <= max_area:
            y1, x1, y2, x2 = region.bbox
            boxes.append([x1, y1, x2, y2])
            valid_labels.append(region.label)
    
    # Create cleaned label mask
    clean_labels = np.zeros_like(labels)
    for i, label in enumerate(valid_labels):
        clean_labels[labels == label] = i + 1
    
    return boxes, clean_labels


def detect_nuclei_and_expand(dapi_channel, expansion_factor=3.0, min_area=100):
    """
    Detect nuclei from DAPI and expand to estimate cell body.
    
    Args:
        dapi_channel: 2D array of DAPI intensity
        expansion_factor: how much to expand nuclei to estimate cell body
        min_area: minimum nucleus area
    
    Returns:
        boxes: list of [x1, y1, x2, y2] bounding boxes
        labels: labeled mask
    """
    # Normalize
    img_norm = normalize_channel(dapi_channel)
    
    # Threshold
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.5
    
    binary = img_norm > thresh
    
    # Clean up
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    
    # Label
    labels = measure.label(binary)
    
    boxes = []
    for region in measure.regionprops(labels):
        y1, x1, y2, x2 = region.bbox
        
        # Expand box to estimate cell body
        h, w = y2 - y1, x2 - x1
        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        
        # Expand
        new_h = h * expansion_factor
        new_w = w * expansion_factor
        
        x1_new = int(max(0, cx - new_w / 2))
        y1_new = int(max(0, cy - new_h / 2))
        x2_new = int(min(dapi_channel.shape[1], cx + new_w / 2))
        y2_new = int(min(dapi_channel.shape[0], cy + new_h / 2))
        
        boxes.append([x1_new, y1_new, x2_new, y2_new])
    
    return boxes, labels


def fuse_detections(fluor_boxes, dapi_boxes, iou_threshold=0.3):
    """
    Fuse detections from fluorescence and DAPI.
    Prefer fluorescence boxes, add DAPI boxes that don't overlap.
    """
    if not fluor_boxes:
        return dapi_boxes
    if not dapi_boxes:
        return fluor_boxes
    
    # Start with all fluorescence boxes
    fused = fluor_boxes.copy()
    
    # Add DAPI boxes that don't overlap with fluorescence
    for dapi_box in dapi_boxes:
        max_iou = 0
        for fluor_box in fluor_boxes:
            iou = compute_iou(dapi_box, fluor_box)
            max_iou = max(max_iou, iou)
        
        if max_iou < iou_threshold:
            fused.append(dapi_box)
    
    return fused


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


def get_gt_boxes(mask):
    """Extract ground truth boxes from instance mask."""
    boxes = []
    for region in measure.regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
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
    print('ACT2N + DAPI BASED CELL DETECTION')
    print('='*70)
    
    # Get raw TIFF files
    raw_dir = Path(RAW_TIFF_DIR)
    tiff_files = sorted(list(raw_dir.glob("*.tiff")))[:5]
    
    if not tiff_files:
        print(f"No TIFF files found in {RAW_TIFF_DIR}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files for testing")
    print('-'*70)
    
    all_results = []
    results_for_napari = []
    
    for tiff_path in tiff_files:
        sample_id = tiff_path.stem
        print(f'\nProcessing: {sample_id[:45]}...')
        
        # Load TIFF
        with tifffile.TiffFile(tiff_path) as tif:
            data = np.squeeze(tif.asarray())
        
        if len(data.shape) != 3 or data.shape[0] < 10:
            print(f'  [SKIP] Unexpected shape: {data.shape}')
            continue
        
        # Extract channels - USE CORRECT CHANNELS
        brightfield = data[CH_BRIGHTFIELD]
        actn2 = data[CH_ACTN2]   # 488nm - Actn2-GFP sarcomere marker
        dapi = data[CH_DAPI]      # Nuclei
        gt_mask = data[CH_MASK]
        
        print(f'  Image shape: {brightfield.shape}')
        print(f'  GT cells: {len(np.unique(gt_mask)) - 1}')
        print(f'  Actn2 (Ch1) range: {actn2.min()}-{actn2.max()}')
        print(f'  DAPI (Ch4) range: {dapi.min()}-{dapi.max()}')
        
        # Method 1: Actn2-based detection with LARGE min_area to prevent oversegmentation
        # Cardiomyocytes are large cells (~50000-200000 pixels)
        actn2_boxes, actn2_labels = detect_cells_from_fluorescence(
            actn2, 
            min_area=10000,   # Large min_area to prevent oversegmentation
            max_area=500000
        )
        print(f'  Actn2 detection: {len(actn2_boxes)} boxes')
        
        # Method 2: DAPI-based detection with expansion (larger expansion for cardiomyocytes)
        dapi_boxes, dapi_labels = detect_nuclei_and_expand(
            dapi, 
            expansion_factor=8.0,  # Larger expansion - cardiomyocytes are big
            min_area=500           # Nuclei min area
        )
        print(f'  DAPI (expanded x8) detection: {len(dapi_boxes)} boxes')
        
        # Fuse detections - prefer Actn2 boxes
        fused_boxes = fuse_detections(actn2_boxes, dapi_boxes)
        print(f'  Fused detection: {len(fused_boxes)} boxes')
        
        # Get GT boxes
        gt_boxes = get_gt_boxes(gt_mask)
        
        # Evaluate each method
        methods = [
            ('Actn2', actn2_boxes),
            ('DAPI_Expanded', dapi_boxes),
            ('Fused', fused_boxes),
        ]
        
        for method_name, pred_boxes in methods:
            if not pred_boxes or not gt_boxes:
                continue
            
            matched, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes)
            tp = len(matched)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f'  {method_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}')
            
            all_results.append({
                'sample_id': sample_id,
                'method': method_name,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Store for napari (using fused detection)
        results_for_napari.append({
            'sample_id': sample_id,
            'brightfield': normalize_channel(brightfield),
            'dapi': normalize_channel(dapi),
            'actn2': normalize_channel(actn2),
            'gt_mask': gt_mask,
            'gt_boxes': gt_boxes,
            'actn2_boxes': actn2_boxes,
            'dapi_boxes': dapi_boxes,
            'fused_boxes': fused_boxes,
            'actn2_labels': actn2_labels,
        })
    
    # Summary by method
    print('\n' + '='*70)
    print('SUMMARY BY METHOD')
    print('='*70)
    
    for method in ['Actn2', 'DAPI_Expanded', 'Fused']:
        method_results = [r for r in all_results if r['method'] == method]
        if method_results:
            total_tp = sum(r['tp'] for r in method_results)
            total_fp = sum(r['fp'] for r in method_results)
            total_fn = sum(r['fn'] for r in method_results)
            
            overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
            
            print(f'{method:15s}: P={overall_p:.3f}, R={overall_r:.3f}, F1={overall_f1:.3f}')
    
    # Compare with CellFinder
    print('\n' + '-'*70)
    print('COMPARISON WITH CELLFINDER')
    print('-'*70)
    print('CellFinder:      P=0.009, R=0.016, F1=0.012')
    
    # Save results
    results_file = Path(OUTPUT_DIR) / 'traditional_detection_results.txt'
    with open(results_file, 'w') as f:
        f.write('Traditional Detection Results (ACT2N/DAPI)\n')
        f.write('='*70 + '\n\n')
        for r in all_results:
            f.write(f"{r['sample_id'][:40]}, {r['method']}: ")
            f.write(f"GT={r['gt_count']}, Pred={r['pred_count']}, ")
            f.write(f"TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, ")
            f.write(f"P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f}\n")
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
                visible=False,
                colormap='blue'
            )
            viewer.add_image(
                result['actn2'],
                name=f"Actn2_{i+1}",
                visible=False,
                colormap='green'
            )
            
            # Add GT mask
            viewer.add_labels(
                result['gt_mask'].astype(np.int32),
                name=f"GT_Mask_{i+1}",
                visible=(i == 0),
                opacity=0.4
            )
            
            # Add Actn2 detection labels
            viewer.add_labels(
                result['actn2_labels'].astype(np.int32),
                name=f"Actn2_Detection_{i+1}",
                visible=False,
                opacity=0.5
            )
            
            # Add boxes as shapes
            if result['gt_boxes']:
                gt_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['gt_boxes']]
                viewer.add_shapes(gt_rects, shape_type='polygon', edge_color='green', edge_width=2, face_color='transparent', name=f"GT_Boxes_{i+1}", visible=(i == 0))
            
            if result['fused_boxes']:
                fused_rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) for b in result['fused_boxes']]
                viewer.add_shapes(fused_rects, shape_type='polygon', edge_color='red', edge_width=2, face_color='transparent', name=f"Fused_Boxes_{i+1}", visible=(i == 0))
        
        print('\nNapari viewer opened!')
        print('Legend:')
        print('  Green boxes = Ground Truth')
        print('  Red boxes = Traditional Detection (Fused)')
        print('  Blue = DAPI, Green = Fluorescence')
        
        napari.run()
        
    except ImportError:
        print('\nNapari not installed. Results summary above.')


if __name__ == "__main__":
    main()
