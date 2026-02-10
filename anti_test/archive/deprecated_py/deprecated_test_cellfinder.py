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
Test CellFinder detection on cardiomyocyte images.
Compares predicted boxes with ground truth boxes.
Visualizes results with napari.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from skimage import measure
from skimage import transform as skt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
from cellSAM import get_model

# Configuration
TEST_DIR = "d:/AI/paper/CellSam/data/processed"


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    p_low, p_high = np.percentile(img, [1, 99])
    img = np.clip(img, p_low, p_high)
    return (img - p_low) / (p_high - p_low + 1e-8)


def get_gt_boxes(mask):
    """Extract ground truth boxes from instance mask."""
    boxes = []
    cell_ids = []
    for region in measure.regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        # Format: [x1, y1, x2, y2]
        boxes.append([x1, y1, x2, y2])
        cell_ids.append(region.label)
    return boxes, cell_ids


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


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes.
    Returns: matched_pairs, unmatched_pred, unmatched_gt
    """
    matched_pairs = []  # (pred_idx, gt_idx, iou)
    matched_gt = set()
    matched_pred = set()
    
    # Compute IoU matrix
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
            matched_pred.add(pred_idx)
    
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return matched_pairs, unmatched_pred, unmatched_gt


def boxes_to_rectangles(boxes, shape, value=1):
    """Convert boxes to rectangle mask for visualization."""
    mask = np.zeros(shape, dtype=np.int32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        # Draw rectangle border
        thickness = 3
        mask[y1:y1+thickness, x1:x2] = i + 1
        mask[y2-thickness:y2, x1:x2] = i + 1
        mask[y1:y2, x1:x1+thickness] = i + 1
        mask[y1:y2, x2-thickness:x2] = i + 1
    return mask


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load CellSAM model (includes CellFinder)
    print('\nLoading CellSAM model (with CellFinder)...')
    model = get_model()
    model = model.to(device)
    model.eval()
    print('Model loaded!')
    
    # Get test samples
    image_dir = Path(TEST_DIR) / 'images'
    mask_dir = Path(TEST_DIR) / 'masks'
    samples = sorted(list(image_dir.glob('*.npy')))[:5]
    
    print(f'\n{"="*70}')
    print('CELLFINDER DETECTION TEST')
    print('='*70)
    print(f'Testing on {len(samples)} samples')
    print(f'IoU threshold for matching: 0.5')
    print('-'*70)
    
    all_results = []
    results_for_napari = []
    
    for sample_path in samples:
        sample_id = sample_path.stem
        print(f'\nProcessing: {sample_id[:45]}...')
        
        # Load data
        image = np.load(sample_path)
        mask = np.load(mask_dir / f'{sample_id}.npy')
        
        # Resize to 1024x1024
        image_resized = skt.resize(image, (1024, 1024), preserve_range=True)
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        image_norm = normalize_image(image_resized)
        
        # Get GT boxes
        gt_boxes, gt_cell_ids = get_gt_boxes(mask_resized.astype(np.int32))
        print(f'  GT boxes: {len(gt_boxes)}')
        
        # Prepare tensor for CellFinder
        img_tensor = np.stack([image_norm] * 3, axis=0)
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
        
        # Run CellFinder detection
        with torch.no_grad():
            try:
                # Use CellSAM's generate_bounding_boxes method
                # This internally uses CellFinder (AnchorDETR)
                pred_boxes_list = model.generate_bounding_boxes(img_tensor)
                
                if pred_boxes_list is not None and len(pred_boxes_list) > 0:
                    # pred_boxes_list is a list of tensors, one per image in batch
                    pred_boxes_tensor = pred_boxes_list[0]  # Get first (only) image
                    if len(pred_boxes_tensor) > 0:
                        pred_boxes = pred_boxes_tensor.cpu().numpy().tolist()
                    else:
                        pred_boxes = []
                else:
                    pred_boxes = []
                    
            except Exception as e:
                print(f'  CellFinder detection error: {e}')
                pred_boxes = []
        
        print(f'  Predicted boxes: {len(pred_boxes)}')
        
        # Match boxes
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            matched, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes)
            
            tp = len(matched)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_iou = np.mean([m[2] for m in matched]) if matched else 0
            
            print(f'  Matched: {tp}, FP: {fp}, FN: {fn}')
            print(f'  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
            print(f'  Avg IoU (matched): {avg_iou:.3f}')
        else:
            tp, fp, fn = 0, len(pred_boxes), len(gt_boxes)
            precision, recall, f1, avg_iou = 0, 0, 0, 0
            matched, unmatched_pred, unmatched_gt = [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
            print(f'  No matches found')
        
        all_results.append({
            'sample_id': sample_id,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou
        })
        
        # Store for napari
        results_for_napari.append({
            'sample_id': sample_id,
            'image': image_resized.astype(np.uint8),
            'gt_mask': mask_resized.astype(np.int32),
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes,
            'matched': matched,
            'unmatched_pred': unmatched_pred,
            'unmatched_gt': unmatched_gt,
        })
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'{"Sample":<50} {"GT":>4} {"Pred":>5} {"TP":>4} {"FP":>4} {"FN":>4} {"P":>6} {"R":>6} {"F1":>6}')
    print('-'*70)
    
    for r in all_results:
        print(f'{r["sample_id"][:48]:<50} {r["gt_count"]:>4} {r["pred_count"]:>5} '
              f'{r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} '
              f'{r["precision"]:>6.3f} {r["recall"]:>6.3f} {r["f1"]:>6.3f}')
    
    # Overall metrics
    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print('-'*70)
    print(f'{"OVERALL":<50} {sum(r["gt_count"] for r in all_results):>4} '
          f'{sum(r["pred_count"] for r in all_results):>5} '
          f'{total_tp:>4} {total_fp:>4} {total_fn:>4} '
          f'{overall_precision:>6.3f} {overall_recall:>6.3f} {overall_f1:>6.3f}')
    print('='*70)
    
    # Launch napari
    print('\n' + '='*70)
    print('LAUNCHING NAPARI VISUALIZATION')
    print('='*70)
    
    try:
        import napari
        
        viewer = napari.Viewer()
        
        for i, result in enumerate(results_for_napari):
            # Add image
            viewer.add_image(
                result['image'],
                name=f"Image_{i+1}",
                visible=(i == 0)
            )
            
            # Add GT mask
            viewer.add_labels(
                result['gt_mask'],
                name=f"GT_Mask_{i+1}",
                visible=(i == 0),
                opacity=0.4
            )
            
            # Add GT boxes as shapes
            if result['gt_boxes']:
                gt_rects = []
                for box in result['gt_boxes']:
                    x1, y1, x2, y2 = box
                    gt_rects.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]))
                viewer.add_shapes(
                    gt_rects,
                    shape_type='polygon',
                    edge_color='green',
                    edge_width=3,
                    face_color='transparent',
                    name=f"GT_Boxes_{i+1}",
                    visible=(i == 0)
                )
            
            # Add predicted boxes as shapes
            if result['pred_boxes']:
                pred_rects = []
                for box in result['pred_boxes']:
                    x1, y1, x2, y2 = box
                    pred_rects.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]))
                viewer.add_shapes(
                    pred_rects,
                    shape_type='polygon',
                    edge_color='red',
                    edge_width=3,
                    face_color='transparent',
                    name=f"Pred_Boxes_{i+1}",
                    visible=(i == 0)
                )
        
        print('\nNapari viewer opened!')
        print('Legend:')
        print('  Green boxes = Ground Truth')
        print('  Red boxes = CellFinder Predictions')
        print('  Colored regions = GT instance mask')
        
        napari.run()
        
    except ImportError:
        print('\nNapari not installed. Results summary above.')
        print('Install with: pip install napari[all]')


if __name__ == "__main__":
    main()
