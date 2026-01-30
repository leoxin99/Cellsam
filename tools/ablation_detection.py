"""
Adaptive vs DAPI Only 检测消融实验

对比两种检测方案:
1. DAPI Only: detect_and_create_boxes() - 仅用核检测
2. Adaptive: detect_with_adaptive_box() - Z-线自适应框

运行:
    python tools/ablation_detection.py

输出:
    - console: 对比表格
    - experiments/ablation_detection/results.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.dapi import detect_and_create_boxes, detect_with_adaptive_box


def load_test_samples(data_dir: Path, split_file: Path, n_samples: int = 20):
    """Load test samples with all channels."""
    # Load test IDs
    with open(split_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    # Limit samples
    test_ids = test_ids[:n_samples]
    
    samples = []
    for sample_id in tqdm(test_ids, desc="Loading samples"):
        image_path = data_dir / "images" / f"{sample_id}.npy"
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        
        if not image_path.exists() or not mask_path.exists():
            continue
            
        # Load 3-channel image: [BF, DAPI, Actn2]
        image = np.load(image_path)  # (3, 1024, 1024)
        mask = np.load(mask_path)    # (1024, 1024)
        
        samples.append({
            'id': sample_id,
            'image': image,
            'mask': mask,
            'dapi': image[1],   # Channel 1 = DAPI
            'actn2': image[2],  # Channel 2 = Actn2
        })
    
    return samples


def get_gt_boxes_from_mask(mask: np.ndarray, min_area: int = 1000):
    """Extract ground truth bounding boxes from instance mask."""
    from skimage import measure
    
    boxes = []
    regions = measure.regionprops(mask)
    
    for region in regions:
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        boxes.append({
            'bbox': (minr, minc, maxr, maxc),
            'area': region.area,
            'centroid': region.centroid
        })
    
    return boxes


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes (minr, minc, maxr, maxc)."""
    r1, c1, r2, c2 = box1
    r3, c3, r4, c4 = box2
    
    # Intersection
    inter_r1 = max(r1, r3)
    inter_c1 = max(c1, c3)
    inter_r2 = min(r2, r4)
    inter_c2 = min(c2, c4)
    
    if inter_r2 <= inter_r1 or inter_c2 <= inter_c1:
        return 0.0
    
    inter_area = (inter_r2 - inter_r1) * (inter_c2 - inter_c1)
    
    # Union
    area1 = (r2 - r1) * (c2 - c1)
    area2 = (r4 - r3) * (c4 - c3)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold: float = 0.3):
    """Match predicted boxes to GT boxes using IoU."""
    if not pred_boxes or not gt_boxes:
        return 0, len(pred_boxes), len(gt_boxes)
    
    matched_gt = set()
    tp = 0
    
    for pred in pred_boxes:
        pred_bbox = pred['bbox'] if isinstance(pred, dict) else pred
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_bbox = gt['bbox'] if isinstance(gt, dict) else gt
            iou = compute_box_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


def evaluate_detection_method(samples, method: str = 'dapi_only'):
    """Evaluate a detection method on all samples."""
    
    all_tp, all_fp, all_fn = 0, 0, 0
    sample_results = []
    
    for sample in tqdm(samples, desc=f"Evaluating {method}"):
        # Get predictions - returns (boxes, cell_groups, regions) tuple
        if method == 'dapi_only':
            result = detect_and_create_boxes(
                sample['dapi'],
                min_nucleus_area=3000,
                max_nucleus_area=30000
            )
        else:  # adaptive
            result = detect_with_adaptive_box(
                sample['dapi'],
                sample['actn2'],
                min_nucleus_area=3000,
                max_nucleus_area=30000,
                min_zlines=15,
                zline_threshold=0.03
            )
        
        # Unpack result - detect functions return (boxes, cell_groups, regions)
        if isinstance(result, tuple):
            pred_boxes_raw = result[0]  # First element is boxes
        else:
            pred_boxes_raw = result
        
        # Convert pred boxes from [x1, y1, x2, y2] to (minr, minc, maxr, maxc) format
        # Note: x1,y1,x2,y2 = (minc, minr, maxc, maxr) in image coordinates
        pred_boxes = []
        for box in pred_boxes_raw:
            if isinstance(box, (list, np.ndarray)) and len(box) == 4:
                x1, y1, x2, y2 = box
                # Convert: (minr, minc, maxr, maxc) = (y1, x1, y2, x2)
                pred_boxes.append({'bbox': (y1, x1, y2, x2)})
        
        # Get GT boxes
        gt_boxes = get_gt_boxes_from_mask(sample['mask'])
        
        # Match
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        sample_results.append({
            'id': sample['id'],
            'n_pred': len(pred_boxes),
            'n_gt': len(gt_boxes),
            'tp': tp, 'fp': fp, 'fn': fn
        })
    
    # Compute metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'method': method,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn,
        'samples': sample_results
    }


def main():
    print("=" * 60)
    print("Adaptive vs DAPI Only 检测消融实验")
    print("=" * 60)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "test_ids.txt"
    output_dir = project_dir / "experiments" / "ablation_detection"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    print("\n[1] Loading test samples...")
    samples = load_test_samples(data_dir, split_file, n_samples=20)
    print(f"    Loaded {len(samples)} samples")
    
    # Evaluate DAPI Only
    print("\n[2] Evaluating DAPI Only method...")
    dapi_results = evaluate_detection_method(samples, 'dapi_only')
    
    # Evaluate Adaptive
    print("\n[3] Evaluating Adaptive method...")
    adaptive_results = evaluate_detection_method(samples, 'adaptive')
    
    # Print comparison
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"{'Method':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    print(f"{'DAPI Only':<15} {dapi_results['precision']:>10.3f} {dapi_results['recall']:>10.3f} {dapi_results['f1']:>10.3f}")
    print(f"{'Adaptive':<15} {adaptive_results['precision']:>10.3f} {adaptive_results['recall']:>10.3f} {adaptive_results['f1']:>10.3f}")
    print("-" * 45)
    
    # Delta
    delta_f1 = adaptive_results['f1'] - dapi_results['f1']
    winner = "Adaptive" if delta_f1 > 0 else "DAPI Only"
    print(f"\n{'Winner:':<15} {winner} (F1 差异: {delta_f1:+.3f})")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(samples),
        'dapi_only': dapi_results,
        'adaptive': adaptive_results,
        'delta_f1': delta_f1,
        'winner': winner
    }
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
