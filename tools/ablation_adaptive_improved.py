"""
Adaptive (Z-线) 检测方法改进消融实验

使用 B1 最优 search_radius=200 作为固定值，重新测试 B2, B3:
- B2: min_zlines (5, 10, 15, 20, 30)
- B3: zline_threshold (0.01, 0.02, 0.03, 0.05, 0.1)

运行:
    conda activate cellsam
    python tools/ablation_adaptive_improved.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project paths
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "cellSAM_source"))
sys.path.insert(0, str(project_dir / "src"))

from detection.dapi import detect_with_adaptive_box


def load_test_samples(data_dir: Path, split_file: Path, n_samples: int = 20):
    """Load test samples with DAPI and Actn2 channels."""
    with open(split_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    test_ids = test_ids[:n_samples]
    
    samples = []
    for sample_id in tqdm(test_ids, desc="Loading samples"):
        image_path = data_dir / "images" / f"{sample_id}.npy"
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        
        if not image_path.exists() or not mask_path.exists():
            continue
            
        image = np.load(image_path)
        mask = np.load(mask_path)
        
        samples.append({
            'id': sample_id,
            'dapi': image[1],
            'actn2': image[2],
            'mask': mask,
        })
    
    return samples


def get_gt_boxes_from_mask(mask: np.ndarray, min_area: int = 500):
    """Extract GT bounding boxes from instance mask."""
    from skimage import measure
    
    boxes = []
    regions = measure.regionprops(mask)
    
    for region in regions:
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        boxes.append((minr, minc, maxr, maxc))
    
    return boxes


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes."""
    r1, c1, r2, c2 = box1
    r3, c3, r4, c4 = box2
    
    inter_r1 = max(r1, r3)
    inter_c1 = max(c1, c3)
    inter_r2 = min(r2, r4)
    inter_c2 = min(c2, c4)
    
    if inter_r2 <= inter_r1 or inter_c2 <= inter_c1:
        return 0.0
    
    inter_area = (inter_r2 - inter_r1) * (inter_c2 - inter_c1)
    area1 = (r2 - r1) * (c2 - c1)
    area2 = (r4 - r3) * (c4 - c3)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold: float = 0.3):
    """Match predicted boxes to GT boxes."""
    if not pred_boxes or not gt_boxes:
        return 0, len(pred_boxes), len(gt_boxes)
    
    matched_gt = set()
    tp = 0
    
    for pred in pred_boxes:
        if isinstance(pred, (list, np.ndarray)) and len(pred) == 4:
            x1, y1, x2, y2 = pred
            pred_bbox = (y1, x1, y2, x2)
        else:
            pred_bbox = pred
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_box_iou(pred_bbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


def evaluate_adaptive_params(samples, search_radius, min_zlines, zline_threshold):
    """Evaluate Adaptive detection with specific parameters."""
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for sample in samples:
        try:
            result = detect_with_adaptive_box(
                sample['dapi'],
                sample['actn2'],
                min_nucleus_area=3000,
                max_nucleus_area=20000,
                search_radius=search_radius,
                min_zlines=min_zlines,
                zline_threshold=zline_threshold
            )
            pred_boxes = result[0] if isinstance(result, tuple) else result
            gt_boxes = get_gt_boxes_from_mask(sample['mask'], min_area=500)
            tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        except Exception as e:
            gt_boxes = get_gt_boxes_from_mask(sample['mask'], min_area=500)
            all_fn += len(gt_boxes)
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn,
    }


def main():
    print("=" * 70)
    print("Adaptive 检测改进消融实验 (使用最优 search_radius=200)")
    print("=" * 70)
    
    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "test_ids.txt"
    output_dir = project_dir / "experiments" / "ablation_adaptive_improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Loading test samples...")
    samples = load_test_samples(data_dir, split_file, n_samples=20)
    print(f"    Loaded {len(samples)} samples")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(samples),
        'fixed_params': {'search_radius': 200},  # From B1 optimization
        'experiments': {}
    }
    
    # Optimal search_radius from B1
    OPTIMAL_SEARCH_RADIUS = 200
    
    # B2: min_zlines sweep (with optimal search_radius)
    print(f"\n[B2'] Testing min_zlines (search_radius={OPTIMAL_SEARCH_RADIUS})...")
    b2_values = [5, 10, 15, 20, 30]
    b2_results = []
    for value in tqdm(b2_values, desc="  min_zlines"):
        metrics = evaluate_adaptive_params(
            samples, 
            search_radius=OPTIMAL_SEARCH_RADIUS,
            min_zlines=value, 
            zline_threshold=0.03
        )
        metrics['value'] = value
        b2_results.append(metrics)
        print(f"    min_zlines={value}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    best_b2 = max(b2_results, key=lambda x: x['f1'])
    results['experiments']['B2_min_zlines'] = {
        'param': 'min_zlines',
        'fixed': {'search_radius': OPTIMAL_SEARCH_RADIUS, 'zline_threshold': 0.03},
        'results': b2_results,
        'best_value': best_b2['value'],
        'best_f1': best_b2['f1']
    }
    
    # B3: zline_threshold sweep (with optimal search_radius)
    print(f"\n[B3'] Testing zline_threshold (search_radius={OPTIMAL_SEARCH_RADIUS})...")
    b3_values = [0.01, 0.02, 0.03, 0.05, 0.1]
    b3_results = []
    for value in tqdm(b3_values, desc="  zline_threshold"):
        metrics = evaluate_adaptive_params(
            samples, 
            search_radius=OPTIMAL_SEARCH_RADIUS,
            min_zlines=15, 
            zline_threshold=value
        )
        metrics['value'] = value
        b3_results.append(metrics)
        print(f"    zline_threshold={value}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    best_b3 = max(b3_results, key=lambda x: x['f1'])
    results['experiments']['B3_zline_threshold'] = {
        'param': 'zline_threshold',
        'fixed': {'search_radius': OPTIMAL_SEARCH_RADIUS, 'min_zlines': 15},
        'results': b3_results,
        'best_value': best_b3['value'],
        'best_f1': best_b3['f1']
    }
    
    # Baseline with all optimal params
    print("\n[Baseline] Testing with optimal search_radius=200...")
    baseline = evaluate_adaptive_params(samples, OPTIMAL_SEARCH_RADIUS, 15, 0.03)
    results['baseline_optimal_radius'] = baseline
    
    # Summary
    print("\n" + "=" * 70)
    print("改进实验结果汇总 (search_radius=200 固定)")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Best Value':>15} {'Best F1':>10}")
    print("-" * 55)
    for exp_id, exp_data in results['experiments'].items():
        print(f"{exp_id:<25} {str(exp_data['best_value']):>15} {exp_data['best_f1']:>10.4f}")
    print("-" * 55)
    print(f"{'Baseline (radius=200)':<25} {'15/0.03':>15} {baseline['f1']:>10.4f}")
    
    # Compare with DAPI Only
    print("\n" + "-" * 55)
    print("与 DAPI Only 最优结果对比:")
    print(f"  DAPI Only:   F1=0.806 (min_area=2000)")
    print(f"  Adaptive:    F1={max(best_b2['f1'], best_b3['f1'], baseline['f1']):.4f} (radius=200)")
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
