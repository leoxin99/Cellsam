"""
DAPI Only 标准化参数消融实验 (Val 集)

使用 val_ids.txt 全部 71 个样本进行参数消融，
避免在 test 集上过拟合参数。

测试参数:
- A1: min_nucleus_area (1000, 2000, 3000, 4000, 5000)
- A2: max_nucleus_area (15000, 20000, 25000, 30000)
- A3: use_relative_distance (True vs False)

运行:
    conda activate cellsam
    python tools/ablation_dapi_val.py
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

from detection.dapi import detect_nuclei, merge_close_nuclei, create_bounding_boxes


def load_samples(data_dir: Path, split_file: Path, n_samples: int = None):
    """Load samples from split file."""
    with open(split_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    if n_samples:
        sample_ids = sample_ids[:n_samples]
    
    samples = []
    for sample_id in tqdm(sample_ids, desc="Loading samples"):
        image_path = data_dir / "images" / f"{sample_id}.npy"
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        
        if not image_path.exists() or not mask_path.exists():
            continue
            
        image = np.load(image_path)
        mask = np.load(mask_path)
        
        samples.append({
            'id': sample_id,
            'dapi': image[1],  # Channel 1 = DAPI
            'mask': mask,
            'image_shape': image.shape[1:],
        })
    
    return samples


def get_gt_boxes_from_mask(mask: np.ndarray, min_area: int = 500):
    """Extract GT bounding boxes from instance mask.

    Note:
        `min_area` is kept only for backward compatibility. GT boxes are not
        area-filtered in this evaluation path.
    """
    from skimage import measure
    
    boxes = []
    regions = measure.regionprops(mask)
    
    for region in regions:
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


def run_detection(dapi, image_shape, min_area, max_area, use_relative_dist):
    """Run detection pipeline."""
    regions = detect_nuclei(dapi, min_area, max_area)
    cell_groups = merge_close_nuclei(regions, use_relative_distance=use_relative_dist)
    boxes = create_bounding_boxes(cell_groups, image_shape)
    return boxes


def evaluate(samples, min_area, max_area, use_relative_dist):
    """Evaluate detection with parameters."""
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for sample in samples:
        pred_boxes = run_detection(
            sample['dapi'], sample['image_shape'],
            min_area, max_area, use_relative_dist
        )
        gt_boxes = get_gt_boxes_from_mask(sample['mask'], min_area=500)
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': all_tp, 'fp': all_fp, 'fn': all_fn,
    }


def main():
    print("=" * 70)
    print("DAPI Only 标准化消融实验 (Val Set - 71 Samples)")
    print("=" * 70)
    
    # Paths
    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "val_ids.txt"  # VAL SET
    output_dir = project_dir / "experiments" / "ablation_dapi_val"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ALL val samples
    print("\n[1] Loading VAL samples (all 71)...")
    samples = load_samples(data_dir, split_file, n_samples=None)  # ALL
    print(f"    Loaded {len(samples)} samples")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'split': 'val',
        'n_samples': len(samples),
        'experiments': {}
    }
    
    # A1: min_nucleus_area
    print("\n[A1] min_nucleus_area 扫描...")
    a1_values = [1000, 1500, 2000, 2500, 3000, 4000, 5000]
    a1_results = []
    for v in tqdm(a1_values, desc="  min_area"):
        m = evaluate(samples, min_area=v, max_area=20000, use_relative_dist=True)
        m['value'] = v
        a1_results.append(m)
        print(f"    min_area={v}: F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    
    best_a1 = max(a1_results, key=lambda x: x['f1'])
    results['experiments']['A1_min_nucleus_area'] = {
        'param': 'min_nucleus_area',
        'fixed': {'max_area': 20000, 'relative': True},
        'results': a1_results,
        'best_value': best_a1['value'],
        'best_f1': best_a1['f1']
    }
    
    # A2: max_nucleus_area (using best min_area)
    print(f"\n[A2] max_nucleus_area 扫描 (min_area={best_a1['value']})...")
    a2_values = [15000, 20000, 25000, 30000, 40000]
    a2_results = []
    for v in tqdm(a2_values, desc="  max_area"):
        m = evaluate(samples, min_area=best_a1['value'], max_area=v, use_relative_dist=True)
        m['value'] = v
        a2_results.append(m)
        print(f"    max_area={v}: F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    
    best_a2 = max(a2_results, key=lambda x: x['f1'])
    results['experiments']['A2_max_nucleus_area'] = {
        'param': 'max_nucleus_area',
        'fixed': {'min_area': best_a1['value'], 'relative': True},
        'results': a2_results,
        'best_value': best_a2['value'],
        'best_f1': best_a2['f1']
    }
    
    # A3: merge distance strategy
    print(f"\n[A3] merge_distance 策略 (min={best_a1['value']}, max={best_a2['value']})...")
    a3_results = []
    for use_rel, label in [(True, 'relative_1.2x'), (False, 'fixed_373px')]:
        m = evaluate(samples, min_area=best_a1['value'], max_area=best_a2['value'], use_relative_dist=use_rel)
        m['value'] = label
        a3_results.append(m)
        print(f"    {label}: F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    
    best_a3 = max(a3_results, key=lambda x: x['f1'])
    results['experiments']['A3_merge_distance'] = {
        'param': 'use_relative_distance',
        'fixed': {'min_area': best_a1['value'], 'max_area': best_a2['value']},
        'results': a3_results,
        'best_value': best_a3['value'],
        'best_f1': best_a3['f1']
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("标准化实验结果汇总 (Val Set)")
    print("=" * 70)
    print(f"{'实验':<25} {'最佳值':>15} {'最佳 F1':>10}")
    print("-" * 55)
    for exp_id, data in results['experiments'].items():
        print(f"{exp_id:<25} {str(data['best_value']):>15} {data['best_f1']:>10.4f}")
    
    # Final optimal
    print("-" * 55)
    final = evaluate(samples, best_a1['value'], best_a2['value'], 
                     use_relative_dist=(best_a3['value'] == 'relative_1.2x'))
    combo_str = f"{best_a1['value']}/{best_a2['value']}/{best_a3['value'][:3]}"
    print(f"{'最优组合':<25} {combo_str:>15} {final['f1']:>10.4f}")
    
    results['optimal'] = {
        'min_nucleus_area': best_a1['value'],
        'max_nucleus_area': best_a2['value'],
        'use_relative_distance': best_a3['value'] == 'relative_1.2x',
        'f1': final['f1'],
        'precision': final['precision'],
        'recall': final['recall']
    }
    
    # Save
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
