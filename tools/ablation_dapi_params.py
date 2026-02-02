"""
DAPI 检测参数消融实验

测试 Part A 参数:
- A1: min_nucleus_area (1000, 2000, 3000, 5000)
- A2: max_nucleus_area (10000, 15000, 20000, 30000)
- A3: use_relative_distance (True=相对距离 vs False=固定距离)

运行:
    conda activate cellsam
    python tools/ablation_dapi_params.py
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


def load_test_samples(data_dir: Path, split_file: Path, n_samples: int = 20):
    """Load test samples with DAPI channel."""
    with open(split_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    test_ids = test_ids[:n_samples]
    
    samples = []
    for sample_id in tqdm(test_ids, desc="Loading samples"):
        image_path = data_dir / "images" / f"{sample_id}.npy"
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        
        if not image_path.exists() or not mask_path.exists():
            continue
            
        image = np.load(image_path)  # (3, 1024, 1024)
        mask = np.load(mask_path)    # (1024, 1024)
        
        samples.append({
            'id': sample_id,
            'dapi': image[1],  # Channel 1 = DAPI
            'mask': mask,
            'image_shape': image.shape[1:],
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
        # Convert pred from [x1,y1,x2,y2] to (minr,minc,maxr,maxc)
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


def run_detection_pipeline(dapi, image_shape, min_area, max_area, use_relative_dist):
    """Run complete detection pipeline with specified parameters."""
    # Step 1: Detect nuclei
    regions = detect_nuclei(dapi, min_area, max_area)
    
    # Step 2: Merge close nuclei
    cell_groups = merge_close_nuclei(
        regions, 
        use_relative_distance=use_relative_dist
    )
    
    # Step 3: Create boxes
    boxes = create_bounding_boxes(cell_groups, image_shape)
    
    return boxes


def evaluate_params(samples, min_area, max_area, use_relative_dist):
    """Evaluate detection with specific parameters."""
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for sample in samples:
        # Run detection
        pred_boxes = run_detection_pipeline(
            sample['dapi'], 
            sample['image_shape'],
            min_area, 
            max_area, 
            use_relative_dist
        )
        
        # Get GT boxes
        gt_boxes = get_gt_boxes_from_mask(sample['mask'], min_area=500)
        
        # Match
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    # Compute metrics
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
    print("DAPI 检测参数消融实验 (Part A)")
    print("=" * 70)
    
    # Paths
    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "test_ids.txt"
    output_dir = project_dir / "experiments" / "ablation_dapi_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    print("\n[1] Loading test samples...")
    samples = load_test_samples(data_dir, split_file, n_samples=20)
    print(f"    Loaded {len(samples)} samples")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(samples),
        'experiments': {}
    }
    
    # A1: min_nucleus_area sweep
    print("\n[A1] Testing min_nucleus_area...")
    a1_values = [1000, 2000, 3000, 5000, 7000]
    a1_results = []
    for value in tqdm(a1_values, desc="  min_area"):
        metrics = evaluate_params(samples, min_area=value, max_area=20000, use_relative_dist=True)
        metrics['value'] = value
        a1_results.append(metrics)
        print(f"    min_area={value}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    best_a1 = max(a1_results, key=lambda x: x['f1'])
    results['experiments']['A1_min_nucleus_area'] = {
        'param': 'min_nucleus_area',
        'results': a1_results,
        'best_value': best_a1['value'],
        'best_f1': best_a1['f1']
    }
    
    # A2: max_nucleus_area sweep
    print("\n[A2] Testing max_nucleus_area...")
    a2_values = [10000, 15000, 20000, 25000, 30000]
    a2_results = []
    for value in tqdm(a2_values, desc="  max_area"):
        metrics = evaluate_params(samples, min_area=3000, max_area=value, use_relative_dist=True)
        metrics['value'] = value
        a2_results.append(metrics)
        print(f"    max_area={value}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    best_a2 = max(a2_results, key=lambda x: x['f1'])
    results['experiments']['A2_max_nucleus_area'] = {
        'param': 'max_nucleus_area',
        'results': a2_results,
        'best_value': best_a2['value'],
        'best_f1': best_a2['f1']
    }
    
    # A3: Relative vs Fixed distance
    print("\n[A3] Testing merge distance strategy...")
    a3_results = []
    for use_rel, label in [(True, 'relative'), (False, 'fixed_373px')]:
        metrics = evaluate_params(samples, min_area=3000, max_area=20000, use_relative_dist=use_rel)
        metrics['value'] = label
        a3_results.append(metrics)
        print(f"    {label}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    best_a3 = max(a3_results, key=lambda x: x['f1'])
    results['experiments']['A3_merge_distance'] = {
        'param': 'use_relative_distance',
        'results': a3_results,
        'best_value': best_a3['value'],
        'best_f1': best_a3['f1']
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Best Value':>15} {'Best F1':>10}")
    print("-" * 55)
    for exp_id, exp_data in results['experiments'].items():
        print(f"{exp_id:<25} {str(exp_data['best_value']):>15} {exp_data['best_f1']:>10.4f}")
    
    # Current baseline
    print("-" * 55)
    baseline = evaluate_params(samples, min_area=3000, max_area=20000, use_relative_dist=True)
    print(f"{'Baseline (current)':<25} {'3000/20000/rel':>15} {baseline['f1']:>10.4f}")
    results['baseline'] = baseline
    
    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
