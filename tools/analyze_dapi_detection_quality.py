"""
DAPI 检测框质量评估

分析目的：
1. DAPI 检测框与 GT 框的 IoU
2. 检测召回率和准确率
3. 框尺寸偏差分析
4. 评估是否需要进一步优化 DAPI 检测

这个评估决定了：
- 如果 DAPI 检测框质量高 → SAM 输入足够好，问题在分割模型
- 如果 DAPI 检测框质量低 → 需要先优化检测，再改进分割
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from augmented_dataset import AugmentedAllenDataset
from detection.dapi import detect_and_create_boxes


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def get_gt_boxes_from_mask(mask):
    """Extract bounding boxes from instance mask."""
    boxes = []
    labels = np.unique(mask)
    labels = labels[labels > 0]
    
    for label in labels:
        region = (mask == label).astype(np.uint8)
        props = measure.regionprops(region)
        if props:
            y1, x1, y2, x2 = props[0].bbox
            boxes.append([x1, y1, x2, y2])
    
    return boxes


def analyze_dapi_detection():
    """Main analysis function."""
    print("="*70)
    print("DAPI 检测框质量评估")
    print("="*70)
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')[:30]
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    print(f"样本数: {len(dataset)}")
    
    all_results = []
    
    for idx in tqdm(range(len(dataset)), desc="分析中"):
        sample = dataset[idx]
        
        # Get image and GT
        image = sample['image'].numpy()  # (3, H, W)
        gt_mask = sample['mask'].numpy()
        
        # Get DAPI channel (channel 1 in normalized data)
        dapi = image[1]  # Already normalized to [0,1]
        
        # Detect using DAPI - need to handle normalized data
        # The detection function uses Otsu, which works on normalized data
        try:
            result = detect_and_create_boxes(dapi)
            # Returns tuple: (boxes, cell_groups, regions)
            if isinstance(result, tuple):
                detected_boxes = result[0]  # First element is boxes
            else:
                detected_boxes = result
            
            # Handle None or empty
            if detected_boxes is None:
                detected_boxes = []
        except Exception as e:
            print(f"Detection error for sample {idx}: {e}")
            detected_boxes = []
        
        # Get GT boxes
        gt_boxes = get_gt_boxes_from_mask(gt_mask)
        
        # Match detected boxes to GT boxes
        matched_gt = set()
        matched_det = set()
        
        for i, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_box_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= 0.3 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_det.add(i)
        
        # Calculate metrics
        n_gt = len(gt_boxes)
        n_det = len(detected_boxes)
        n_tp = len(matched_gt)
        
        precision = n_tp / n_det if n_det > 0 else 0
        recall = n_tp / n_gt if n_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_results.append({
            'n_gt': n_gt,
            'n_det': n_det,
            'n_tp': n_tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    
    # Aggregate results
    total_gt = sum(r['n_gt'] for r in all_results)
    total_det = sum(r['n_det'] for r in all_results)
    total_tp = sum(r['n_tp'] for r in all_results)
    
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    
    print("\n" + "="*70)
    print("DAPI 检测框评估结果")
    print("="*70)
    
    print(f"\n【总体统计】")
    print(f"  GT 框总数:      {total_gt}")
    print(f"  检测框总数:     {total_det}")
    print(f"  匹配成功 (TP):  {total_tp}")
    
    print(f"\n【检测质量指标】(IoU ≥ 0.3)")
    print(f"  平均 Precision: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
    print(f"  平均 Recall:    {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  平均 F1:        {avg_f1:.4f} ({avg_f1*100:.1f}%)")
    
    # Determine if DAPI detection is sufficient
    print(f"\n【评估结论】")
    if avg_recall < 0.5:
        print(f"  ⚠️ 召回率过低 ({avg_recall:.2f}) - DAPI 检测漏检严重")
        print(f"  → 问题根源: DAPI 通道已被二值化，丢失了原始强度信息")
        print(f"  → 建议: 检查数据预处理流程，保留原始 DAPI 强度")
    elif avg_precision < 0.5:
        print(f"  ⚠️ 精确率过低 ({avg_precision:.2f}) - DAPI 检测误检过多")
        print(f"  → 建议: 调整 DAPI 检测参数")
    elif avg_f1 >= 0.7:
        print(f"  ✅ DAPI 检测质量良好 (F1={avg_f1:.2f})")
        print(f"  → SAM 分割质量主要受分割模型本身影响")
    else:
        print(f"  ⚠️ DAPI 检测质量中等 (F1={avg_f1:.2f})")
        print(f"  → 建议同时优化检测和分割")
    
    # Compare with evaluation using GT boxes
    print(f"\n【关键对比】")
    print(f"  当前评估使用: GT 框 (从 mask 提取)")
    print(f"  实际推理使用: DAPI 检测框")
    if avg_recall < 0.7:
        print(f"  ⚠️ 实际推理性能会比评估结果更低！")
        print(f"     因为 DAPI 漏检的细胞无法被分割")
    
    # Save results
    output_dir = Path("experiments/dapi_detection_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "detection_quality.json", 'w') as f:
        json.dump({
            'total': {
                'n_gt': total_gt,
                'n_det': total_det,
                'n_tp': total_tp,
            },
            'metrics': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1),
            },
            'per_sample': all_results,
        }, f, indent=2)
    
    print(f"\n结果保存至: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    analyze_dapi_detection()
