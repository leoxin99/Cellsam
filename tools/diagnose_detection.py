"""
检测问题诊断脚本

分析为什么 DAPI 检测只覆盖 43% 的 GT 细胞 (4.3/10)

诊断内容:
1. GT 框 vs DAPI 检测框的匹配情况
2. 漏检细胞的特征 (位置、大小、形状)
3. 误检框的特征
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from augmented_dataset import AugmentedAllenDataset
from detection.dapi import detect_nuclei, merge_close_nuclei, create_bounding_boxes

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def extract_gt_boxes_from_mask(mask):
    """Extract bounding boxes from GT instance mask."""
    from skimage import measure
    boxes = []
    for region in measure.regionprops(mask):
        y1, x1, y2, x2 = region.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes


def run_dapi_detection(dapi_channel, image_shape):
    """Run DAPI detection pipeline."""
    min_area = 500
    max_area = 30000
    
    nuclei = detect_nuclei(dapi_channel, min_area=min_area, max_area=max_area)
    cell_groups = merge_close_nuclei(nuclei, use_relative_distance=True)
    boxes = create_bounding_boxes(cell_groups, image_shape)
    
    return boxes, nuclei, cell_groups


def analyze_detection():
    """Main analysis function."""
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')[:20]
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    
    print(f"Analyzing {len(dataset)} samples...")
    
    stats = {
        'samples': [],
        'total_gt': 0,
        'total_pred': 0,
        'total_matched': 0,
        'total_missed': 0,
        'total_false_positive': 0,
        'missed_characteristics': [],
        'fp_characteristics': [],
    }
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        
        # Get data
        image = sample['image'].numpy()  # (3, H, W) - [BF, DAPI, Actn2]
        mask = sample['mask'].numpy().astype(np.int32)
        
        dapi_channel = image[1]  # DAPI is channel 1
        image_shape = (image.shape[1], image.shape[2])
        
        # Run detection
        pred_boxes, nuclei, cell_groups = run_dapi_detection(dapi_channel, image_shape)
        
        # Get GT boxes
        gt_boxes = extract_gt_boxes_from_mask(mask)
        
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        
        stats['total_gt'] += n_gt
        stats['total_pred'] += n_pred
        
        # Match GT to predictions
        matched_gt = set()
        matched_pred = set()
        
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_j = -1
            for j, pred_box in enumerate(pred_boxes):
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_iou >= 0.3:  # Match threshold
                matched_gt.add(i)
                matched_pred.add(best_j)
        
        n_matched = len(matched_gt)
        n_missed = n_gt - n_matched
        n_fp = n_pred - len(matched_pred)
        
        stats['total_matched'] += n_matched
        stats['total_missed'] += n_missed
        stats['total_false_positive'] += n_fp
        
        # Analyze missed GT cells
        for i, gt_box in enumerate(gt_boxes):
            if i not in matched_gt:
                x1, y1, x2, y2 = gt_box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if on edge
                on_edge = (x1 < 50 or y1 < 50 or 
                          x2 > image_shape[1] - 50 or 
                          y2 > image_shape[0] - 50)
                
                stats['missed_characteristics'].append({
                    'sample_idx': idx,
                    'area': area,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0,
                    'center_x': center_x,
                    'center_y': center_y,
                    'on_edge': on_edge,
                })
        
        # Analyze false positive detections
        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_pred:
                x1, y1, x2, y2 = pred_box
                stats['fp_characteristics'].append({
                    'sample_idx': idx,
                    'area': (x2-x1) * (y2-y1),
                    'width': x2-x1,
                    'height': y2-y1,
                })
        
        stats['samples'].append({
            'idx': idx,
            'n_gt': n_gt,
            'n_pred': n_pred,
            'n_matched': n_matched,
            'n_missed': n_missed,
            'n_fp': n_fp,
            'n_nuclei_detected': len(nuclei),
            'n_cell_groups': len(cell_groups),
        })
    
    # Summarize
    print("\n" + "="*60)
    print("检测问题诊断结果")
    print("="*60)
    
    print(f"\n【总体统计】")
    print(f"  总 GT 细胞数:        {stats['total_gt']}")
    print(f"  总预测框数:          {stats['total_pred']}")
    print(f"  成功匹配数:          {stats['total_matched']} ({stats['total_matched']/stats['total_gt']*100:.1f}%)")
    print(f"  漏检数:              {stats['total_missed']} ({stats['total_missed']/stats['total_gt']*100:.1f}%)")
    print(f"  误检数 (FP):         {stats['total_false_positive']}")
    
    # Analyze missed characteristics
    if stats['missed_characteristics']:
        missed = stats['missed_characteristics']
        areas = [m['area'] for m in missed]
        widths = [m['width'] for m in missed]
        heights = [m['height'] for m in missed]
        on_edge = sum(1 for m in missed if m['on_edge'])
        
        print(f"\n【漏检细胞特征】(共 {len(missed)} 个)")
        print(f"  面积:  均值={np.mean(areas):.0f}, 中位数={np.median(areas):.0f}, 范围=[{np.min(areas):.0f}, {np.max(areas):.0f}]")
        print(f"  宽度:  均值={np.mean(widths):.0f}, 中位数={np.median(widths):.0f}")
        print(f"  高度:  均值={np.mean(heights):.0f}, 中位数={np.median(heights):.0f}")
        print(f"  边缘:  {on_edge} 个 ({on_edge/len(missed)*100:.1f}%) 位于图像边缘")
        
        # Size distribution analysis
        small_cells = sum(1 for a in areas if a < 40000)
        medium_cells = sum(1 for a in areas if 40000 <= a < 200000)
        large_cells = sum(1 for a in areas if a >= 200000)
        print(f"\n  尺寸分布:")
        print(f"    小细胞 (<40K):   {small_cells} ({small_cells/len(missed)*100:.1f}%)")
        print(f"    中等 (40K-200K): {medium_cells} ({medium_cells/len(missed)*100:.1f}%)")
        print(f"    大细胞 (>200K):  {large_cells} ({large_cells/len(missed)*100:.1f}%)")
    
    # Analyze why nuclei detection fails
    print(f"\n【细胞核检测分析】")
    total_nuclei = sum(s['n_nuclei_detected'] for s in stats['samples'])
    total_groups = sum(s['n_cell_groups'] for s in stats['samples'])
    print(f"  检测到核总数:        {total_nuclei} (均值 {total_nuclei/len(stats['samples']):.1f}/样本)")
    print(f"  合并后细胞组数:      {total_groups} (均值 {total_groups/len(stats['samples']):.1f}/样本)")
    print(f"  GT 细胞平均数:       {stats['total_gt']/len(stats['samples']):.1f}/样本")
    
    # Save detailed results
    output_dir = Path("experiments/detection_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "diagnosis.json", 'w') as f:
        json.dump({
            'summary': {
                'total_gt': stats['total_gt'],
                'total_pred': stats['total_pred'],
                'total_matched': stats['total_matched'],
                'total_missed': stats['total_missed'],
                'total_fp': stats['total_false_positive'],
                'recall': stats['total_matched'] / stats['total_gt'] if stats['total_gt'] > 0 else 0,
                'precision': stats['total_matched'] / stats['total_pred'] if stats['total_pred'] > 0 else 0,
            },
            'per_sample': stats['samples'],
        }, f, indent=2)
    
    print(f"\n详细结果已保存: {output_dir / 'diagnosis.json'}")
    
    return stats


if __name__ == "__main__":
    analyze_detection()
