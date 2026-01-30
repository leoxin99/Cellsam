"""
evaluate_box_generation.py

功能: 对比评估检测框生成效果 (DAPI+Actn2 vs Adaptive vs GT)
所属实验: E18 扩展 - 框生成评估
创建日期: 2026-01-23
最后修改: 2026-01-23
版本: v2

评估指标:
- Box IoU: 框与GT框的重叠度
- Cell Coverage: 框是否完全包含细胞
- Box Efficiency: 框是否过大
- Size Ratio: 预测框与GT框的大小比
- FP/FN: 误检/漏检统计

依赖函数:
- detection.dapi.detect_cardiomyocytes (DAPI+Actn2 检测)
- detection.dapi.detect_with_adaptive_box (Z-线自适应框)

边缘过滤逻辑:
- filter_boxes_by_edge_overlap(): 框有 >50% 在图外则排除

更新日志:
- 2026-01-23 v2: 使用 detect_cardiomyocytes, 添加边缘过滤
- 2026-01-23 v1: 初始版本

Usage:
    conda activate cellsam
    python tools/evaluate_box_generation.py [--visualize] [--num_samples 5]
"""
import numpy as np
import tifffile
import os
import sys
import argparse
from skimage import measure

sys.path.insert(0, 'src')

from detection.dapi import (
    detect_cardiomyocytes,     # 正确的 DAPI+Actn2 检测方法
    detect_with_adaptive_box,
    detect_nuclei,
    merge_close_nuclei,
    create_bounding_boxes
)


def extract_gt_boxes(gt_mask: np.ndarray) -> list:
    """
    Extract bounding boxes from GT instance mask.
    
    Note: GT boxes should NOT be edge-filtered. GT represents all correctly
    annotated cells. Our detection should try to match GT (excluding edges
    where cells are incomplete).
    
    Args:
        gt_mask: Instance segmentation mask
    
    Returns:
        List of dicts with 'box', 'area', 'centroid', 'label' for each cell
    """
    boxes = []
    
    for region in measure.regionprops(gt_mask.astype(int)):
        y1, x1, y2, x2 = region.bbox
        boxes.append({
            'box': [x1, y1, x2, y2],  # [x1, y1, x2, y2]
            'area': region.area,
            'centroid': region.centroid,  # (y, x)
            'label': region.label
        })
    return boxes


def filter_boxes_by_edge_overlap(boxes, image_shape, max_outside_ratio=0.5):
    """
    过滤掉有超过 max_outside_ratio 面积在图外的框。
    
    Args:
        boxes: 边界框列表 [[x1, y1, x2, y2], ...]
        image_shape: (H, W)
        max_outside_ratio: 最大允许在图外的比例 (默认 0.5 = 50%)
    
    Returns:
        过滤后的框列表和保留的索引
    """
    h, w = image_shape
    filtered = []
    kept_indices = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # 框总面积
        total_area = (x2 - x1) * (y2 - y1)
        if total_area <= 0:
            continue
        
        # 框在图内的部分
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(w, x2)
        y2_clip = min(h, y2)
        
        inside_area = max(0, x2_clip - x1_clip) * max(0, y2_clip - y1_clip)
        inside_ratio = inside_area / total_area
        
        # 如果图内部分 >= (1 - max_outside_ratio)，则保留
        if inside_ratio >= (1 - max_outside_ratio):
            filtered.append(box)
            kept_indices.append(i)
    
    return filtered, kept_indices


def filter_nuclei_by_edge_distance(nuclei_groups, image_shape, min_edge_distance=100):
    """
    基于核心距边缘距离过滤细胞 (Dev Set 分析推荐阈值: 100-150px)
    
    Args:
        nuclei_groups: 细胞组列表，每组是 regionprops 列表
        image_shape: (H, W)
        min_edge_distance: 最小边缘距离 (默认 100px, Dev Set 分析)
    
    Returns:
        过滤后的细胞组列表
    """
    h, w = image_shape
    filtered_groups = []
    
    for group in nuclei_groups:
        # 计算组内所有核的中心
        centroids = [r.centroid for r in group]  # [(y, x), ...]
        
        # 使用组中心的加权平均作为细胞位置
        avg_y = sum(c[0] for c in centroids) / len(centroids)
        avg_x = sum(c[1] for c in centroids) / len(centroids)
        
        # 计算到最近边缘的距离
        edge_dist = min(avg_y, h - avg_y, avg_x, w - avg_x)
        
        # 距离 >= min_edge_distance 则保留
        if edge_dist >= min_edge_distance:
            filtered_groups.append(group)
    
    return filtered_groups


def box_iou(box1, box2):
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


def box_area(box):
    """Compute box area."""
    return (box[2] - box[0]) * (box[3] - box[1])


def cell_coverage(box, gt_mask, cell_id):
    """Compute what fraction of the cell is inside the box."""
    x1, y1, x2, y2 = box
    h, w = gt_mask.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cell_mask = (gt_mask == cell_id)
    cell_total = cell_mask.sum()
    
    if cell_total == 0:
        return 0.0
    
    box_mask = np.zeros_like(cell_mask)
    box_mask[y1:y2, x1:x2] = True
    
    covered = (cell_mask & box_mask).sum()
    return covered / cell_total


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.1):
    """
    Match predicted boxes to GT boxes using Hungarian algorithm.
    
    Returns:
        matches: List of matched pairs with metrics
        unmatched_pred: Indices of FP (False Positive) predictions
        unmatched_gt: Indices of FN (False Negative) GT boxes
    """
    from scipy.optimize import linear_sum_assignment
    
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = box_iou(pb, gb['box'])
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] > iou_threshold:
            matches.append({
                'pred_idx': i,
                'gt_idx': j,
                'iou': iou_matrix[i, j],
                'pred_box': pred_boxes[i],
                'gt_box': gt_boxes[j]
            })
            matched_pred.add(i)
            matched_gt.add(j)
    
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gt = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    
    return matches, unmatched_pred, unmatched_gt


def evaluate_boxes(pred_boxes, gt_boxes, gt_mask):
    """Evaluate predicted boxes against GT with FP/FN statistics."""
    matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes)
    
    results = {
        'num_pred': len(pred_boxes),
        'num_gt': len(gt_boxes),
        'num_matched': len(matches),
        'num_fp': len(unmatched_pred),  # False Positives
        'num_fn': len(unmatched_gt),     # False Negatives
        'fp_indices': unmatched_pred,
        'fn_indices': unmatched_gt,
        'box_iou': [],
        'cell_coverage': [],
        'size_ratio': [],
        'box_efficiency': []
    }
    
    for m in matches:
        pb = m['pred_box']
        gb = m['gt_box']
        
        results['box_iou'].append(m['iou'])
        
        cell_id = gb['label']
        cov = cell_coverage(pb, gt_mask, cell_id)
        results['cell_coverage'].append(cov)
        
        pred_a = box_area(pb)
        gt_a = box_area(gb['box'])
        results['size_ratio'].append(pred_a / gt_a if gt_a > 0 else 0)
        
        cell_area = gb['area']
        efficiency = cell_area / pred_a if pred_a > 0 else 0
        results['box_efficiency'].append(efficiency)
    
    return results


def visualize_matching(img, dapi_boxes, adaptive_boxes, gt_boxes, 
                       dapi_results, adaptive_results, fname):
    """Visualize matching results in Napari."""
    import napari
    
    viewer = napari.Viewer(title=f"Box Matching: {fname[:30]}")
    
    bf = img[0]
    actn2 = img[1]
    gt = img[9]
    
    viewer.add_image(bf, name='Brightfield')
    viewer.add_image(actn2, name='Actn2', colormap='hot', visible=False)
    viewer.add_labels(gt, name='GT_Mask')
    
    # GT boxes (White)
    if gt_boxes:
        gt_rects = [[[b['box'][1], b['box'][0]], [b['box'][1], b['box'][2]], 
                     [b['box'][3], b['box'][2]], [b['box'][3], b['box'][0]]] 
                    for b in gt_boxes]
        viewer.add_shapes(gt_rects, shape_type='polygon', edge_color='white',
                         face_color='transparent', edge_width=2, name='GT_Boxes')
    
    # DAPI matched (Green) vs unmatched (Red)
    dapi_matched = [i for i in range(len(dapi_boxes)) if i not in dapi_results['fp_indices']]
    
    if dapi_matched:
        matched_rects = [[[dapi_boxes[i][1], dapi_boxes[i][0]], 
                          [dapi_boxes[i][1], dapi_boxes[i][2]],
                          [dapi_boxes[i][3], dapi_boxes[i][2]], 
                          [dapi_boxes[i][3], dapi_boxes[i][0]]] 
                         for i in dapi_matched]
        viewer.add_shapes(matched_rects, shape_type='polygon', edge_color='green',
                         face_color='transparent', edge_width=2, name='DAPI_Matched')
    
    if dapi_results['fp_indices']:
        fp_rects = [[[dapi_boxes[i][1], dapi_boxes[i][0]], 
                     [dapi_boxes[i][1], dapi_boxes[i][2]],
                     [dapi_boxes[i][3], dapi_boxes[i][2]], 
                     [dapi_boxes[i][3], dapi_boxes[i][0]]] 
                    for i in dapi_results['fp_indices']]
        viewer.add_shapes(fp_rects, shape_type='polygon', edge_color='red',
                         face_color='transparent', edge_width=2, name='DAPI_FP')
    
    # Adaptive matched (Cyan) vs unmatched (Orange)
    adapt_matched = [i for i in range(len(adaptive_boxes)) if i not in adaptive_results['fp_indices']]
    
    if adapt_matched:
        matched_rects = [[[adaptive_boxes[i][1], adaptive_boxes[i][0]], 
                          [adaptive_boxes[i][1], adaptive_boxes[i][2]],
                          [adaptive_boxes[i][3], adaptive_boxes[i][2]], 
                          [adaptive_boxes[i][3], adaptive_boxes[i][0]]] 
                         for i in adapt_matched]
        viewer.add_shapes(matched_rects, shape_type='polygon', edge_color='cyan',
                         face_color='transparent', edge_width=3, name='Adaptive_Matched')
    
    if adaptive_results['fp_indices']:
        fp_rects = [[[adaptive_boxes[i][1], adaptive_boxes[i][0]], 
                     [adaptive_boxes[i][1], adaptive_boxes[i][2]],
                     [adaptive_boxes[i][3], adaptive_boxes[i][2]], 
                     [adaptive_boxes[i][3], adaptive_boxes[i][0]]] 
                    for i in adaptive_results['fp_indices']]
        viewer.add_shapes(fp_rects, shape_type='polygon', edge_color='orange',
                         face_color='transparent', edge_width=3, name='Adaptive_FP')
    
    # FN GT boxes (Magenta)
    if dapi_results['fn_indices']:
        fn_rects = [[[gt_boxes[i]['box'][1], gt_boxes[i]['box'][0]], 
                     [gt_boxes[i]['box'][1], gt_boxes[i]['box'][2]], 
                     [gt_boxes[i]['box'][3], gt_boxes[i]['box'][2]], 
                     [gt_boxes[i]['box'][3], gt_boxes[i]['box'][0]]] 
                    for i in dapi_results['fn_indices']]
        viewer.add_shapes(fn_rects, shape_type='polygon', edge_color='magenta',
                         face_color='transparent', edge_width=2, name='GT_FN')
    
    print(f"\nVisualization legend:")
    print(f"  ⬜ White = GT boxes")
    print(f"  🟢 Green = DAPI matched")
    print(f"  🔴 Red = DAPI FP (False Positive)")
    print(f"  🔵 Cyan = Adaptive matched")
    print(f"  🟠 Orange = Adaptive FP")
    print(f"  🟣 Magenta = GT FN (missed by detection)")
    
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--visualize', action='store_true', help='Open Napari visualization')
    args = parser.parse_args()
    
    data_dir = 'data/raw/allen_segmented_fields_full'
    files = sorted(os.listdir(data_dir))[:args.num_samples]
    
    print("=" * 60)
    print("Box Generation Evaluation: DAPI vs Adaptive vs GT")
    print("=" * 60)
    print("\nParameters (from Dev Set analysis 2026-01-25):")
    print("  - Edge filter: nucleus distance >= 100px")
    print("  - Binuclear merge: distance <= 373px (P95)")
    
    all_dapi = {'iou': [], 'coverage': [], 'size_ratio': [], 'efficiency': [], 'fp': 0, 'fn': 0, 'tp': 0}
    all_adaptive = {'iou': [], 'coverage': [], 'size_ratio': [], 'efficiency': [], 'fp': 0, 'fn': 0, 'tp': 0}
    
    for fname in files:
        print(f"\nProcessing {fname[:30]}...")
        img = tifffile.imread(os.path.join(data_dir, fname))
        
        dapi = img[4]
        actn2 = img[1]
        gt = img[9]
        
        # Extract GT boxes (ALL cells, no edge filtering)
        gt_boxes = extract_gt_boxes(gt)
        print(f"  GT: {len(gt_boxes)} cells")
        
        # DAPI+Actn2 检测 (正确方法: 在肌节区域选择核)
        # 使用新的双核合并阈值 373px (Dev Set P95)
        from detection.dapi import detect_nuclei, merge_close_nuclei, filter_by_actn2, create_bounding_boxes
        
        regions = detect_nuclei(dapi, min_area=500, max_area=30000)
        # Revert to 1.5x diameter merge (more reliable than fixed 373px)
        all_groups = merge_close_nuclei(regions, use_relative_distance=True)
        filtered_groups = filter_by_actn2(all_groups, actn2, coverage_threshold=0.3)
        
        # 使用新的边缘距离过滤 (100px, Dev Set Refined: exclude 7.3% valid GT)
        edge_filtered_groups = filter_nuclei_by_edge_distance(filtered_groups, dapi.shape, min_edge_distance=100)
        dapi_boxes = create_bounding_boxes(edge_filtered_groups, dapi.shape, exclude_edges=False)
        
        # Adaptive 检测
        adaptive_boxes_raw, _, adaptive_debug = detect_with_adaptive_box(
            dapi, actn2,
            exclude_edges=False
        )
        # 对 Adaptive 框也应用边缘过滤
        adaptive_boxes, _ = filter_boxes_by_edge_overlap(adaptive_boxes_raw, dapi.shape, max_outside_ratio=0.5)
        
        # Evaluate
        dapi_results = evaluate_boxes(dapi_boxes, gt_boxes, gt)
        adaptive_results = evaluate_boxes(adaptive_boxes, gt_boxes, gt)
        
        print(f"  DAPI: {dapi_results['num_matched']}/{dapi_results['num_pred']} matched, "
              f"FP={dapi_results['num_fp']}, FN={dapi_results['num_fn']}")
        print(f"  Adaptive: {adaptive_results['num_matched']}/{adaptive_results['num_pred']} matched, "
              f"FP={adaptive_results['num_fp']}, FN={adaptive_results['num_fn']}")
        
        # Aggregate
        all_dapi['iou'].extend(dapi_results['box_iou'])
        all_dapi['coverage'].extend(dapi_results['cell_coverage'])
        all_dapi['size_ratio'].extend(dapi_results['size_ratio'])
        all_dapi['efficiency'].extend(dapi_results['box_efficiency'])
        all_dapi['fp'] += dapi_results['num_fp']
        all_dapi['fn'] += dapi_results['num_fn']
        all_dapi['tp'] += dapi_results['num_matched']
        
        all_adaptive['iou'].extend(adaptive_results['box_iou'])
        all_adaptive['coverage'].extend(adaptive_results['cell_coverage'])
        all_adaptive['size_ratio'].extend(adaptive_results['size_ratio'])
        all_adaptive['efficiency'].extend(adaptive_results['box_efficiency'])
        all_adaptive['fp'] += adaptive_results['num_fp']
        all_adaptive['fn'] += adaptive_results['num_fn']
        all_adaptive['tp'] += adaptive_results['num_matched']
        
        # Visualize first sample if requested
        if args.visualize and fname == files[0]:
            visualize_matching(img, dapi_boxes, adaptive_boxes, gt_boxes,
                             dapi_results, adaptive_results, fname)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    def compute_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    dapi_p, dapi_r, dapi_f1 = compute_f1(all_dapi['tp'], all_dapi['fp'], all_dapi['fn'])
    adapt_p, adapt_r, adapt_f1 = compute_f1(all_adaptive['tp'], all_adaptive['fp'], all_adaptive['fn'])
    
    print("\nDAPI Detection:")
    print(f"  TP={all_dapi['tp']}, FP={all_dapi['fp']}, FN={all_dapi['fn']}")
    print(f"  Precision: {dapi_p:.3f}, Recall: {dapi_r:.3f}, F1: {dapi_f1:.3f}")
    print(f"  Mean Box IoU:      {np.mean(all_dapi['iou']):.3f}")
    print(f"  Mean Coverage:     {np.mean(all_dapi['coverage']):.3f}")
    print(f"  Mean Size Ratio:   {np.mean(all_dapi['size_ratio']):.3f}")
    print(f"  Mean Efficiency:   {np.mean(all_dapi['efficiency']):.3f}")
    
    print("\nAdaptive Detection:")
    print(f"  TP={all_adaptive['tp']}, FP={all_adaptive['fp']}, FN={all_adaptive['fn']}")
    print(f"  Precision: {adapt_p:.3f}, Recall: {adapt_r:.3f}, F1: {adapt_f1:.3f}")
    print(f"  Mean Box IoU:      {np.mean(all_adaptive['iou']):.3f}")
    print(f"  Mean Coverage:     {np.mean(all_adaptive['coverage']):.3f}")
    print(f"  Mean Size Ratio:   {np.mean(all_adaptive['size_ratio']):.3f}")
    print(f"  Mean Efficiency:   {np.mean(all_adaptive['efficiency']):.3f}")


if __name__ == "__main__":
    main()

