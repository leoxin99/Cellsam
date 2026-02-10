"""
统一指标模块 (Unified Metrics Module)

所有评估脚本应共用这里的指标计算函数，确保训练验证与推理评估口径一致。

指标命名规范:
  - BM-1to1:      Best-Match Dice (Hungarian 一对一匹配) — 主指标
  - BM-Coverage:  Best-Match Dice (每 GT 独立取最大)   — 辅助诊断
  - PQ@0.5:       Panoptic Quality                     — 主指标
  - AJI:          Aggregated Jaccard Index              — 辅助
  - Semantic Dice: 二值前景 Dice                        — 辅助
  - Gap:          Coverage - 1to1                       — 诊断粘连

Author: Claude (Antigravity) - Phase 0 Implementation
Date: 2026-02-09, Updated 2026-02-10
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, List, Union


# ==============================================================
# IoU 矩阵辅助函数 (避免重复计算)
# ==============================================================

def _build_iou_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    gt_ids: np.ndarray,
    pred_ids: np.ndarray
) -> np.ndarray:
    """构建 GT-Pred IoU 矩阵，供多个指标共用。"""
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        gt_cell = (gt_mask == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_cell = (pred_mask == pred_id)
            intersection = (gt_cell & pred_cell).sum()
            union = (gt_cell | pred_cell).sum()
            iou_matrix[i, j] = intersection / (union + 1e-8)
    return iou_matrix


def _build_dice_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    gt_ids: np.ndarray,
    pred_ids: np.ndarray
) -> np.ndarray:
    """构建 GT-Pred Dice 矩阵。"""
    dice_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        gt_cell = (gt_mask == gt_id)
        gt_area = gt_cell.sum()
        for j, pred_id in enumerate(pred_ids):
            pred_cell = (pred_mask == pred_id)
            intersection = (gt_cell & pred_cell).sum()
            dice_matrix[i, j] = 2 * intersection / (gt_area + pred_cell.sum() + 1e-8)
    return dice_matrix


# ==============================================================
# Best-Match Dice: 双定义
# ==============================================================

def compute_bm_1to1_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    return_per_cell: bool = False
) -> Union[float, Tuple[float, List[float]]]:
    """
    BM-1to1 Dice — 主指标
    
    使用 Hungarian 算法做一对一最优匹配，
    每个 Pred 最多匹配一个 GT，每个 GT 最多匹配一个 Pred。
    
    Args:
        pred_mask: [H, W] 预测实例 mask
        gt_mask: [H, W] GT 实例 mask
        return_per_cell: 是否返回每个 GT 的 Dice
    
    Returns:
        mean_dice (float) 或 (mean_dice, per_cell_list)
    """
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0:
        return (0.0, []) if return_per_cell else 0.0
    if len(pred_ids) == 0:
        return (0.0, [0.0] * len(gt_ids)) if return_per_cell else 0.0

    dice_matrix = _build_dice_matrix(pred_mask, gt_mask, gt_ids, pred_ids)

    # Hungarian 匹配 (最大化 -> 取负最小化)
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    per_cell = np.zeros(len(gt_ids))
    for r, c in zip(row_ind, col_ind):
        per_cell[r] = dice_matrix[r, c]

    mean_dice = float(np.mean(per_cell))
    if return_per_cell:
        return mean_dice, per_cell.tolist()
    return mean_dice


def compute_bm_coverage_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    return_per_cell: bool = False
) -> Union[float, Tuple[float, List[float]]]:
    """
    BM-Coverage Dice — 辅助诊断指标
    
    每个 GT 独立地选取 Dice 最高的 Pred (允许多个 GT 共享同一 Pred)。
    用途: 观察"模型有没有看到每个细胞"，与 BM-1to1 的差值反映粘连程度。
    
    Args:
        pred_mask: [H, W] 预测实例 mask
        gt_mask: [H, W] GT 实例 mask
        return_per_cell: 是否返回每个 GT 的 Dice
    
    Returns:
        mean_dice (float) 或 (mean_dice, per_cell_list)
    """
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0:
        return (0.0, []) if return_per_cell else 0.0
    if len(pred_ids) == 0:
        return (0.0, [0.0] * len(gt_ids)) if return_per_cell else 0.0

    # 如果 pred 是 binary (max<=1), 先做连通组件
    if pred_mask.max() <= 1:
        pred_binary = (pred_mask > 0.5).astype(np.int32)
        pred_labeled, _ = ndimage.label(pred_binary)
        pred_ids = np.unique(pred_labeled)
        pred_ids = pred_ids[pred_ids > 0]
        if len(pred_ids) == 0:
            return (0.0, [0.0] * len(gt_ids)) if return_per_cell else 0.0
    else:
        pred_labeled = pred_mask

    per_cell = []
    for gt_id in gt_ids:
        gt_cell = (gt_mask == gt_id)
        gt_area = gt_cell.sum()
        best_dice = 0.0
        for pred_id in pred_ids:
            pred_cell = (pred_labeled == pred_id)
            intersection = (gt_cell & pred_cell).sum()
            dice = 2.0 * intersection / (gt_area + pred_cell.sum() + 1e-8)
            best_dice = max(best_dice, dice)
        per_cell.append(best_dice)

    mean_dice = float(np.mean(per_cell)) if per_cell else 0.0
    if return_per_cell:
        return mean_dice, per_cell
    return mean_dice


# ==============================================================
# Panoptic Quality
# ==============================================================

def compute_pq(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Panoptic Quality (PQ = SQ × RQ)

    SQ = 匹配对的平均 IoU
    RQ = TP / (TP + 0.5*FP + 0.5*FN)
    """
    _, pq, sq, rq = compute_pq_detailed(pred_mask, gt_mask, iou_threshold)
    return pq, sq, rq


def compute_pq_detailed(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[dict, float, float, float]:
    """
    Panoptic Quality with TP/FP/FN details.
    
    Returns:
        (details_dict, pq, sq, rq) where details_dict = {'tp': int, 'fp': int, 'fn': int}
    """
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    n_gt, n_pred = len(gt_ids), len(pred_ids)

    if n_gt == 0 and n_pred == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0}, 1.0, 1.0, 1.0
    if n_gt == 0 or n_pred == 0:
        return {'tp': 0, 'fp': n_pred, 'fn': n_gt}, 0.0, 0.0, 0.0

    iou_matrix = _build_iou_matrix(pred_mask, gt_mask, gt_ids, pred_ids)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    tp, matched_ious = 0, []
    matched_gt, matched_pred = set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1
            matched_ious.append(iou_matrix[r, c])
            matched_gt.add(r)
            matched_pred.add(c)

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn}, sq * rq, sq, rq


# ==============================================================
# AJI
# ==============================================================

def compute_aji(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """Aggregated Jaccard Index"""
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    used_pred = set()
    total_intersection, total_union = 0, 0

    for gt_id in gt_ids:
        gt_cell = (gt_mask == gt_id)
        best_iou, best_pred_id, best_pred_cell = 0, None, None
        for pred_id in pred_ids:
            if pred_id in used_pred:
                continue
            pred_cell = (pred_mask == pred_id)
            inter = (gt_cell & pred_cell).sum()
            union = (gt_cell | pred_cell).sum()
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou, best_pred_id, best_pred_cell = iou, pred_id, pred_cell
        if best_pred_id is not None:
            used_pred.add(best_pred_id)
            total_intersection += (gt_cell & best_pred_cell).sum()
            total_union += (gt_cell | best_pred_cell).sum()
        else:
            total_union += gt_cell.sum()

    for pred_id in pred_ids:
        if pred_id not in used_pred:
            total_union += (pred_mask == pred_id).sum()

    return float(total_intersection / (total_union + 1e-8))


# ==============================================================
# Semantic Dice
# ==============================================================

def compute_semantic_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """Semantic Dice (忽略实例 ID, 仅前景/背景)"""
    pred_bin = (pred_mask > 0).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)
    inter = (pred_bin * gt_bin).sum()
    return float(2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8))


# ==============================================================
# 汇总函数
# ==============================================================

def compute_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算所有标准指标 (单张图像)
    
    Returns:
        dict 包含:
          bm_1to1_dice, bm_coverage_dice, gap_dice,
          pq, sq, rq, aji, semantic_dice,
          n_gt_cells, n_pred_cells
    """
    bm_1to1 = compute_bm_1to1_dice(pred_mask, gt_mask)
    bm_coverage = compute_bm_coverage_dice(pred_mask, gt_mask)
    pq_details, pq, sq, rq = compute_pq_detailed(pred_mask, gt_mask, iou_threshold)
    aji = compute_aji(pred_mask, gt_mask)
    sem_dice = compute_semantic_dice(pred_mask, gt_mask)

    return {
        'bm_1to1_dice': bm_1to1,
        'bm_coverage_dice': bm_coverage,
        'gap_dice': bm_coverage - bm_1to1,
        'pq': pq,
        'sq': sq,
        'rq': rq,
        'tp': pq_details['tp'],
        'fp': pq_details['fp'],
        'fn': pq_details['fn'],
        'aji': aji,
        'semantic_dice': sem_dice,
        'n_gt_cells': int((np.unique(gt_mask) > 0).sum()),
        'n_pred_cells': int((np.unique(pred_mask) > 0).sum()),
    }
