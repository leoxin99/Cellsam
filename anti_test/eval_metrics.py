"""
Comprehensive Evaluation Metrics for Cell Segmentation

Author: Bioimage Analysis Evaluation Architect
Date: 2026-01-11

Implements the 4-tier validation framework:
- Tier 1: Detection metrics (AP)
- Tier 2: Panoptic Quality (PQ = SQ × RQ), Aggregated Jaccard Index (AJI), Rand Index
- Tier 3: Boundary IoU, Hausdorff Distance (HD95)
- Tier 4: Biological metrics (SarcGraph integration - future)

Usage:
    from eval_metrics import evaluate_instance_segmentation
    results = evaluate_instance_segmentation(pred_mask, gt_mask)
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# TIER 1: INSTANCE MATCHING
# ============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def match_instances(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                    iou_threshold: float = 0.5) -> Dict:
    """
    Match predicted instances to ground truth using Hungarian algorithm.
    
    Returns:
        matched: List of (pred_id, gt_id, iou) tuples
        unmatched_pred: List of pred_ids (false positives)
        unmatched_gt: List of gt_ids (false negatives)
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return {
            'matched': [],
            'unmatched_pred': list(pred_ids),
            'unmatched_gt': list(gt_ids),
            'iou_matrix': np.array([])
        }
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pred_id in enumerate(pred_ids):
        pred_binary = (pred_mask == pred_id)
        for j, gt_id in enumerate(gt_ids):
            gt_binary = (gt_mask == gt_id)
            iou_matrix[i, j] = compute_iou(pred_binary, gt_binary)
    
    # Hungarian matching (maximize IoU)
    cost_matrix = 1 - iou_matrix  # Convert to cost (minimize)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched = []
    matched_pred = set()
    matched_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            matched.append((pred_ids[i], gt_ids[j], iou))
            matched_pred.add(pred_ids[i])
            matched_gt.add(gt_ids[j])
    
    unmatched_pred = [p for p in pred_ids if p not in matched_pred]
    unmatched_gt = [g for g in gt_ids if g not in matched_gt]
    
    return {
        'matched': matched,
        'unmatched_pred': unmatched_pred,
        'unmatched_gt': unmatched_gt,
        'iou_matrix': iou_matrix
    }


# ============================================================================
# TIER 2: PANOPTIC QUALITY (PQ)
# ============================================================================

def compute_panoptic_quality(pred_mask: np.ndarray, gt_mask: np.ndarray,
                              iou_threshold: float = 0.5) -> Dict:
    """
    Compute Panoptic Quality = SQ × RQ
    
    SQ (Segmentation Quality): Average IoU of matched pairs
    RQ (Recognition Quality): F1 score of detection
    
    Reference: Kirillov et al., "Panoptic Segmentation"
    """
    matching = match_instances(pred_mask, gt_mask, iou_threshold)
    
    tp = len(matching['matched'])
    fp = len(matching['unmatched_pred'])
    fn = len(matching['unmatched_gt'])
    
    # Segmentation Quality: average IoU of true positives
    if tp > 0:
        sq = np.mean([m[2] for m in matching['matched']])
    else:
        sq = 0.0
    
    # Recognition Quality: F1 score
    if tp + fp + fn > 0:
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    else:
        rq = 0.0
    
    # Panoptic Quality
    pq = sq * rq
    
    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': tp,
        'FP': fp,
        'FN': fn
    }


# ============================================================================
# TIER 2: AGGREGATED JACCARD INDEX (AJI)
# ============================================================================

def compute_aji(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).
    
    AJI = Σ|G_i ∩ P_M(i)| / (Σ|G_i ∪ P_M(i)| + Σ|P_k|)
    
    Where:
    - G_i is ground truth object i
    - P_M(i) is best matching prediction for G_i
    - P_k are unmatched predictions (false positives)
    
    Reference: Kumar et al., "A Dataset and a Technique for Generalized 
               Nuclear Segmentation for Computational Pathology"
    """
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]
    
    if len(gt_ids) == 0:
        return 0.0 if len(pred_ids) > 0 else 1.0
    
    total_intersection = 0
    total_union = 0
    matched_preds = set()
    
    # For each GT object, find best matching prediction
    for gt_id in gt_ids:
        gt_binary = (gt_mask == gt_id)
        gt_area = gt_binary.sum()
        
        best_iou = 0
        best_pred_id = None
        best_intersection = 0
        best_union = gt_area
        
        for pred_id in pred_ids:
            if pred_id in matched_preds:
                continue
            
            pred_binary = (pred_mask == pred_id)
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id
                    best_intersection = intersection
                    best_union = union
        
        total_intersection += best_intersection
        total_union += best_union
        
        if best_pred_id is not None:
            matched_preds.add(best_pred_id)
    
    # Add unmatched predictions to denominator
    for pred_id in pred_ids:
        if pred_id not in matched_preds:
            pred_binary = (pred_mask == pred_id)
            total_union += pred_binary.sum()
    
    aji = total_intersection / total_union if total_union > 0 else 0.0
    
    return aji


# ============================================================================
# TIER 2: RAND INDEX
# ============================================================================

def compute_rand_index(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
    """
    Compute Rand Index and Adjusted Rand Index.
    
    RI = (TP + TN) / (TP + TN + FP + FN)
    
    Where pairs are:
    - TP: Same cluster in both pred and gt
    - TN: Different clusters in both pred and gt
    - FP: Same cluster in pred, different in gt
    - FN: Different clusters in pred, same in gt
    
    Reference: Rand, "Objective Criteria for the Evaluation of Clustering Methods"
    """
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Only consider foreground pixels
    fg_mask = (gt_flat > 0) | (pred_flat > 0)
    pred_fg = pred_flat[fg_mask]
    gt_fg = gt_flat[fg_mask]
    
    n = len(pred_fg)
    if n < 2:
        return {'RI': 1.0, 'ARI': 1.0}
    
    # For efficiency, use contingency table approach
    # Build contingency matrix
    pred_ids = np.unique(pred_fg)
    gt_ids = np.unique(gt_fg)
    
    contingency = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    pred_id_map = {v: i for i, v in enumerate(pred_ids)}
    gt_id_map = {v: i for i, v in enumerate(gt_ids)}
    
    for p, g in zip(pred_fg, gt_fg):
        contingency[pred_id_map[p], gt_id_map[g]] += 1
    
    # Compute sums
    sum_comb_c = sum(n_ij * (n_ij - 1) / 2 for n_ij in contingency.flatten())
    sum_comb_a = sum(n_i * (n_i - 1) / 2 for n_i in contingency.sum(axis=1))
    sum_comb_b = sum(n_j * (n_j - 1) / 2 for n_j in contingency.sum(axis=0))
    
    total_pairs = n * (n - 1) / 2
    
    # Rand Index
    tp_fn = sum_comb_a
    tp_fp = sum_comb_b
    tp = sum_comb_c
    
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = total_pairs - tp - fp - fn
    
    ri = (tp + tn) / total_pairs if total_pairs > 0 else 1.0
    
    # Adjusted Rand Index
    expected = (sum_comb_a * sum_comb_b) / total_pairs if total_pairs > 0 else 0
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    if max_index - expected != 0:
        ari = (sum_comb_c - expected) / (max_index - expected)
    else:
        ari = 1.0
    
    return {'RI': ri, 'ARI': ari}


# ============================================================================
# TIER 3: BOUNDARY IOU
# ============================================================================

def get_boundary_mask(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    """Extract boundary pixels from a mask."""
    eroded = morphology.binary_erosion(mask, morphology.disk(dilation))
    boundary = mask & ~eroded
    return boundary


def compute_boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray,
                          dilation: int = 2) -> float:
    """
    Compute Boundary IoU.
    
    Only evaluates pixels near the boundary, not the entire mask.
    
    Reference: Cheng et al., "Boundary IoU: Improving Object-Centric 
               Image Segmentation Evaluation"
    """
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    pred_boundary = get_boundary_mask(pred_binary, dilation)
    gt_boundary = get_boundary_mask(gt_binary, dilation)
    
    # Dilate boundaries for evaluation region
    pred_region = morphology.binary_dilation(pred_boundary, morphology.disk(dilation))
    gt_region = morphology.binary_dilation(gt_boundary, morphology.disk(dilation))
    
    eval_region = pred_region | gt_region
    
    intersection = np.logical_and(pred_binary & eval_region, gt_binary & eval_region).sum()
    union = np.logical_or(pred_binary & eval_region, gt_binary & eval_region).sum()
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# TIER 3: HAUSDORFF DISTANCE (HD95)
# ============================================================================

def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff Distance.
    
    HD95 measures the maximum boundary deviation at 95th percentile
    (to exclude outliers).
    """
    from scipy.ndimage import distance_transform_edt
    
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    if not pred_binary.any() or not gt_binary.any():
        return float('inf')
    
    # Get boundaries
    pred_boundary = get_boundary_mask(pred_binary, 1)
    gt_boundary = get_boundary_mask(gt_binary, 1)
    
    if not pred_boundary.any() or not gt_boundary.any():
        return float('inf')
    
    # Distance from pred boundary to gt
    gt_dist = distance_transform_edt(~gt_boundary)
    pred_to_gt = gt_dist[pred_boundary]
    
    # Distance from gt boundary to pred
    pred_dist = distance_transform_edt(~pred_boundary)
    gt_to_pred = pred_dist[gt_boundary]
    
    # 95th percentile of max distances
    all_distances = np.concatenate([pred_to_gt, gt_to_pred])
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_instance_segmentation(pred_mask: np.ndarray, 
                                    gt_mask: np.ndarray,
                                    iou_thresholds: List[float] = [0.3, 0.5]) -> Dict:
    """
    Comprehensive evaluation of instance segmentation.
    
    Returns all metrics from the 4-tier validation framework.
    
    Args:
        pred_mask: Predicted instance mask (each cell has unique ID)
        gt_mask: Ground truth instance mask
        iou_thresholds: List of IoU thresholds for matching
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Basic stats
    n_pred = len(np.unique(pred_mask)) - (1 if 0 in pred_mask else 0)
    n_gt = len(np.unique(gt_mask)) - (1 if 0 in gt_mask else 0)
    results['n_pred'] = n_pred
    results['n_gt'] = n_gt
    
    # Tier 2: Panoptic Quality at multiple thresholds
    for thresh in iou_thresholds:
        pq_results = compute_panoptic_quality(pred_mask, gt_mask, thresh)
        results[f'PQ@{thresh}'] = pq_results['PQ']
        results[f'SQ@{thresh}'] = pq_results['SQ']
        results[f'RQ@{thresh}'] = pq_results['RQ']
        results[f'TP@{thresh}'] = pq_results['TP']
        results[f'FP@{thresh}'] = pq_results['FP']
        results[f'FN@{thresh}'] = pq_results['FN']
    
    # Use 0.5 as primary
    results['PQ'] = results.get('PQ@0.5', 0)
    results['SQ'] = results.get('SQ@0.5', 0)
    results['RQ'] = results.get('RQ@0.5', 0)
    results['TP'] = results.get('TP@0.5', 0)
    results['FP'] = results.get('FP@0.5', 0)
    results['FN'] = results.get('FN@0.5', 0)
    
    # Tier 2: AJI
    results['AJI'] = compute_aji(pred_mask, gt_mask)
    
    # Tier 2: Rand Index
    rand_results = compute_rand_index(pred_mask, gt_mask)
    results.update(rand_results)
    
    # Tier 3: Boundary IoU
    results['Boundary_IoU'] = compute_boundary_iou(pred_mask, gt_mask)
    
    # Tier 3: HD95
    results['HD95'] = compute_hd95(pred_mask, gt_mask)
    
    # Overall Dice (for reference)
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)
    results['Dice'] = dice
    
    # Max IoU analysis (for debugging)
    matching = match_instances(pred_mask, gt_mask, iou_threshold=0.0)
    if matching['iou_matrix'].size > 0:
        results['Max_IoU'] = matching['iou_matrix'].max()
        results['Mean_Max_IoU'] = np.mean(matching['iou_matrix'].max(axis=1)) if matching['iou_matrix'].shape[0] > 0 else 0
    else:
        results['Max_IoU'] = 0
        results['Mean_Max_IoU'] = 0
    
    return results


def print_evaluation_report(results: Dict):
    """Print formatted evaluation report."""
    print("="*60)
    print("INSTANCE SEGMENTATION EVALUATION REPORT")
    print("="*60)
    print(f"\nCells: GT={results['n_gt']}, Pred={results['n_pred']}")
    print()
    print("TIER 2: Panoptic Quality")
    print(f"  @IoU=0.3: PQ={results.get('PQ@0.3', 0):.4f}, TP={results.get('TP@0.3', 0)}, FP={results.get('FP@0.3', 0)}, FN={results.get('FN@0.3', 0)}")
    print(f"  @IoU=0.5: PQ={results.get('PQ@0.5', 0):.4f}, TP={results.get('TP@0.5', 0)}, FP={results.get('FP@0.5', 0)}, FN={results.get('FN@0.5', 0)}")
    print(f"  AJI = {results['AJI']:.4f}")
    print(f"  RI  = {results['RI']:.4f}, ARI = {results['ARI']:.4f}")
    print()
    print("TIER 3: Boundary Precision")
    print(f"  Boundary IoU = {results['Boundary_IoU']:.4f}")
    print(f"  HD95         = {results['HD95']:.2f} px")
    print()
    print("Instance Matching:")
    print(f"  Max IoU      = {results.get('Max_IoU', 0):.4f}")
    print(f"  Mean Max IoU = {results.get('Mean_Max_IoU', 0):.4f}")
    print()
    print("Reference:")
    print(f"  Dice = {results['Dice']:.4f}")
    print("="*60)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import tifffile
    from pathlib import Path
    
    print("Testing Evaluation Metrics Module")
    print("="*60)
    
    # Load a test sample
    DATA_DIR = Path("d:/AI/paper/CellSam/experiments/exp_20260109_204227/images")
    
    if DATA_DIR.exists():
        gt_files = sorted(DATA_DIR.glob("*_gt_mask.npy"))
        pred_files = sorted(DATA_DIR.glob("*_pred_mask.npy"))
        
        if gt_files and pred_files:
            all_results = []
            
            for gt_path, pred_path in zip(gt_files[:5], pred_files[:5]):
                print(f"\nEvaluating: {gt_path.stem}")
                
                gt_mask = np.load(gt_path)
                pred_mask = np.load(pred_path)
                
                results = evaluate_instance_segmentation(pred_mask, gt_mask)
                print_evaluation_report(results)
                all_results.append(results)
            
            # Summary
            print("\n" + "="*60)
            print("SUMMARY (Mean across samples)")
            print("="*60)
            for key in ['PQ', 'SQ', 'RQ', 'AJI', 'RI', 'Boundary_IoU', 'Dice']:
                mean_val = np.mean([r[key] for r in all_results])
                print(f"  {key}: {mean_val:.4f}")
        else:
            print("No mask files found. Run visualize_test_results.py first.")
    else:
        print(f"Data directory not found: {DATA_DIR}")
        print("Creating synthetic test...")
        
        # Synthetic test
        gt_mask = np.zeros((100, 100), dtype=np.int32)
        gt_mask[20:40, 20:40] = 1
        gt_mask[60:80, 60:80] = 2
        
        pred_mask = np.zeros((100, 100), dtype=np.int32)
        pred_mask[22:42, 22:42] = 1  # Slightly shifted
        pred_mask[58:78, 58:78] = 2  # Slightly shifted
        pred_mask[10:15, 10:15] = 3  # False positive
        
        results = evaluate_instance_segmentation(pred_mask, gt_mask)
        print_evaluation_report(results)
