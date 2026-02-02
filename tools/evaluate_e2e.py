"""
端到端评估：DAPI 检测 → SAM 分割

与 comprehensive_eval.py 的区别：
- comprehensive_eval: 使用 GT 框 (来自 mask) → 评估分割模型
- 本脚本: 使用 DAPI 检测框 → 评估完整推理管线

评估流程：
1. 加载图像
2. DAPI 核检测 → 生成框
3. SAM 分割每个框
4. 与 GT mask 对比计算指标
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from datetime import datetime
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset
from detection.dapi import detect_and_create_boxes


def compute_metrics(pred_mask, gt_mask):
    """Compute instance segmentation metrics."""
    # Get unique labels
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels > 0]
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels > 0]
    
    n_pred = len(pred_labels)
    n_gt = len(gt_labels)
    
    # Dice (pixel-level)
    if gt_mask.sum() > 0:
        intersection = ((pred_mask > 0) & (gt_mask > 0)).sum()
        dice = 2 * intersection / ((pred_mask > 0).sum() + (gt_mask > 0).sum() + 1e-8)
    else:
        dice = 0.0
    
    # Instance matching for PQ
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0
    
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0
        best_pred = -1
        
        for pred_label in pred_labels:
            if pred_label in matched_pred:
                continue
            pred_region = (pred_mask == pred_label)
            intersection = (gt_region & pred_region).sum()
            union = (gt_region | pred_region).sum()
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label
        
        if best_iou >= 0.5 and best_pred >= 0:
            matched_gt.add(gt_label)
            matched_pred.add(best_pred)
            iou_sum += best_iou
    
    # PQ = SQ * RQ
    tp = len(matched_gt)
    fp = n_pred - tp
    fn = n_gt - tp
    
    sq = iou_sum / tp if tp > 0 else 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0
    pq = sq * rq
    
    # Also compute PQ@0.3
    matched_pred_03 = set()
    matched_gt_03 = set()
    iou_sum_03 = 0
    
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0
        best_pred = -1
        
        for pred_label in pred_labels:
            if pred_label in matched_pred_03:
                continue
            pred_region = (pred_mask == pred_label)
            intersection = (gt_region & pred_region).sum()
            union = (gt_region | pred_region).sum()
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label
        
        if best_iou >= 0.3 and best_pred >= 0:
            matched_gt_03.add(gt_label)
            matched_pred_03.add(best_pred)
            iou_sum_03 += best_iou
    
    tp_03 = len(matched_gt_03)
    fp_03 = n_pred - tp_03
    fn_03 = n_gt - tp_03
    
    sq_03 = iou_sum_03 / tp_03 if tp_03 > 0 else 0
    rq_03 = tp_03 / (tp_03 + 0.5 * fp_03 + 0.5 * fn_03) if (tp_03 + fp_03 + fn_03) > 0 else 0
    pq_03 = sq_03 * rq_03
    
    return {
        'dice': dice,
        'pq_05': pq,
        'pq_03': pq_03,
        'n_pred': n_pred,
        'n_gt': n_gt,
        'tp_05': tp,
        'tp_03': tp_03,
    }


def evaluate_e2e():
    """End-to-end evaluation with DAPI detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = get_model()
    checkpoint = torch.load("checkpoints/bf_baseline_full_best.pt", map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Test samples: {len(dataset)}")
    
    all_results = []
    
    for idx in tqdm(range(len(dataset)), desc="E2E Eval"):
        sample = dataset[idx]
        
        image = sample['image'].numpy()  # (3, 1024, 1024)
        gt_mask = sample['mask'].numpy().astype(np.int32)
        
        # Step 1: DAPI detection (use normalized DAPI channel)
        dapi = image[1]  # DAPI channel, normalized [0, 1]
        result = detect_and_create_boxes(dapi)
        boxes = result[0] if isinstance(result, tuple) else result
        
        if not boxes or len(boxes) == 0:
            all_results.append({
                'dice': 0, 'pq_05': 0, 'pq_03': 0,
                'n_pred': 0, 'n_gt': len(np.unique(gt_mask)) - 1,
                'tp_05': 0, 'tp_03': 0,
            })
            continue
        
        # Step 2: Prepare BF image for SAM
        img_bf = np.stack([image[0], image[0], image[0]], axis=0)
        img_tensor = torch.from_numpy(img_bf).float().unsqueeze(0).to(device)
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_preprocessed = model.sam_preprocess(img_tensor)
        
        with torch.no_grad():
            image_embedding = model.model.image_encoder(img_preprocessed)
        
        # Step 3: Segment each box
        pred_mask = np.zeros((1024, 1024), dtype=np.int32)
        
        for i, box in enumerate(boxes):
            if len(box) < 4:
                continue
            
            box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
            
            pred = F.interpolate(
                low_res_masks, size=(1024, 1024),
                mode='bilinear', align_corners=False
            ).squeeze()
            
            mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            
            # Box clipping (expand=0.1)
            x1, y1, x2, y2 = [int(b) for b in box]
            h, w = mask.shape
            bw, bh = x2 - x1, y2 - y1
            expand = 0.1
            x1_clip = max(0, int(x1 - bw * expand))
            y1_clip = max(0, int(y1 - bh * expand))
            x2_clip = min(w, int(x2 + bw * expand))
            y2_clip = min(h, int(y2 + bh * expand))
            
            mask_clipped = np.zeros_like(mask)
            mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]
            
            pred_mask[mask_clipped > 0] = i + 1
        
        # Step 4: Compute metrics
        metrics = compute_metrics(pred_mask, gt_mask)
        all_results.append(metrics)
    
    # Aggregate results
    print("\n" + "="*70)
    print("端到端评估结果 (DAPI检测 → SAM分割)")
    print("="*70)
    
    avg_dice = np.mean([r['dice'] for r in all_results])
    avg_pq_05 = np.mean([r['pq_05'] for r in all_results])
    avg_pq_03 = np.mean([r['pq_03'] for r in all_results])
    avg_n_pred = np.mean([r['n_pred'] for r in all_results])
    avg_n_gt = np.mean([r['n_gt'] for r in all_results])
    avg_tp_05 = np.mean([r['tp_05'] for r in all_results])
    avg_tp_03 = np.mean([r['tp_03'] for r in all_results])
    
    std_dice = np.std([r['dice'] for r in all_results])
    std_pq_05 = np.std([r['pq_05'] for r in all_results])
    std_pq_03 = np.std([r['pq_03'] for r in all_results])
    
    print(f"\n【分割质量】")
    print(f"  Dice:      {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"  PQ@0.5:    {avg_pq_05:.4f} ± {std_pq_05:.4f}")
    print(f"  PQ@0.3:    {avg_pq_03:.4f} ± {std_pq_03:.4f}")
    
    print(f"\n【检测匹配】")
    print(f"  平均预测数:     {avg_n_pred:.1f}")
    print(f"  平均GT数:       {avg_n_gt:.1f}")
    print(f"  平均TP@0.5:     {avg_tp_05:.1f} ({avg_tp_05/avg_n_gt*100:.1f}%召回)")
    print(f"  平均TP@0.3:     {avg_tp_03:.1f} ({avg_tp_03/avg_n_gt*100:.1f}%召回)")
    
    # Compare with GT-box evaluation
    print(f"\n【与GT框评估对比】")
    print(f"  GT框评估:   使用 GT mask 提取的框 (理想情况)")
    print(f"  端到端评估: 使用 DAPI 检测框 (实际推理)")
    
    # Save results
    output_dir = Path("experiments/e2e_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': 'bf_baseline_full_best.pt',
                'detection': 'DAPI detect_and_create_boxes',
                'box_expand': 0.1,
            },
            'metrics': {
                'dice': {'mean': avg_dice, 'std': std_dice},
                'pq_05': {'mean': avg_pq_05, 'std': std_pq_05},
                'pq_03': {'mean': avg_pq_03, 'std': std_pq_03},
            },
            'detection': {
                'avg_n_pred': avg_n_pred,
                'avg_n_gt': avg_n_gt,
                'avg_tp_05': avg_tp_05,
                'avg_tp_03': avg_tp_03,
            },
            'per_sample': all_results,
        }, f, indent=2)
    
    print(f"\n结果保存至: {output_dir / 'results.json'}")
    
    return all_results


if __name__ == "__main__":
    evaluate_e2e()
