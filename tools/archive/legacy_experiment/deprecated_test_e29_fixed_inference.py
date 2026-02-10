# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: One-off experiment/visualization script (Phase B cleanup)
# Replacement entry points:
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Fixed E29 Inference Script - Direct Model Call
===============================================
问题原因:
- 之前使用 segment_cellular_image() 虽然传入了加载后的 model
- 但 E29 训练效果很小 (decoder 权重变化仅 0.0001 级别)
- 导致 baseline ≈ E29 结果

推理流程 (修复后):
1. 加载预训练 CellSAM
2. 加载 E29 checkpoint 覆盖 mask_decoder 权重
3. 直接调用 model.predict() 进行推理
4. 计算 Instance Dice 和 PQ

解决方案:
- 使用 model.predict() 直接推理，而非 segment_cellular_image()
- 验证权重确实被加载 (打印权重变化)
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cellSAM import get_model
from src.augmented_dataset import AugmentedAllenDataset, load_split_ids
from skimage.measure import regionprops
from tqdm import tqdm
import torch.nn.functional as F


def verify_weight_loading(pretrained_model, finetuned_model):
    """验证权重是否正确加载"""
    decoder_keys = [k for k in pretrained_model.state_dict().keys() if 'mask_decoder' in k]
    
    total_diff = 0
    for k in decoder_keys:
        diff = (pretrained_model.state_dict()[k] - finetuned_model.state_dict()[k]).abs().mean().item()
        total_diff += diff
    
    avg_diff = total_diff / len(decoder_keys)
    print(f"   Decoder 平均权重差异: {avg_diff:.6f}")
    
    if avg_diff < 1e-6:
        print("   ⚠️ 警告: 权重几乎没有变化！")
    else:
        print(f"   ✅ 权重已更新 (差异 > 0)")
    
    return avg_diff


def run_direct_inference(checkpoint_path, num_samples=71, device='cuda'):
    """
    使用 model.predict() 直接推理，避免任何中间 API 干扰
    """
    
    print("="*60)
    print("E29 Fixed Inference (Direct Model Call)")
    print("="*60)
    
    # 1. 加载预训练模型 (用于对比)
    print("\n1. 加载模型...")
    pretrained = get_model()
    
    # 2. 加载微调模型
    finetuned = get_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        finetuned.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✅ 加载 epoch {checkpoint.get('epoch', '?')}, best_dice={checkpoint.get('best_dice', 0):.4f}")
    else:
        finetuned.load_state_dict(checkpoint, strict=False)
    
    # 3. 验证权重差异
    print("\n2. 验证权重加载...")
    avg_diff = verify_weight_loading(pretrained, finetuned)
    
    finetuned = finetuned.to(device)
    finetuned.eval()
    
    # 4. 加载数据
    print("\n3. 加载验证数据...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=True,
        is_training=False
    )
    print(f"   ✅ 加载 {len(dataset)} 样本")
    
    # 5. 推理
    instance_dices = []
    pq_scores = []
    
    print(f"\n4. 运行推理 ({min(num_samples, len(dataset))} 样本)...")
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[idx]
            
            image = sample['image'].unsqueeze(0).to(device)  # (1, 3, H, W)
            gt_mask = sample['mask'].numpy()
            boxes = sample['boxes']
            
            # 准备 bounding boxes
            boxes_array = boxes.numpy() if torch.is_tensor(boxes) else boxes
            valid_boxes = [b for b in boxes_array if sum(b) > 0]
            
            if len(valid_boxes) == 0:
                continue
            
            boxes_tensor = torch.tensor(valid_boxes).unsqueeze(0).to(device)
            
            # 直接调用 model.predict()
            try:
                preds = finetuned.predict(image, boxes_per_heatmap=boxes_tensor)
                if preds is None:
                    continue
                
                pred_masks, _, _, _ = preds
                pred_mask = pred_masks.squeeze().cpu().numpy()
                
                # 如果是多个 mask，合并
                if pred_mask.ndim == 3:
                    combined = np.zeros_like(pred_mask[0], dtype=np.int32)
                    for i, m in enumerate(pred_mask):
                        combined[m > 0.5] = i + 1
                    pred_mask = combined
                else:
                    pred_mask = (pred_mask > 0.5).astype(np.int32)
                    
            except Exception as e:
                continue
            
            # 计算 Instance Dice
            gt_regions = regionprops(gt_mask)
            pred_regions = regionprops(pred_mask.astype(np.int32))
            
            for gt_region in gt_regions:
                gt_cell = (gt_mask == gt_region.label)
                best_dice = 0
                for pred_region in pred_regions:
                    pred_cell = (pred_mask == pred_region.label)
                    intersection = np.sum(gt_cell & pred_cell)
                    union = np.sum(gt_cell) + np.sum(pred_cell)
                    if union > 0:
                        dice = 2 * intersection / union
                        best_dice = max(best_dice, dice)
                instance_dices.append(best_dice)
            
            # 计算 PQ
            iou_threshold = 0.5
            tp, matched_ious = 0, []
            pred_matched = set()
            
            for gt_region in gt_regions:
                gt_cell = (gt_mask == gt_region.label)
                best_iou, best_pred = 0, None
                
                for pred_region in pred_regions:
                    if pred_region.label in pred_matched:
                        continue
                    pred_cell = (pred_mask == pred_region.label)
                    intersection = np.sum(gt_cell & pred_cell)
                    union = np.sum(gt_cell | pred_cell)
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_pred = iou, pred_region.label
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_ious.append(best_iou)
                    pred_matched.add(best_pred)
            
            fn = len(gt_regions) - tp
            fp = len(pred_regions) - len(pred_matched)
            sq = np.mean(matched_ious) if matched_ious else 0
            rq = tp / (tp + 0.5*fp + 0.5*fn) if (tp + fp + fn) > 0 else 0
            pq_scores.append(sq * rq)
    
    # 结果
    print("\n" + "="*60)
    print("E29 Fixed Inference 结果:")
    print(f"  Instance Dice: {np.mean(instance_dices):.4f} ± {np.std(instance_dices):.4f}")
    print(f"  PQ@0.5: {np.mean(pq_scores):.4f} ± {np.std(pq_scores):.4f}")
    print(f"  Decoder 权重差异: {avg_diff:.6f}")
    print("="*60)
    
    return {
        'instance_dice_mean': np.mean(instance_dices),
        'instance_dice_std': np.std(instance_dices),
        'pq_mean': np.mean(pq_scores),
        'pq_std': np.std(pq_scores),
        'weight_diff': avg_diff
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/E29_bf_instance_best.pt')
    parser.add_argument('--samples', type=int, default=71)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_direct_inference(args.checkpoint, args.samples, args.device)
