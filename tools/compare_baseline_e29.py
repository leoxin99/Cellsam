"""
对照测试: Baseline vs E29 训练效果
=================================
目的: 用完全相同的代码测试 baseline 和 E29，确认训练是否有效
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cellSAM import get_model
from cellSAM.model import segment_cellular_image
from src.augmented_dataset import AugmentedAllenDataset, load_split_ids
from skimage.measure import regionprops
from tqdm import tqdm


def compute_metrics(pred_mask, gt_mask):
    """计算 Instance Dice 和 PQ"""
    gt_regions = regionprops(gt_mask)
    pred_regions = regionprops(pred_mask.astype(np.int32))
    
    # Instance Dice
    dices = []
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
        dices.append(best_dice)
    
    # PQ
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
        if best_iou >= 0.5:
            tp += 1
            matched_ious.append(best_iou)
            pred_matched.add(best_pred)
    
    fn = len(gt_regions) - tp
    fp = len(pred_regions) - len(pred_matched)
    sq = np.mean(matched_ious) if matched_ious else 0
    rq = tp / (tp + 0.5*fp + 0.5*fn) if (tp + fp + fn) > 0 else 0
    
    return np.mean(dices) if dices else 0, sq * rq


def run_comparison(num_samples=20, device='cuda'):
    """对照测试 baseline vs E29"""
    
    print("="*70)
    print("对照测试: Baseline vs E29")
    print("="*70)
    
    # 加载数据 (只用一次，保证两个模型测试相同样本)
    print("\n1. 加载验证数据...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=True,
        is_training=False
    )
    print(f"   ✅ 加载 {len(dataset)} 样本，测试 {num_samples} 个")
    
    # 加载两个模型
    print("\n2. 加载模型...")
    
    # Baseline (预训练)
    baseline_model = get_model()
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    print("   ✅ Baseline: 预训练 CellSAM")
    
    # E29 (微调)
    e29_model = get_model()
    checkpoint = torch.load('checkpoints/E29_bf_instance_best.pt', map_location='cpu', weights_only=False)
    e29_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    e29_model = e29_model.to(device)
    e29_model.eval()
    print(f"   ✅ E29: 微调模型 (epoch {checkpoint.get('epoch', '?')}, dice={checkpoint.get('best_dice', 0):.4f})")
    
    # 验证权重差异
    print("\n3. 验证权重差异...")
    decoder_keys = [k for k in baseline_model.state_dict().keys() if 'mask_decoder' in k]
    total_diff = 0
    for k in decoder_keys:
        diff = (baseline_model.state_dict()[k] - e29_model.state_dict()[k]).abs().mean().item()
        total_diff += diff
    avg_diff = total_diff / len(decoder_keys)
    print(f"   Decoder 平均权重差异: {avg_diff:.6f}")
    
    # 测试
    print(f"\n4. 对照测试 ({num_samples} 样本)...")
    
    baseline_dices, baseline_pqs = [], []
    e29_dices, e29_pqs = [], []
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[idx]
            
            image = sample['image']
            gt_mask = sample['mask'].numpy()
            boxes = sample['boxes']
            
            img_np = image.numpy().transpose(1, 2, 0)
            boxes_array = boxes.numpy() if torch.is_tensor(boxes) else boxes
            boxes_list = [box.tolist() for box in boxes_array if sum(box) > 0]
            
            if len(boxes_list) == 0:
                continue
            
            # Baseline 推理
            try:
                baseline_pred, _, _ = segment_cellular_image(
                    img_np, baseline_model, normalize=False,
                    bounding_boxes=boxes_list, device=device
                )
                dice, pq = compute_metrics(baseline_pred, gt_mask)
                baseline_dices.append(dice)
                baseline_pqs.append(pq)
            except:
                pass
            
            # E29 推理
            try:
                e29_pred, _, _ = segment_cellular_image(
                    img_np, e29_model, normalize=False,
                    bounding_boxes=boxes_list, device=device
                )
                dice, pq = compute_metrics(e29_pred, gt_mask)
                e29_dices.append(dice)
                e29_pqs.append(pq)
            except:
                pass
    
    # 结果
    print("\n" + "="*70)
    print("对照测试结果:")
    print("-"*70)
    print(f"{'模型':<20} {'Instance Dice':<25} {'PQ@0.5':<25}")
    print("-"*70)
    print(f"{'Baseline (预训练)':<20} {np.mean(baseline_dices):.4f} ± {np.std(baseline_dices):.4f}     {np.mean(baseline_pqs):.4f} ± {np.std(baseline_pqs):.4f}")
    print(f"{'E29 (微调)':<20} {np.mean(e29_dices):.4f} ± {np.std(e29_dices):.4f}     {np.mean(e29_pqs):.4f} ± {np.std(e29_pqs):.4f}")
    print("-"*70)
    
    dice_improvement = np.mean(e29_dices) - np.mean(baseline_dices)
    pq_improvement = np.mean(e29_pqs) - np.mean(baseline_pqs)
    
    print(f"\n改进:")
    print(f"  Instance Dice: {dice_improvement:+.4f} ({dice_improvement/np.mean(baseline_dices)*100:+.1f}%)")
    print(f"  PQ@0.5: {pq_improvement:+.4f} ({pq_improvement/max(np.mean(baseline_pqs),0.001)*100:+.1f}%)")
    print(f"\n权重差异: {avg_diff:.6f}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_comparison(args.samples, args.device)
