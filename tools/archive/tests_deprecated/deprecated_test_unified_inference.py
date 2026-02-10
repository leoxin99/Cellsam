"""
[DEPRECATED] Test unified inference core module.

This script uses the pre-Phase-0 API:
  - compute_best_match_dice (removed, now compute_bm_1to1_dice)
  - get_default_config (removed, now InferenceConfig.default())

Use tools/test_phase0_regression.py instead.
"""
import warnings
warnings.warn(
    "test_unified_inference.py is deprecated. Use test_phase0_regression.py.",
    DeprecationWarning, stacklevel=2
)
import sys
sys.path.insert(0, 'cellSAM_source')
sys.path.insert(0, 'src')

import torch
import numpy as np
from torch.utils.data import DataLoader

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from inference.core import segment_with_boxes, InferenceConfig  # get_default_config removed
from metrics.instance_metrics import compute_bm_1to1_dice, compute_pq, compute_all_metrics  # compute_best_match_dice removed


def test_unified_inference():
    print("=" * 60)
    print("测试统一推理核心模块")
    print("=" * 60)
    
    # 加载数据
    val_ids = load_split_ids("val", "data/splits")[:2]
    val_dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        is_training=False,
        max_boxes_per_image=20,
        sample_ids=val_ids,
        use_bf_only=True,
        use_semantic_mapping=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 加载模型
    print("\n加载模型...")
    model = get_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"设备: {device}")
    
    # 推理配置
    config = get_default_config()
    print(f"\n推理配置:")
    print(f"  mask_threshold: {config.mask_threshold}")
    print(f"  conflict_policy: {config.conflict_policy}")
    print(f"  apply_box_clipping: {config.apply_box_clipping}")
    
    # 测试不同冲突策略
    policies = ["argmax_prob", "first_write", "last_write"]
    
    for policy in policies:
        print(f"\n{'='*60}")
        print(f"测试冲突策略: {policy}")
        print(f"{'='*60}")
        
        config.conflict_policy = policy
        all_dice = []
        all_pq = []
        
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image']  # [B, C, H, W]
            masks = batch['mask']    # [B, H, W]
            boxes = batch['boxes']   # List of [N, 4]
            
            for i in range(images.shape[0]):
                image = images[i]
                gt_mask = masks[i].numpy()
                sample_boxes = boxes[i]
                
                # 跳过空 box
                if sample_boxes.shape[0] == 0:
                    continue
                
                # 统一推理
                result = segment_with_boxes(
                    model=model,
                    image=image,
                    boxes=sample_boxes,
                    config=config,
                    device=device,
                    return_confidence=False
                )
                
                pred_mask = result.instance_mask
                
                # 计算指标
                dice = compute_best_match_dice(pred_mask, gt_mask)
                pq, sq, rq = compute_pq(pred_mask, gt_mask)
                
                all_dice.append(dice)
                all_pq.append(pq)
                
                print(f"  样本 {batch_idx}-{i}: "
                      f"Dice={dice:.4f}, PQ={pq:.4f}, "
                      f"n_instances={result.n_instances}, "
                      f"conflict_pixels={result.conflict_pixels}")
        
        if all_dice:
            print(f"\n  平均 Dice: {np.mean(all_dice):.4f}")
            print(f"  平均 PQ: {np.mean(all_pq):.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    print("[DEPRECATED] This script uses removed APIs (get_default_config, compute_best_match_dice).")
    print("Use instead: python tools/test_phase0_regression.py")
    sys.exit(1)
