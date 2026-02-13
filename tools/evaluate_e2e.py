"""
端到端评估：DAPI 检测 → SAM 分割

与 standardized_inference.py 的区别：
- standardized_inference: 使用 GT 框 (Oracle) → 评估分割能力上限
- 本脚本: 使用 DAPI 检测框 (E2E) → 评估真实部署效果

指标 (统一口径):
- BM-1to1 Dice: Hungarian 一对一匹配 (主指标)
- BM-Coverage Dice: 每 GT 取最大  (辅助)
- Gap: Coverage - 1to1 (粘连诊断)
- PQ@0.5
更新: 2026-02-10 - 接入统一推理核心
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset
from detection.dapi import detect_and_create_boxes
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
)
from metrics.instance_metrics import compute_all_metrics



# NOTE: 本地 compute_metrics 已移除
# 统一使用 metrics.instance_metrics.compute_all_metrics


def evaluate_e2e(checkpoint_path="checkpoints/bf_baseline_full_best.pt",
                 adapter_cls=None, adapter_kwargs=None):
    """End-to-end evaluation: DAPI detection -> SAM segmentation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model (unified loader with adapter support)
    model, adapter, ckpt_info = load_cellsam_checkpoint(
        checkpoint_path, str(device),
        adapter_cls=adapter_cls,
        adapter_kwargs=adapter_kwargs,
    )
    print(f"Model loaded: {ckpt_info}")
    
    # 统一推理配置 (单一来源)
    infer_cfg = InferenceConfig.default()
    
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
        
        image = sample['image']  # (C, H, W) tensor
        gt_mask = sample['mask'].numpy().astype(np.int32)
        
        # Step 1: DAPI detection
        dapi = image[1].numpy() if image.shape[0] > 1 else image[0].numpy()
        result = detect_and_create_boxes(dapi)
        boxes = result[0] if isinstance(result, tuple) else result
        
        if not boxes or len(boxes) == 0:
            all_results.append(compute_all_metrics(
                np.zeros_like(gt_mask), gt_mask
            ))
            continue
        
        # Step 2: SAM segmentation via unified core
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        # Adapter preprocessing (if adapter loaded)
        if adapter is not None:
            img_np = image.numpy()
            # Channel rearrange for semantic adapter: [BF,DAPI,ACTN2] -> [ACTN2,BF,DAPI]
            img_semantic = np.stack([img_np[2], img_np[0], img_np[1]], axis=0)
            img_tensor = torch.from_numpy(img_semantic).float().unsqueeze(0).to(device)
            with torch.no_grad():
                img_tensor = adapter(img_tensor)
            image_for_seg = img_tensor.squeeze(0)  # [3, H, W]
        else:
            image_for_seg = image
        
        try:
            seg_result = segment_with_boxes(
                model=model,
                image=image_for_seg,
                boxes=boxes_tensor,
                config=infer_cfg,
                device=str(device),
            )
            
            m = compute_all_metrics(seg_result.instance_mask, gt_mask)
            m['conflict_pixels'] = seg_result.conflict_pixels
            all_results.append(m)
            
        except Exception as e:
            print(f"  \u26a0\ufe0f Sample {idx} failed: {e}")
            all_results.append(compute_all_metrics(
                np.zeros_like(gt_mask), gt_mask
            ))
    
    # Aggregate results
    print("\n" + "="*70)
    print("=== Evaluation Report ===")
    print(f"Task:  E2E (DAPI detection \u2192 SAM segmentation)")
    print("="*70)
    
    for key in ['bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
                'pq', 'sq', 'rq', 'aji', 'semantic_dice']:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            print(f"  {key:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    avg_gt = np.mean([r.get('n_gt_cells', 0) for r in all_results])
    avg_pred = np.mean([r.get('n_pred_cells', 0) for r in all_results])
    avg_tp = np.mean([r.get('tp', 0) for r in all_results])
    avg_fp = np.mean([r.get('fp', 0) for r in all_results])
    avg_fn = np.mean([r.get('fn', 0) for r in all_results])
    print(f"  {'n_gt_cells':20s}: {avg_gt:.1f}")
    print(f"  {'n_pred_cells':20s}: {avg_pred:.1f}")
    print(f"  {'TP/FP/FN':20s}: {avg_tp:.1f} / {avg_fp:.1f} / {avg_fn:.1f}")
    print("="*70)
    
    # Save results
    output_dir = Path("experiments/e2e_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'task': 'E2E (DAPI detection -> SAM segmentation)',
            'config': {
                'model': checkpoint_path,
                'detection': 'DAPI detect_and_create_boxes',
                'inference': {
                    'mask_threshold': infer_cfg.mask_threshold,
                    'box_expand': infer_cfg.box_expand,
                    'conflict_policy': infer_cfg.conflict_policy,
                    'apply_box_clipping': infer_cfg.apply_box_clipping,
                },
            },
            'summary': {
                key: {'mean': float(np.mean([r[key] for r in all_results if key in r])),
                      'std': float(np.std([r[key] for r in all_results if key in r]))}
                for key in ['bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
                            'pq', 'sq', 'rq', 'aji', 'semantic_dice',
                            'tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells']
            },
            'per_sample': all_results,
        }, f, indent=2, default=str)
    
    print(f"\n结果保存至: {output_dir / 'results.json'}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='E2E Evaluation (DAPI detection → SAM segmentation)')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/bf_baseline_full_best.pt",
                        help='Path to model checkpoint')
    args = parser.parse_args()
    evaluate_e2e(checkpoint_path=args.checkpoint)
