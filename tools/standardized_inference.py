"""
标准化推理脚本 - Oracle 评估 (GT 框)
====================================================
创建日期: 2026-02-07, 更新: 2026-02-10
目的: 使用统一推理核心和指标进行 Oracle 评估

指标:
- BM-1to1 Dice: Hungarian 一对一匹配 (主指标)
- BM-Coverage Dice: 每 GT 取最大  (辅助诊断)
- Gap: Coverage - 1to1             (粘连诊断)
- PQ@0.5: Panoptic Quality
- 使用 GT 框进行推理
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, load_split_ids
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
)
from metrics.instance_metrics import compute_all_metrics


# NOTE: 本地 compute_best_match_instance_dice / compute_pq 已移除
# 统一使用 metrics.instance_metrics 中的实现


def run_standardized_inference(
    model_name: str,
    model_or_checkpoint,
    num_samples: int = 71,
    device: str = 'cuda',
    adapter_cls=None,
    adapter_kwargs: dict = None,
):
    """
    Run Oracle evaluation (GT boxes) using unified core.
    
    Args:
        adapter_cls: Adapter class (e.g. IndependentChannelAdapter), None for BF-only
        adapter_kwargs: Adapter constructor kwargs
    
    Returns:
        Results dict with BM-1to1, BM-Coverage, Gap, PQ
    """
    print("="*70)
    print(f"标准化推理 (Oracle): {model_name}")
    print("="*70)
    
    # 加载模型
    print("\n1. 加载模型...")
    if isinstance(model_or_checkpoint, str):
        model, adapter, ckpt_info = load_cellsam_checkpoint(
            model_or_checkpoint, device,
            adapter_cls=adapter_cls,
            adapter_kwargs=adapter_kwargs,
        )
        print(f"   Loaded checkpoint: {ckpt_info}")
    else:
        model = model_or_checkpoint
        model = model.to(device)
        model.eval()
        print("   ✅ 使用传入的模型")
    
    # 统一推理配置 (单一来源)
    infer_cfg = InferenceConfig.default()
    
    # 加载数据
    print("\n2. 加载验证数据...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=True,
        is_training=False
    )
    print(f"   ✅ 加载 {len(dataset)} 样本，测试 {min(num_samples, len(dataset))} 个")
    
    # 推理
    all_metrics = []
    
    print(f"\n3. 推理中...")
    print(f"   配置: threshold={infer_cfg.mask_threshold}, "
          f"conflict={infer_cfg.conflict_policy}, "
          f"clipping={infer_cfg.apply_box_clipping}")
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[idx]
            
            image = sample['image']  # [C, H, W]
            gt_mask = sample['mask'].numpy()
            boxes = sample['boxes']  # [N, 4]
            
            # 过滤零 box
            valid = boxes.sum(dim=1) > 0
            boxes = boxes[valid]
            
            if len(boxes) == 0:
                continue
            
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
                result = segment_with_boxes(
                    model=model,
                    image=image_for_seg,
                    boxes=boxes,
                    config=infer_cfg,
                    device=device,
                )
                
                m = compute_all_metrics(result.instance_mask, gt_mask)
                m['conflict_pixels'] = result.conflict_pixels
                all_metrics.append(m)
                
            except Exception as e:
                print(f"   ⚠️ 样本 {idx} 失败: {e}")
                continue
    
    # 汇总结果
    if not all_metrics:
        print("   ❌ 无有效结果")
        return {}
    
    results = {
        'model': model_name,
        'task': 'Oracle (GT boxes)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_samples': len(all_metrics),
        'bm_1to1_dice_mean': float(np.mean([m['bm_1to1_dice'] for m in all_metrics])),
        'bm_1to1_dice_std': float(np.std([m['bm_1to1_dice'] for m in all_metrics])),
        'bm_coverage_dice_mean': float(np.mean([m['bm_coverage_dice'] for m in all_metrics])),
        'gap_dice_mean': float(np.mean([m['gap_dice'] for m in all_metrics])),
        'pq_mean': float(np.mean([m['pq'] for m in all_metrics])),
        'pq_std': float(np.std([m['pq'] for m in all_metrics])),
        'aji_mean': float(np.mean([m['aji'] for m in all_metrics])),
        'semantic_dice_mean': float(np.mean([m['semantic_dice'] for m in all_metrics])),
        'gt_cells_per_image': float(np.mean([m['n_gt_cells'] for m in all_metrics])),
        'pred_cells_per_image': float(np.mean([m['n_pred_cells'] for m in all_metrics])),
    }
    
    print("\n" + "="*70)
    print(f"=== Evaluation Report ===")
    print(f"Task:  Oracle (GT boxes)")
    print(f"Model: {model_name}")
    print("-"*70)
    print(f"  BM-1to1 Dice:    {results['bm_1to1_dice_mean']:.4f} ± {results['bm_1to1_dice_std']:.4f}")
    print(f"  BM-Coverage Dice:{results['bm_coverage_dice_mean']:.4f}")
    print(f"  Gap (粘连指标):   {results['gap_dice_mean']:.4f}")
    print(f"  PQ@0.5:          {results['pq_mean']:.4f} ± {results['pq_std']:.4f}")
    print(f"  AJI:             {results['aji_mean']:.4f}")
    print(f"  Semantic Dice:   {results['semantic_dice_mean']:.4f}")
    print(f"  细胞数: GT={results['gt_cells_per_image']:.1f}, Pred={results['pred_cells_per_image']:.1f}")
    print("="*70)
    
    return results


def compare_baseline_vs_e29(num_samples=71, device='cuda'):
    """对比 Baseline 和 E29"""
    print("\n" + "#"*70)
    print("# Oracle 对比: Baseline vs E29")
    print("#"*70)
    
    # Baseline
    print("\n>>> Baseline (预训练 CellSAM)")
    baseline_model = get_model()
    baseline_results = run_standardized_inference(
        "Baseline (预训练)", baseline_model, num_samples, device
    )
    
    # E29
    print("\n>>> E29 (BF Instance P1 微调)")
    e29_results = run_standardized_inference(
        "E29 (BF Instance P1)", "checkpoints/E29_bf_instance_best.pt", num_samples, device
    )
    
    # 对比
    print("\n" + "="*70)
    print("对比结果 (Oracle):")
    print("-"*70)
    print(f"{'模型':<25} {'BM-1to1':<12} {'BM-Coverage':<12} {'Gap':<8} {'PQ@0.5':<12}")
    print("-"*70)
    for r in [baseline_results, e29_results]:
        if r:
            print(f"{r['model']:<25} {r['bm_1to1_dice_mean']:.4f}      {r['bm_coverage_dice_mean']:.4f}       {r['gap_dice_mean']:.4f}  {r['pq_mean']:.4f}")
    print("="*70)
    
    if baseline_results and e29_results:
        d1 = e29_results['bm_1to1_dice_mean'] - baseline_results['bm_1to1_dice_mean']
        dp = e29_results['pq_mean'] - baseline_results['pq_mean']
        print(f"\n改进:")
        print(f"  BM-1to1 Dice: {d1:+.4f}")
        print(f"  PQ@0.5:       {dp:+.4f}")
    
    return baseline_results, e29_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=71)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default=None, 
                       help='checkpoint path or "baseline"')
    args = parser.parse_args()
    
    if args.model:
        if args.model == 'baseline':
            model = get_model()
            run_standardized_inference("Baseline", model, args.samples, args.device)
        else:
            run_standardized_inference(f"Custom ({args.model})", args.model, args.samples, args.device)
    else:
        # 默认: 对比 baseline 和 E29
        compare_baseline_vs_e29(args.samples, args.device)
