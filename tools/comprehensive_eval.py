"""
全面评估脚本：Oracle 评估 (GT 框)
使用统一推理核心和指标

指标 (统一口径):
- BM-1to1 Dice, BM-Coverage Dice, Gap
- PQ@0.5, AJI, Semantic Dice

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
from adapters.channel_adapter import IndependentChannelAdapter
from augmented_dataset import AugmentedAllenDataset
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
)
from metrics.instance_metrics import compute_all_metrics

# Checkpoints
CHECKPOINTS = {
    "BF_Baseline_Full": {
        "path": "checkpoints/bf_baseline_full_best.pt",
        "adapter": False,
        "semantic": False,
    },
    "Semantic_Adapter": {
        "path": "checkpoints/semantic_adapter_v2_best.pt", 
        "adapter": True,
        "semantic": True,
    },
}


# NOTE: 本地 load_model_with_config / segment_image 已移除
# 统一使用 inference.core.load_cellsam_checkpoint + segment_with_boxes


def run_comprehensive_evaluation():
    """运行 Oracle 全面评估 (统一口径)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 统一推理配置 (单一来源)
    infer_cfg = InferenceConfig.default()
    print(f"Inference config: threshold={infer_cfg.mask_threshold}, "
          f"conflict={infer_cfg.conflict_policy}, "
          f"box_expand={infer_cfg.box_expand}")
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Test samples: {len(dataset)}")
    
    results = {}
    
    for model_name, config in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        if not Path(config["path"]).exists():
            print(f"  ⚠️ Checkpoint not found: {config['path']}")
            continue
        
        # 统一加载 (adapter 支持)
        adapter_cls = IndependentChannelAdapter if config["adapter"] else None
        adapter_kwargs = {"kernel_size": 3, "use_relu": True} if config["adapter"] else None
        
        model, adapter, ckpt_info = load_cellsam_checkpoint(
            config["path"], str(device),
            adapter_cls=adapter_cls,
            adapter_kwargs=adapter_kwargs,
        )
        print(f"  Checkpoint info: {ckpt_info}")
        
        all_metrics = []
        
        for idx in tqdm(range(len(dataset)), desc=model_name):
            sample = dataset[idx]
            
            image = sample['image']  # [C, H, W] tensor
            gt_mask = sample['mask'].numpy()
            boxes = sample['boxes']
            num_boxes = sample['num_boxes']
            
            # 过滤零 box
            valid_mask = boxes[:num_boxes].sum(dim=1) > 0
            valid_boxes = boxes[:num_boxes][valid_mask]
            
            if len(valid_boxes) == 0:
                continue
            
            # Adapter 预处理 (若有)
            if adapter is not None and config["semantic"]:
                img_np = image.numpy()
                img_semantic = np.stack([img_np[2], img_np[0], img_np[1]], axis=0)
                img_tensor = torch.from_numpy(img_semantic).float().unsqueeze(0).to(device)
                img_tensor = adapter(img_tensor)
                image_for_seg = img_tensor.squeeze(0)  # [3, H, W]
            else:
                image_for_seg = image
            
            try:
                result = segment_with_boxes(
                    model=model,
                    image=image_for_seg,
                    boxes=valid_boxes,
                    config=infer_cfg,
                    device=str(device),
                )
                
                m = compute_all_metrics(result.instance_mask, gt_mask)
                m['conflict_pixels'] = result.conflict_pixels
                all_metrics.append(m)
            except Exception as e:
                print(f"  ⚠️ Sample {idx} failed: {e}")
                continue
        
        # Aggregate results
        if all_metrics:
            aggregated = {}
            for key in ['bm_1to1_dice', 'bm_coverage_dice', 'gap_dice', 'pq', 'aji', 'semantic_dice']:
                values = [m[key] for m in all_metrics]
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
            
            results[model_name] = aggregated
            
            print(f"\n=== {model_name} ===")
            print(f"  BM-1to1 Dice: {aggregated['bm_1to1_dice']['mean']:.4f} ± {aggregated['bm_1to1_dice']['std']:.4f}")
            print(f"  BM-Coverage:  {aggregated['bm_coverage_dice']['mean']:.4f} ± {aggregated['bm_coverage_dice']['std']:.4f}")
            print(f"  Gap:          {aggregated['gap_dice']['mean']:.4f}")
            print(f"  PQ@0.5:       {aggregated['pq']['mean']:.4f} ± {aggregated['pq']['std']:.4f}")
            print(f"  AJI:          {aggregated['aji']['mean']:.4f} ± {aggregated['aji']['std']:.4f}")
            print(f"  Semantic Dice:{aggregated['semantic_dice']['mean']:.4f} ± {aggregated['semantic_dice']['std']:.4f}")
    
    # Save results
    output_dir = Path("experiments/comprehensive_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "task": "Oracle (GT boxes)",
            "inference_config": {
                "mask_threshold": infer_cfg.mask_threshold,
                "box_expand": infer_cfg.box_expand,
                "conflict_policy": infer_cfg.conflict_policy,
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'results.json'}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()
