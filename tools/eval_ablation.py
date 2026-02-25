"""
Ablation Oracle 评估脚本 — 自动查找 checkpoint 并评估

Usage:
    python tools/eval_ablation.py --exp-dir checkpoints/Ab1_no_boundary_20260222_...
    python tools/eval_ablation.py --exp-dir checkpoints/Ab1_no_boundary_20260222_... --output experiments/ablation_eval/

自动查找 best_model.pt, 加载模型, 在 test(73) 上跑 Oracle 评估.
"""

import sys
import argparse
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
from inference.core import segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
from metrics.instance_metrics import compute_all_metrics


def eval_checkpoint(ckpt_path: str, device: str = "cuda") -> dict:
    """Evaluate a single checkpoint on test(73) with GT boxes."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model via unified loader (auto-detects LoRA + adapter)
    model, adapter, info = load_cellsam_checkpoint(ckpt_path, device=str(device))
    epoch = info.get('epoch', '?')
    train_dice = info.get('best_dice', 0)
    
    # Extract train_pq from checkpoint directly (not in info)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    train_pq = checkpoint.get('best_pq', 0) if isinstance(checkpoint, dict) else 0
    
    print(f"  Loaded: epoch={epoch}, train_pq={train_pq:.4f}, train_dice={train_dice:.4f}")
    
    # Load test data (BF-only, consistent with ablation training)
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids,
        use_bf_only=True,
    )
    
    infer_cfg = InferenceConfig.default()
    all_metrics = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Oracle eval"):
            sample = dataset[idx]
            image = sample['image']
            gt_mask = sample['mask'].numpy()
            boxes = sample['boxes']
            num_boxes = sample['num_boxes']
            
            valid_mask = boxes[:num_boxes].sum(dim=1) > 0
            valid_boxes = boxes[:num_boxes][valid_mask]
            
            if len(valid_boxes) == 0:
                continue
            
            try:
                result = segment_with_boxes(
                    model=model, image=image, boxes=valid_boxes,
                    config=infer_cfg, device=str(device),
                )
                m = compute_all_metrics(result.instance_mask, gt_mask)
                m['conflict_pixels'] = result.conflict_pixels
                all_metrics.append(m)
            except Exception as e:
                print(f"  Sample {idx} failed: {e}")
                continue
    
    # Aggregate
    aggregated = {}
    for key in ['pq', 'bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
                'aji', 'semantic_dice', 'tp', 'fp', 'fn', 'conflict_pixels']:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    
    return {
        'checkpoint': ckpt_path,
        'epoch': epoch,
        'train_pq': train_pq,
        'train_dice': train_dice,
        'n_samples': len(all_metrics),
        'metrics': aggregated,
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation Oracle Evaluation')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory (contains best_model.pt)')
    parser.add_argument('--output', type=str, default='experiments/ablation_eval',
                        help='Output directory for results JSON')
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    ckpt = exp_dir / "best_model.pt"
    
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found")
        sys.exit(1)
    
    exp_name = exp_dir.name
    # Strip timestamp suffix for cleaner experiment name
    # e.g. "Ab1_no_boundary_20260222_120000" -> "Ab1_no_boundary"
    parts = exp_name.split('_')
    # Find where timestamp starts (8 consecutive digits)
    clean_name = exp_name
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            clean_name = '_'.join(parts[:i])
            break
    
    print(f"=== Evaluating: {clean_name} ===")
    print(f"  Checkpoint: {ckpt}")
    
    result = eval_checkpoint(str(ckpt))
    result['experiment_name'] = clean_name
    
    # Print summary
    m = result['metrics']
    print(f"\n=== {clean_name} (test73, Oracle) ===")
    print(f"  PQ:        {m['pq']['mean']:.4f} +/- {m['pq']['std']:.4f}")
    print(f"  BM-Dice:   {m['bm_1to1_dice']['mean']:.4f} +/- {m['bm_1to1_dice']['std']:.4f}")
    print(f"  AJI:       {m['aji']['mean']:.4f} +/- {m['aji']['std']:.4f}")
    print(f"  Sem.Dice:  {m['semantic_dice']['mean']:.4f} +/- {m['semantic_dice']['std']:.4f}")
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{clean_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'result': result,
        }, f, indent=2)
    
    print(f"  Saved: {output_file}")


if __name__ == "__main__":
    main()
