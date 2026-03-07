#!/usr/bin/env python3
"""
Unified checkpoint evaluator.

Evaluates a trained checkpoint on val and/or test splits,
outputting all standard metrics: PQ, SQ, RQ(=F1), Precision, Recall, AJI,
BM-1to1 Dice, BM-Coverage Dice, Semantic Dice.

Usage:
  python tools/eval_checkpoint.py \
    --checkpoint checkpoints/T32_NeckOnly_Baseline_seed42_.../best_model.pt \
    --config src/config/t32_stage2_like_neck_only.yaml \
    --splits val test \
    --output-dir experiments/t32_eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from inference.core import segment_with_boxes, InferenceConfig
from metrics.instance_metrics import compute_all_metrics
from cellSAM import get_model


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_split(model, dataloader, device, box_expand=0.1, split_name="val"):
    """Evaluate model on one split, return per-image + aggregate metrics."""
    model.eval()
    infer_cfg = InferenceConfig.default()
    infer_cfg.box_expand = box_expand
    infer_cfg.validate_size = False

    per_image = []
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes']

            for i in range(images.shape[0]):
                sample_boxes = boxes[i]
                sample_mask = masks[i]

                if sample_boxes.shape[0] == 0 or sample_boxes.sum() == 0:
                    continue

                result = segment_with_boxes(
                    model=model,
                    image=images[i],
                    boxes=sample_boxes,
                    config=infer_cfg,
                    device=str(device),
                )

                pred_np = result.instance_mask
                gt_np = sample_mask.cpu().numpy()
                metrics = compute_all_metrics(pred_np, gt_np, iou_threshold=0.5)

                per_image.append(metrics)
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{split_name}] Processed {batch_idx + 1}/{len(dataloader)} batches")

    n = len(per_image)
    if n == 0:
        print(f"  WARNING: No valid samples in {split_name}")
        return {}

    # Per-image averages
    agg = {}
    metric_keys = ['pq', 'sq', 'rq', 'bm_1to1_dice', 'bm_coverage_dice', 'aji', 'semantic_dice']
    for key in metric_keys:
        values = [m[key] for m in per_image]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    # Global detection metrics (from total TP/FP/FN, not per-image average)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    agg['precision'] = precision
    agg['recall'] = recall
    agg['f1'] = f1
    agg['tp_total'] = total_tp
    agg['fp_total'] = total_fp
    agg['fn_total'] = total_fn
    agg['n_samples'] = n

    return agg


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate checkpoint with full metrics")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Which splits to evaluate (default: val test)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results JSON")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--box-expand", type=float, default=0.1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(args.config)

    # Load model (same as train.py)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = get_model()
    model.adv_mode = True
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded on {device}")

    # Output directory
    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.output_dir = str(ckpt_dir / "eval_results")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Evaluating split: {split}")
        print(f"{'='*60}")

        # Load split IDs
        split_ids = load_split_ids(split, config['data']['splits_dir'])
        if not split_ids:
            print(f"  Skip: no IDs for split '{split}'")
            continue

        # Create dataset (no augmentation for eval)
        ds = AugmentedAllenDataset(
            data_dir=config['data']['processed_data_dir'],
            sample_ids=split_ids,
            target_size=tuple(config['data']['target_size']),
            max_boxes_per_image=config['data'].get('max_boxes_per_image', 30),
            use_bf_only=config['data'].get('use_bf_only', True),
            use_semantic_mapping=config['data'].get('use_semantic_mapping', False),
            is_training=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, collate_fn=collate_fn)
        print(f"  Dataset: {len(ds)} images")

        t0 = time.time()
        split_results = evaluate_split(model, loader, device,
                                       box_expand=args.box_expand,
                                       split_name=split)
        elapsed = time.time() - t0

        if split_results:
            split_results['elapsed_seconds'] = round(elapsed, 1)
            results[split] = split_results

            # Print summary
            print(f"\n  Results ({split}, n={split_results['n_samples']}):")
            print(f"  PQ:        {split_results['pq_mean']:.4f} ± {split_results['pq_std']:.4f}")
            print(f"  SQ:        {split_results['sq_mean']:.4f} ± {split_results['sq_std']:.4f}")
            print(f"  RQ:        {split_results['rq_mean']:.4f} ± {split_results['rq_std']:.4f}")
            print(f"  F1:        {split_results['f1']:.4f}")
            print(f"  Precision: {split_results['precision']:.4f}")
            print(f"  Recall:    {split_results['recall']:.4f}")
            print(f"  BM-Dice:   {split_results['bm_1to1_dice_mean']:.4f}")
            print(f"  AJI:       {split_results['aji_mean']:.4f}")
            print(f"  Sem Dice:  {split_results['semantic_dice_mean']:.4f}")
            print(f"  TP/FP/FN:  {split_results['tp_total']}/{split_results['fp_total']}/{split_results['fn_total']}")
            print(f"  Time:      {elapsed:.1f}s")

    # Add metadata
    results['metadata'] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'config': args.config,
        'splits': args.splits,
        'box_expand': args.box_expand,
        'script': 'tools/eval_checkpoint.py',
    }

    # Save
    out_file = output_dir / "results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()

