"""
E2E Smoke Test - Runtime baseline data collection.

Runs evaluation using the unified inference core to collect baseline metrics.
Supports fixed random seed, sample ID tracking, TP/FP/FN reporting,
and per-sample CSV output for reproducible comparisons.

Usage:
    python tools/smoke_test_e2e.py --n_samples 30
    python tools/smoke_test_e2e.py --n_samples 30 --checkpoint checkpoints/E29_.../best_model.pt
    python tools/smoke_test_e2e.py --n_samples 30 --seed 42 --output baseline_results.csv
"""
import sys
import argparse
import time
import csv
import random

sys.path.insert(0, 'cellSAM_source')
sys.path.insert(0, 'src')

import numpy as np
import torch

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from torch.utils.data import DataLoader
from inference.core import segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
from metrics.instance_metrics import compute_all_metrics


def run_smoke_test(n_samples=30, checkpoint_path=None, device=None, seed=42, output_csv=None):
    """Run E2E smoke test with reproducible sampling and detailed reporting."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 70)
    print("E2E Smoke Test - Runtime Baseline")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  Samples:    {n_samples}")
    print(f"  Seed:       {seed}")
    print(f"  Checkpoint: {checkpoint_path or 'pretrained (no finetune)'}")
    print(f"  Output:     {output_csv or '(stdout only)'}")
    print()
    
    # Load model
    if checkpoint_path:
        model, adapter, ckpt_info = load_cellsam_checkpoint(checkpoint_path, device=device)
        print(f"  Checkpoint info: {ckpt_info}")
    else:
        model = get_model()
        model.eval()
        model.to(device)
        adapter = None
    
    # Load validation data with deterministic sampling
    all_val_ids = load_split_ids("val", "data/splits")
    if len(all_val_ids) == 0:
        print("[SKIP] No validation data found in data/splits/")
        return None
    
    # Shuffle with fixed seed, then take first n_samples
    shuffled_ids = list(all_val_ids)
    random.shuffle(shuffled_ids)
    val_ids = shuffled_ids[:n_samples]
    
    print(f"  Val pool:   {len(all_val_ids)} total, selected {len(val_ids)}")
    print(f"  Sample IDs: {val_ids[:5]}{'...' if len(val_ids) > 5 else ''}")
    print()
    
    val_dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        is_training=False,
        max_boxes_per_image=30,
        sample_ids=val_ids,
        use_bf_only=True,
        use_semantic_mapping=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    config = InferenceConfig.default()
    
    # Run inference
    all_metrics = []
    sample_ids_used = []
    t_start = time.time()
    
    print(f"{'Idx':>4} {'SampleID':>12} {'BM-1to1':>8} {'BM-Cov':>8} {'PQ':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'AJI':>6} {'nGT':>4} {'nPred':>5} {'Conflict':>10}")
    print("-" * 90)
    
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image']
        masks = batch['mask']
        boxes_list = batch['boxes']
        
        for i in range(images.shape[0]):
            image = images[i]
            gt_mask = masks[i].numpy()
            sample_boxes = boxes_list[i]
            
            if sample_boxes.shape[0] == 0:
                continue
            
            sample_id = val_ids[batch_idx] if batch_idx < len(val_ids) else f"unk_{batch_idx}"
            
            # Apply adapter if present
            image_for_seg = image
            if adapter is not None:
                img_np = image.numpy()
                img_semantic = np.stack([img_np[2], img_np[0], img_np[1]], axis=0)
                img_tensor = torch.from_numpy(img_semantic).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    img_tensor = adapter(img_tensor)
                image_for_seg = img_tensor.squeeze(0)  # keep on device
            
            # Segment
            with torch.no_grad():
                result = segment_with_boxes(
                    model=model,
                    image=image_for_seg,
                    boxes=sample_boxes,
                    config=config,
                    device=device,
                    return_confidence=False
                )
            
            pred_mask = result.instance_mask
            
            # Compute metrics (now includes TP/FP/FN)
            metrics = compute_all_metrics(pred_mask, gt_mask)
            metrics['conflict_pixels'] = result.conflict_pixels
            metrics['sample_id'] = sample_id
            all_metrics.append(metrics)
            sample_ids_used.append(sample_id)
            
            print(f"{len(all_metrics):>4} {str(sample_id):>12} "
                  f"{metrics['bm_1to1_dice']:>8.4f} {metrics['bm_coverage_dice']:>8.4f} "
                  f"{metrics['pq']:>6.4f} "
                  f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4} "
                  f"{metrics['aji']:>6.4f} "
                  f"{int(metrics['n_gt_cells']):>4} {int(metrics['n_pred_cells']):>5} "
                  f"{metrics['conflict_pixels']:>10}")
    
    elapsed = time.time() - t_start
    
    if not all_metrics:
        print("[WARN] No samples processed")
        return None
    
    # Aggregate (exclude non-numeric keys)
    numeric_keys = [k for k in all_metrics[0] if k != 'sample_id' and isinstance(all_metrics[0][k], (int, float))]
    summary = {}
    for key in numeric_keys:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    print()
    print("=" * 70)
    print("Baseline Metrics Summary")
    print("=" * 70)
    print(f"  Samples evaluated: {len(all_metrics)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_metrics):.1f}s/sample)")
    print(f"  Seed: {seed}")
    print()
    print(f"  {'Metric':20s} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key in ['bm_1to1_dice', 'bm_coverage_dice', 'gap_dice', 'pq', 'sq', 'rq',
                'tp', 'fp', 'fn', 'aji', 'semantic_dice', 'n_gt_cells', 'n_pred_cells',
                'conflict_pixels']:
        if key in summary:
            s = summary[key]
            print(f"  {key:20s} {s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")
    print("=" * 70)
    
    # Save CSV if requested
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id'] + numeric_keys)
            writer.writeheader()
            for m in all_metrics:
                row = {k: m[k] for k in ['sample_id'] + numeric_keys}
                writer.writerow(row)
        print(f"\nPer-sample results saved to: {output_csv}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E2E Smoke Test')
    parser.add_argument('--n_samples', type=int, default=30,
                        help='Number of validation samples (default: 30)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for per-sample results')
    args = parser.parse_args()
    
    run_smoke_test(
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed,
        output_csv=args.output
    )
