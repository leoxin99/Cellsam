#!/usr/bin/env python3
"""T31: Cellpose Paper-Aligned Evaluation

Runs Cellpose cyto3 on test(73) with CellSAM paper-aligned settings:
  - Input: [blank, DAPI, BF]  (R=0, G=nuclear, B=whole-cell proxy)
  - Model: cyto3
  - Channels: [3, 2]  (B=cytoplasm, G=nucleus)
  - Diameter: None (auto) or user-specified

Outputs BOTH project metrics (PQ, BM-Dice, AJI, ...) and
CellSAM paper metrics (F1, Recall) for each sample and aggregated.

Usage:
    python tools/cellpose_paper_aligned_eval.py
    python tools/cellpose_paper_aligned_eval.py --diameter 200
    python tools/cellpose_paper_aligned_eval.py --split val --diameter 120 160 200 240

References:
    - CellSAM eval: cellSAM_source/paper_evaluation/eval_main.py
    - A1 plan: docs/t31_cellpose_baseline_rerun_plan_3.04.md
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics.instance_metrics import compute_all_metrics


# ── Helpers ──────────────────────────────────────────────────

def load_split_ids(split: str, splits_dir: str = "data/splits") -> list:
    """Load image IDs for the given split."""
    path = Path(splits_dir) / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids = [line.strip() for line in open(path) if line.strip()]
    print(f"Loaded {len(ids)} IDs from {path}")
    return ids


def load_image_3ch(image_id: str, processed_dir: str = "data/processed") -> np.ndarray:
    """Load a processed 3-channel image as float32 [H, W, 3].
    
    Processed data layout: image[0]=BF, image[1]=DAPI, image[2]=Actn2
    """
    img_path = Path(processed_dir) / "images" / f"{image_id}.npy"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = np.load(img_path)  # shape: (C, H, W) or (H, W, C) depending on format
    
    # Ensure (H, W, C) format
    if img.ndim == 3 and img.shape[0] in (3, 4, 5):
        img = np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    
    return img.astype(np.float32)


def load_gt_mask(image_id: str, processed_dir: str = "data/processed") -> np.ndarray:
    """Load GT instance segmentation mask."""
    mask_path = Path(processed_dir) / "masks" / f"{image_id}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return np.load(mask_path)


def build_cellpose_input(img_3ch: np.ndarray) -> np.ndarray:
    """Build CellSAM paper-aligned input: [blank, DAPI, BF].
    
    Input img_3ch layout: [BF, DAPI, Actn2] (H, W, 3)
    Output RGB: [blank, DAPI, BF] (H, W, 3)
    
    Per A1 plan:
      - R = 0 (blank)
      - G = DAPI (nuclear)
      - B = BF (whole-cell proxy)
    """
    bf = img_3ch[:, :, 0]     # BF channel
    dapi = img_3ch[:, :, 1]   # DAPI channel
    
    # Per-channel normalize to [0, 1]
    def normalize(ch):
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            return (ch - ch_min) / (ch_max - ch_min)
        return np.zeros_like(ch)
    
    bf_norm = normalize(bf)
    dapi_norm = normalize(dapi)
    blank = np.zeros_like(bf_norm)
    
    # RGB = [blank, DAPI, BF]
    rgb = np.stack([blank, dapi_norm, bf_norm], axis=-1).astype(np.float32)
    return rgb


def compute_paper_metrics(pred_masks: list, gt_masks: list) -> dict:
    """Compute CellSAM paper-aligned metrics (F1, Recall) using cellpose.metrics.
    
    Based on: cellSAM_source/paper_evaluation/cpm.py
    """
    try:
        import cellpose.metrics as cp_metrics
    except ImportError:
        print("WARNING: cellpose.metrics not available, skipping paper metrics")
        return {"f1_mean": None, "recall_mean": None}
    
    # average_precision returns (ap, tp, fp, fn) at threshold [0.5]
    ap, tp_list, fp_list, fn_list = cp_metrics.average_precision(
        gt_masks, pred_masks, threshold=[0.5]
    )
    
    # Per-image F1 and Recall (matching cpm.py logic)
    f1_per_image = []
    recall_per_image = []
    precision_per_image = []
    
    for tp, fp, fn in zip(tp_list, fp_list, fn_list):
        tp, fp, fn = float(tp), float(fp), float(fn)
        f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_per_image.append(f1)
        recall_per_image.append(recall)
        precision_per_image.append(precision)
    
    return {
        "f1_mean": float(np.mean(f1_per_image)),
        "f1_std": float(np.std(f1_per_image)),
        "recall_mean": float(np.mean(recall_per_image)),
        "recall_std": float(np.std(recall_per_image)),
        "precision_mean": float(np.mean(precision_per_image)),
        "precision_std": float(np.std(precision_per_image)),
        "ap_mean": float(np.mean(ap)),
        "f1_per_image": [float(x) for x in f1_per_image],
        "recall_per_image": [float(x) for x in recall_per_image],
    }


# ── Main ─────────────────────────────────────────────────────

def run_cellpose_eval(
    split: str = "test",
    diameter=None,
    model_type: str = "cyto3",
    channels: list = [3, 2],
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    output_dir: str = None,
    gpu: bool = True,
):
    """Run Cellpose evaluation with CellSAM paper-aligned settings."""
    
    from cellpose import models
    
    # ── Setup ────────────────────────────────────────────
    ids = load_split_ids(split, splits_dir)
    
    diam_str = f"d{diameter}" if diameter else "dauto"
    if output_dir is None:
        output_dir = f"experiments/cellpose_paper_aligned_{split}{len(ids)}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"T31: Cellpose Paper-Aligned Evaluation")
    print(f"{'='*60}")
    print(f"  Split:      {split} ({len(ids)} samples)")
    print(f"  Model:      {model_type}")
    print(f"  Channels:   {channels}")
    print(f"  Diameter:   {diameter if diameter else 'auto'}")
    print(f"  Input:      [blank, DAPI, BF]")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")
    
    # ── Load Model ───────────────────────────────────────
    print("Loading Cellpose model...")
    model = models.Cellpose(model_type=model_type, gpu=gpu)
    
    # ── Evaluate ─────────────────────────────────────────
    per_sample = []
    all_pred_masks = []
    all_gt_masks = []
    
    t_start = time.time()
    
    for img_id in tqdm(ids, desc="Evaluating"):
        try:
            # Load data
            img_3ch = load_image_3ch(img_id, processed_dir)
            gt_mask = load_gt_mask(img_id, processed_dir)
            
            # Build paper-aligned input
            rgb = build_cellpose_input(img_3ch)
            
            # Run Cellpose
            masks, flows, styles, diams = model.eval(
                rgb,
                channels=channels,
                diameter=diameter,
            )
            
            # Project metrics (PQ, BM-Dice, AJI, etc.)
            metrics = compute_all_metrics(masks, gt_mask)
            
            # Store for paper metrics
            all_pred_masks.append(masks)
            all_gt_masks.append(gt_mask)
            
            sample_result = {
                "image_id": img_id,
                "estimated_diameter": float(diams) if diams is not None else None,
                "n_pred_cells": int(metrics.get("n_pred_cells", 0)),
                "n_gt_cells": int(metrics.get("n_gt_cells", 0)),
                **{k: float(v) if v is not None else None for k, v in metrics.items()
                   if k not in ("n_pred_cells", "n_gt_cells")},
            }
            per_sample.append(sample_result)
            
        except Exception as e:
            print(f"  ERROR on {img_id}: {e}")
            per_sample.append({"image_id": img_id, "error": str(e)})
    
    t_elapsed = time.time() - t_start
    
    # ── Paper Metrics ────────────────────────────────────
    print("\nComputing CellSAM paper metrics (F1, Recall)...")
    paper_metrics = compute_paper_metrics(all_pred_masks, all_gt_masks)
    
    # ── Aggregate Project Metrics ────────────────────────
    valid = [s for s in per_sample if "error" not in s]
    metric_keys = ["pq", "sq", "rq", "bm_1to1_dice", "bm_coverage_dice", 
                   "aji", "semantic_dice"]
    
    aggregate = {}
    for k in metric_keys:
        values = [s[k] for s in valid if k in s and s[k] is not None]
        if values:
            aggregate[f"{k}_mean"] = float(np.mean(values))
            aggregate[f"{k}_std"] = float(np.std(values))
    
    # TP/FP/FN totals
    tp_total = sum(s.get("tp", 0) for s in valid if isinstance(s.get("tp"), (int, float)))
    fp_total = sum(s.get("fp", 0) for s in valid if isinstance(s.get("fp"), (int, float)))
    fn_total = sum(s.get("fn", 0) for s in valid if isinstance(s.get("fn"), (int, float)))
    n_pred_total = sum(s.get("n_pred_cells", 0) for s in valid)
    n_gt_total = sum(s.get("n_gt_cells", 0) for s in valid)
    
    # ── Results ──────────────────────────────────────────
    results = {
        "experiment": "T31_cellpose_paper_aligned",
        "split": split,
        "n_samples": len(ids),
        "n_valid": len(valid),
        "model_type": model_type,
        "channels": channels,
        "input_encoding": "[blank, DAPI, BF]",
        "diameter": diameter if diameter else "auto",
        "elapsed_seconds": round(t_elapsed, 1),
        "script": "tools/cellpose_paper_aligned_eval.py",
        "project_metrics": {
            **aggregate,
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "n_pred_total": n_pred_total,
            "n_gt_total": n_gt_total,
        },
        "paper_metrics": {
            "f1_mean": paper_metrics.get("f1_mean"),
            "f1_std": paper_metrics.get("f1_std"),
            "recall_mean": paper_metrics.get("recall_mean"),
            "recall_std": paper_metrics.get("recall_std"),
            "precision_mean": paper_metrics.get("precision_mean"),
            "precision_std": paper_metrics.get("precision_std"),
            "ap_mean": paper_metrics.get("ap_mean"),
        },
    }
    
    # ── Print Summary ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_type} | {split}({len(ids)}) | diameter={diameter or 'auto'}")
    print(f"{'='*60}")
    print(f"\n--- Project Metrics ---")
    for k in metric_keys:
        mean_k = f"{k}_mean"
        std_k = f"{k}_std"
        if mean_k in aggregate:
            print(f"  {k:20s}: {aggregate[mean_k]:.4f} ± {aggregate.get(std_k, 0):.4f}")
    print(f"  {'TP/FP/FN':20s}: {tp_total}/{fp_total}/{fn_total}")
    print(f"  {'n_pred/n_gt':20s}: {n_pred_total}/{n_gt_total}")
    
    print(f"\n--- CellSAM Paper Metrics ---")
    if paper_metrics.get("f1_mean") is not None:
        print(f"  {'F1':20s}: {paper_metrics['f1_mean']:.4f} ± {paper_metrics.get('f1_std', 0):.4f}")
        print(f"  {'Recall':20s}: {paper_metrics['recall_mean']:.4f} ± {paper_metrics.get('recall_std', 0):.4f}")
        print(f"  {'Precision':20s}: {paper_metrics['precision_mean']:.4f} ± {paper_metrics.get('precision_std', 0):.4f}")
        print(f"  {'AP@0.5':20s}: {paper_metrics.get('ap_mean', 0):.4f}")
    
    print(f"\n  Time: {t_elapsed:.1f}s ({t_elapsed/len(ids):.1f}s/sample)")
    
    # ── Save ─────────────────────────────────────────────
    results_path = os.path.join(output_dir, f"results_{diam_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {results_path}")
    
    per_sample_path = os.path.join(output_dir, f"per_sample_{diam_str}.json")
    with open(per_sample_path, "w") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {per_sample_path}")
    
    return results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="T31: Cellpose Paper-Aligned Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Main test run (default)
  python tools/cellpose_paper_aligned_eval.py

  # Supplementary diameter test
  python tools/cellpose_paper_aligned_eval.py --diameter 200

  # Val sweep for diameter tuning
  python tools/cellpose_paper_aligned_eval.py --split val --diameter 120 160 200 240
        """
    )
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--diameter", nargs="+", type=float, default=[None],
                        help="Diameter(s) to test. Default: None (auto)")
    parser.add_argument("--model-type", default="cyto3")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    
    # Handle diameter=None case
    diameters = args.diameter if args.diameter != [None] else [None]
    
    for d in diameters:
        run_cellpose_eval(
            split=args.split,
            diameter=d,
            model_type=args.model_type,
            gpu=not args.no_gpu,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
