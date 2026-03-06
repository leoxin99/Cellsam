#!/usr/bin/env python3
"""T34: Official-Path vs Unified Evaluation Ablation

Three-arm comparison using the same T27a checkpoint + GT boxes:
  Arm A: Unified default  (box_clipping=True, conflict=argmax_prob, postprocess=True)
  Arm B: Unified no-clip  (box_clipping=False, conflict=argmax_prob, postprocess=True)
  Arm C: Official path    (CellSAM model.predict() + postprocess_predictions + fill_holes)

Usage:
  python tools/eval_t34_official_path.py --split val
  python tools/eval_t34_official_path.py --split test
"""
import sys, json, time, argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from inference.core import (
    load_cellsam_checkpoint, segment_with_boxes, InferenceConfig
)

# ── CellSAM official path imports ──
from cellSAM.model import postprocess_predictions
from cellSAM.utils import fill_holes_and_remove_small_masks, subtract_boundaries

# ── Metrics ──
from metrics.instance_metrics import compute_all_metrics



def load_data(split: str):
    """Load split IDs, images, masks, and extract GT boxes."""
    from skimage.measure import regionprops
    
    splits_dir = PROJECT_ROOT / "data" / "splits"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    ids = [l.strip() for l in open(splits_dir / f"{split}_ids.txt") if l.strip()]
    
    samples = []
    for img_id in ids:
        img = np.load(processed_dir / "images" / f"{img_id}.npy")
        mask = np.load(processed_dir / "masks" / f"{img_id}.npy")
        
        # Extract GT boxes
        boxes = []
        for prop in regionprops(mask.astype(int)):
            y1, x1, y2, x2 = prop.bbox
            boxes.append([x1, y1, x2, y2])
        
        samples.append({
            'id': img_id,
            'image': img,
            'mask': mask,
            'boxes': boxes
        })
    
    return samples


def arm_a_unified_default(model, samples, device):
    """Arm A: Unified default (box_clipping=True, argmax_prob, postprocess=True)"""
    config = InferenceConfig.default()
    # Defaults: apply_box_clipping=True, conflict_policy='argmax_prob', apply_postprocess=True
    
    preds = []
    for s in tqdm(samples, desc="Arm A (unified default)"):
        if len(s['boxes']) == 0:
            preds.append(np.zeros_like(s['mask'], dtype=np.int32))
            continue
        
        img_tensor = torch.from_numpy(s['image']).float()
        boxes_tensor = torch.tensor(s['boxes'], dtype=torch.float32)
        
        result = segment_with_boxes(model, img_tensor, boxes_tensor, config, device=device)
        preds.append(result.instance_mask)
    
    return preds


def arm_b_unified_noclip(model, samples, device):
    """Arm B: Unified no-clip (box_clipping=False, argmax_prob, postprocess=True)"""
    config = InferenceConfig.default()
    config.apply_box_clipping = False
    
    preds = []
    for s in tqdm(samples, desc="Arm B (unified no-clip)"):
        if len(s['boxes']) == 0:
            preds.append(np.zeros_like(s['mask'], dtype=np.int32))
            continue
        
        img_tensor = torch.from_numpy(s['image']).float()
        boxes_tensor = torch.tensor(s['boxes'], dtype=torch.float32)
        
        result = segment_with_boxes(model, img_tensor, boxes_tensor, config, device=device)
        preds.append(result.instance_mask)
    
    return preds


def arm_c_official(model, samples, device, use_postprocess=True):
    """Arm C: Official CellSAM path using model.predict()"""
    import torch.nn.functional as F
    from official_preprocess import official_preprocess_and_encode
    
    preds = []
    for s in tqdm(samples, desc=f"Arm C (official, pp={use_postprocess})"):
        if len(s['boxes']) == 0:
            preds.append(np.zeros_like(s['mask'], dtype=np.int32))
            continue
        
        img = s['image']  # (C, H, W) or (H, W, C)
        boxes = s['boxes']
        
        if img.ndim == 3 and img.shape[0] in (3, 4, 5):
            C, H, W = img.shape
        else:
            H, W = img.shape[:2]
            img = img.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
            C = img.shape[0]
        
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Official preprocess + encode
            model.adv_mode = True
            image_embedding = official_preprocess_and_encode(model, img_tensor, device)
            
            # Per-box: prompt encode + mask decode + threshold
            all_masks = np.zeros((H, W), dtype=np.int32)
            
            for idx, box in enumerate(boxes):
                box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
                
                sparse_emb, dense_emb = model.model_cp.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                
                low_res_masks, iou_pred = model.model_cp.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model_cp.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Upsample to original size
                upscaled = F.interpolate(
                    low_res_masks, size=(H, W),
                    mode="bilinear", align_corners=False
                )
                
                # Official path: threshold then assign instance ID
                binary_mask = (torch.sigmoid(upscaled[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
                
                # Official: np.max aggregation (last-write-wins for overlaps)
                instance_mask = binary_mask * (idx + 1)
                all_masks = np.maximum(all_masks, instance_mask)
        
        # Official postprocessing
        if use_postprocess:
            try:
                all_masks = postprocess_predictions(all_masks)
            except Exception:
                pass  # Skip if postprocess fails (e.g., no cells)
        
        # fill_holes_and_remove_small_masks (always in official path)
        try:
            all_masks = fill_holes_and_remove_small_masks(all_masks, min_size=25)
        except Exception:
            pass
        
        preds.append(all_masks.astype(np.int32))
    
    return preds


def evaluate_arm(preds, samples, arm_name):
    """Compute metrics for an arm's predictions."""
    metrics_list = []
    tp_total, fp_total, fn_total = 0, 0, 0
    n_pred_total, n_gt_total = 0, 0
    
    for pred, s in zip(preds, samples):
        gt = s['mask']
        m = compute_all_metrics(pred, gt)
        metrics_list.append(m)
        
        tp_total += m.get('tp', 0)
        fp_total += m.get('fp', 0)
        fn_total += m.get('fn', 0)
        n_pred = len(np.unique(pred)) - (1 if 0 in pred else 0)
        n_gt = len(np.unique(gt)) - (1 if 0 in gt else 0)
        n_pred_total += n_pred
        n_gt_total += n_gt
    
    # Aggregate
    keys = ['pq', 'sq', 'rq', 'bm_1to1_dice', 'bm_coverage_dice', 'aji', 'semantic_dice']
    result = {}
    for k in keys:
        vals = [m.get(k, 0) for m in metrics_list]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    
    result['tp_total'] = int(tp_total)
    result['fp_total'] = int(fp_total)
    result['fn_total'] = int(fn_total)
    result['n_pred_total'] = int(n_pred_total)
    result['n_gt_total'] = int(n_gt_total)
    result['n_samples'] = len(samples)
    result['arm'] = arm_name
    
    return result


def main():
    parser = argparse.ArgumentParser(description="T34: Official path ablation")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint", default=str(
        PROJECT_ROOT / "checkpoints" / "T27a_PlanB_DecoderOnly_20260302_033621" / "best_model.pt"
    ))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "t34_official_path_ablation"))
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("T34: Official-Path Evaluation Ablation")
    print("=" * 60)
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, adapter, ckpt_info = load_cellsam_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"  Checkpoint info: {ckpt_info}")
    
    # Load data
    print(f"\nLoading {args.split} data...")
    samples = load_data(args.split)
    print(f"  {len(samples)} samples loaded")
    
    # Run all three arms
    all_results = {}
    
    # Arm A
    t0 = time.time()
    preds_a = arm_a_unified_default(model, samples, device)
    elapsed_a = time.time() - t0
    results_a = evaluate_arm(preds_a, samples, "A_unified_default")
    results_a['elapsed_seconds'] = round(elapsed_a, 1)
    all_results['arm_a'] = results_a
    print(f"\n  Arm A: PQ={results_a['pq_mean']:.4f}, BM-Dice={results_a['bm_1to1_dice_mean']:.4f} ({elapsed_a:.1f}s)")
    
    # Arm B
    t0 = time.time()
    preds_b = arm_b_unified_noclip(model, samples, device)
    elapsed_b = time.time() - t0
    results_b = evaluate_arm(preds_b, samples, "B_unified_noclip")
    results_b['elapsed_seconds'] = round(elapsed_b, 1)
    all_results['arm_b'] = results_b
    print(f"  Arm B: PQ={results_b['pq_mean']:.4f}, BM-Dice={results_b['bm_1to1_dice_mean']:.4f} ({elapsed_b:.1f}s)")
    
    # Arm C
    t0 = time.time()
    preds_c = arm_c_official(model, samples, device, use_postprocess=True)
    elapsed_c = time.time() - t0
    results_c = evaluate_arm(preds_c, samples, "C_official")
    results_c['elapsed_seconds'] = round(elapsed_c, 1)
    all_results['arm_c'] = results_c
    print(f"  Arm C: PQ={results_c['pq_mean']:.4f}, BM-Dice={results_c['bm_1to1_dice_mean']:.4f} ({elapsed_c:.1f}s)")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Arm':<25} {'PQ':>8} {'BM-Dice':>8} {'AJI':>8} {'Sem-Dice':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 85)
    for key, label in [('arm_a', 'A: Unified default'), ('arm_b', 'B: Unified no-clip'), ('arm_c', 'C: Official path')]:
        r = all_results[key]
        print(f"{label:<25} {r['pq_mean']:>8.4f} {r['bm_1to1_dice_mean']:>8.4f} {r['aji_mean']:>8.4f} {r['semantic_dice_mean']:>8.4f} {r['tp_total']:>6} {r['fp_total']:>6} {r['fn_total']:>6}")
    
    # Save results
    out_file = output_dir / f"results_{args.split}.json"
    all_results['metadata'] = {
        'experiment': 'T34_official_path_ablation',
        'split': args.split,
        'checkpoint': args.checkpoint,
        'n_samples': len(samples),
        'script': 'tools/eval_t34_official_path.py'
    }
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
