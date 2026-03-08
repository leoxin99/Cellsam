#!/usr/bin/env python3
"""
DAPI Detection Evaluation — E2E evaluation of T27a with DAPI-based detection.

Two detection methods:
  A. Nucleus detection: DAPI Otsu → nuclei → merge binucleated → bounding boxes
  B. Z-line adaptive: DAPI nuclei + Actn2 Z-line extent → adaptive boxes

For each method, runs T27a model segmentation and computes full metrics
(PQ, SQ, RQ, F1, Precision, Recall, AJI, BM-Dice) against GT masks.

Usage:
  conda activate cellsam
  python tools/eval_dapi_detection.py \
    --checkpoint checkpoints/T27a_PlanB_DecoderOnly_20260302_033621/best_model.pt \
    --splits val test \
    --output-dir experiments/t27a_dapi_eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from inference.core import segment_with_boxes, InferenceConfig
from metrics.instance_metrics import compute_all_metrics
from detection.dapi import detect_and_create_boxes, detect_with_adaptive_box
from detection.profiles import get_detection_profile
from cellSAM import get_model


# Load detection profiles (locked_eval is the validated parameter set)
_PROFILE = get_detection_profile("locked_eval")
_DAPI_PARAMS = _PROFILE["dapi"]
_ADAPTIVE_PARAMS = _PROFILE["adaptive"]


def get_raw_channels(sample_id: str) -> np.ndarray:
    """Load raw (un-normalized) image (3, H, W) for DAPI/Actn2 detection."""
    raw_path = PROJECT_ROOT / "data" / "processed" / "images" / f"{sample_id}.npy"
    raw = np.load(str(raw_path))
    if raw.ndim == 3 and raw.shape[0] == 3:
        return raw  # (3, H, W)
    return raw.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def to_uint8(ch: np.ndarray) -> np.ndarray:
    """Convert any numeric channel to uint8."""
    if ch.max() > 255:
        return ((ch.astype(np.float32) / ch.max()) * 255).astype(np.uint8)
    elif ch.max() <= 1.0 and ch.dtype in [np.float32, np.float64]:
        return (ch * 255).astype(np.uint8)
    return ch.astype(np.uint8)


def detect_dapi_boxes(raw_image: np.ndarray) -> list:
    """DAPI nucleus detection → boxes. raw_image: (3, H, W)"""
    dapi_u8 = to_uint8(raw_image[1])
    boxes, _, _ = detect_and_create_boxes(
        dapi_u8,
        min_nucleus_area=_DAPI_PARAMS["min_nucleus_area"],
        max_nucleus_area=_DAPI_PARAMS["max_nucleus_area"],
        size_ratio_threshold=_DAPI_PARAMS["size_ratio_threshold"],
        use_relative_distance=_DAPI_PARAMS["use_relative_distance"],
        fixed_merge_distance=_DAPI_PARAMS["fixed_merge_distance"],
        merge_coeff=_DAPI_PARAMS["merge_coeff"],
        margin=_DAPI_PARAMS["edge_margin"],
    )
    return boxes if boxes else []


def detect_zline_boxes(raw_image: np.ndarray) -> list:
    """DAPI + Actn2 Z-line adaptive detection → boxes. raw_image: (3, H, W)"""
    dapi_u8 = to_uint8(raw_image[1])
    actn2_u8 = to_uint8(raw_image[2])
    boxes, _, _ = detect_with_adaptive_box(
        dapi_channel=dapi_u8,
        actn2_channel=actn2_u8,
        min_nucleus_area=_ADAPTIVE_PARAMS["min_nucleus_area"],
        max_nucleus_area=_ADAPTIVE_PARAMS["max_nucleus_area"],
        search_radius=_ADAPTIVE_PARAMS["search_radius"],
        min_zlines=_ADAPTIVE_PARAMS["min_zlines"],
        zline_threshold=_ADAPTIVE_PARAMS["zline_threshold"],
        margin=_ADAPTIVE_PARAMS["edge_margin"],
        size_ratio_threshold=_ADAPTIVE_PARAMS["size_ratio_threshold"],
        use_relative_distance=_ADAPTIVE_PARAMS["use_relative_distance"],
        fixed_merge_distance=_ADAPTIVE_PARAMS["fixed_merge_distance"],
        merge_coeff=_ADAPTIVE_PARAMS["merge_coeff"],
    )
    return boxes if boxes else []


def run_inference(model, image_tensor, boxes, config, device):
    """Run segmentation with given boxes, return instance mask."""
    if len(boxes) == 0:
        H, W = image_tensor.shape[-2:]
        return np.zeros((H, W), dtype=np.int32)
    boxes_np = np.array(boxes, dtype=np.float32)
    boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)
    result = segment_with_boxes(
        model=model, image=image_tensor, boxes=boxes_tensor,
        config=config, device=device,
    )
    return result.instance_mask


def evaluate_with_detection(model, dataset, device, detection_fn, method_name,
                            infer_cfg, use_bf_only=True):
    """Evaluate model with auto-detected boxes vs GT masks."""
    per_image = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_detected = 0
    total_gt = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_id = sample['sample_id']
        gt_mask = sample['mask'].numpy()
        image = sample['image']  # (3, H, W) normalized

        # Load raw channels for DAPI/Actn2 detection
        raw_image = get_raw_channels(sample_id)

        # Auto-detect boxes
        detected_boxes = detection_fn(raw_image)

        # Prepare BF-only input for segmentation (as T27a was trained BF-only)
        if use_bf_only:
            bf = image[0]  # BF channel
            input_tensor = torch.stack([bf, bf, bf], dim=0).float()
        else:
            input_tensor = image.float()

        # Run segmentation with detected boxes
        pred_mask = run_inference(model, input_tensor, detected_boxes, infer_cfg, device)

        # Compute metrics against GT
        metrics = compute_all_metrics(pred_mask, gt_mask, iou_threshold=0.5)
        per_image.append(metrics)
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_detected += len(detected_boxes)
        total_gt += metrics['n_gt_cells']

        if (idx + 1) % 10 == 0:
            print(f"  [{method_name}] {idx+1}/{len(dataset)}")

    n = len(per_image)
    if n == 0:
        return {}

    # Per-image averages
    agg = {}
    for key in ['pq', 'sq', 'rq', 'bm_1to1_dice', 'bm_coverage_dice', 'aji', 'semantic_dice']:
        values = [m[key] for m in per_image]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    # Global detection metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    agg.update({
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp_total': total_tp, 'fp_total': total_fp, 'fn_total': total_fn,
        'total_detected_boxes': total_detected,
        'total_gt_cells': total_gt,
        'avg_detected_per_image': round(total_detected / n, 1),
        'avg_gt_per_image': round(total_gt / n, 1),
        'n_samples': n,
        'method': method_name,
    })
    return agg


def main():
    parser = argparse.ArgumentParser(description="DAPI detection evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", default="experiments/t27a_dapi_eval")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = get_model()
    model.adv_mode = True
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded on {device}")

    infer_cfg = InferenceConfig.default()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        split_ids = load_split_ids(split, str(PROJECT_ROOT / "data/splits"))
        if not split_ids:
            print(f"  Skip: no IDs for '{split}'")
            continue

        dataset = AugmentedAllenDataset(
            data_dir=str(PROJECT_ROOT / "data/processed"),
            sample_ids=split_ids,
            is_training=False,
            use_bf_only=False,  # Need all 3 channels for raw data access
        )
        print(f"  Dataset: {len(dataset)} images")

        split_results = {}

        # --- Default (clip ON) ---
        # Method A: DAPI nucleus detection
        print(f"\n  --- Method A: DAPI Nucleus Detection (clip=on) ---")
        t0 = time.time()
        res_a = evaluate_with_detection(
            model, dataset, device, detect_dapi_boxes,
            "DAPI_nucleus", infer_cfg, use_bf_only=True
        )
        res_a['elapsed_seconds'] = round(time.time() - t0, 1)
        split_results['dapi_nucleus'] = res_a
        print(f"\n  PQ={res_a['pq_mean']:.4f}, F1={res_a['f1']:.4f}, P={res_a['precision']:.4f}, R={res_a['recall']:.4f}")
        print(f"  BM-Dice={res_a['bm_1to1_dice_mean']:.4f}, AJI={res_a['aji_mean']:.4f}")
        print(f"  Detected/GT: {res_a['avg_detected_per_image']}/{res_a['avg_gt_per_image']} per image")
        print(f"  TP/FP/FN: {res_a['tp_total']}/{res_a['fp_total']}/{res_a['fn_total']}")

        # Method B: Z-line adaptive detection
        print(f"\n  --- Method B: Z-line Adaptive Detection (clip=on) ---")
        t0 = time.time()
        res_b = evaluate_with_detection(
            model, dataset, device, detect_zline_boxes,
            "zline_adaptive", infer_cfg, use_bf_only=True
        )
        res_b['elapsed_seconds'] = round(time.time() - t0, 1)
        split_results['zline_adaptive'] = res_b
        print(f"\n  PQ={res_b['pq_mean']:.4f}, F1={res_b['f1']:.4f}, P={res_b['precision']:.4f}, R={res_b['recall']:.4f}")
        print(f"  BM-Dice={res_b['bm_1to1_dice_mean']:.4f}, AJI={res_b['aji_mean']:.4f}")
        print(f"  Detected/GT: {res_b['avg_detected_per_image']}/{res_b['avg_gt_per_image']} per image")
        print(f"  TP/FP/FN: {res_b['tp_total']}/{res_b['fp_total']}/{res_b['fn_total']}")

        # --- Box clipping OFF ---
        infer_cfg_noclip = InferenceConfig.default()
        infer_cfg_noclip.apply_box_clipping = False

        # Method C: DAPI nucleus (no clip)
        print(f"\n  --- Method C: DAPI Nucleus Detection (clip=off) ---")
        t0 = time.time()
        res_c = evaluate_with_detection(
            model, dataset, device, detect_dapi_boxes,
            "DAPI_nucleus_noclip", infer_cfg_noclip, use_bf_only=True
        )
        res_c['elapsed_seconds'] = round(time.time() - t0, 1)
        split_results['dapi_nucleus_noclip'] = res_c
        print(f"\n  PQ={res_c['pq_mean']:.4f}, F1={res_c['f1']:.4f}, P={res_c['precision']:.4f}, R={res_c['recall']:.4f}")
        print(f"  BM-Dice={res_c['bm_1to1_dice_mean']:.4f}, AJI={res_c['aji_mean']:.4f}")

        # Method D: Z-line adaptive (no clip)
        print(f"\n  --- Method D: Z-line Adaptive Detection (clip=off) ---")
        t0 = time.time()
        res_d = evaluate_with_detection(
            model, dataset, device, detect_zline_boxes,
            "zline_adaptive_noclip", infer_cfg_noclip, use_bf_only=True
        )
        res_d['elapsed_seconds'] = round(time.time() - t0, 1)
        split_results['zline_adaptive_noclip'] = res_d
        print(f"\n  PQ={res_d['pq_mean']:.4f}, F1={res_d['f1']:.4f}, P={res_d['precision']:.4f}, R={res_d['recall']:.4f}")
        print(f"  BM-Dice={res_d['bm_1to1_dice_mean']:.4f}, AJI={res_d['aji_mean']:.4f}")

        results[split] = split_results

    results['metadata'] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'methods': ['dapi_nucleus', 'zline_adaptive', 'dapi_nucleus_noclip', 'zline_adaptive_noclip'],
        'splits': args.splits,
        'detection_profile': 'locked_eval',
        'script': 'tools/eval_dapi_detection.py',
    }

    out_file = output_dir / "results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
