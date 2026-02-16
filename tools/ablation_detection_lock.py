#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E34 test lock evaluation for detection (DAPI vs Adaptive).

Policy:
  - Fixed params from val lock / E34b.
  - Run on test split once, write a lock file.
  - No parameter tuning based on test output.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from detection.dapi import detect_nuclei, merge_close_nuclei, create_bounding_boxes, detect_with_adaptive_box


def load_samples(data_dir: Path, split_file: Path, n_samples: int = None):
    with open(split_file, "r", encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]
    if n_samples is not None:
        sample_ids = sample_ids[:n_samples]

    samples = []
    for sample_id in tqdm(sample_ids, desc="Loading samples"):
        image_path = data_dir / "images" / f"{sample_id}.npy"
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        if not image_path.exists() or not mask_path.exists():
            continue

        image = np.load(image_path)
        mask = np.load(mask_path)
        samples.append(
            {
                "id": sample_id,
                "dapi": image[1],
                "actn2": image[2],
                "mask": mask,
                "image_shape": image.shape[1:],
            }
        )
    return samples


def get_gt_boxes_from_mask(mask: np.ndarray):
    from skimage import measure

    boxes = []
    regions = measure.regionprops(mask)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        boxes.append((minr, minc, maxr, maxc))
    return boxes


def compute_box_iou(box1, box2):
    r1, c1, r2, c2 = box1
    r3, c3, r4, c4 = box2

    inter_r1 = max(r1, r3)
    inter_c1 = max(c1, c3)
    inter_r2 = min(r2, r4)
    inter_c2 = min(c2, c4)
    if inter_r2 <= inter_r1 or inter_c2 <= inter_c1:
        return 0.0

    inter_area = (inter_r2 - inter_r1) * (inter_c2 - inter_c1)
    area1 = (r2 - r1) * (c2 - c1)
    area2 = (r4 - r3) * (c4 - c3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold: float = 0.3):
    if not pred_boxes or not gt_boxes:
        return 0, len(pred_boxes), len(gt_boxes)

    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        x1, y1, x2, y2 = pred
        pred_bbox = (y1, x1, y2, x2)

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_box_iou(pred_bbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_dapi(samples, args):
    all_tp, all_fp, all_fn = 0, 0, 0
    for sample in tqdm(samples, desc="DAPI lock eval"):
        regions = detect_nuclei(
            sample["dapi"],
            min_area=args.dapi_min_nucleus_area,
            max_area=args.dapi_max_nucleus_area,
        )
        groups = merge_close_nuclei(
            regions,
            size_ratio_threshold=args.size_ratio_threshold,
            use_relative_distance=args.use_relative_distance,
            fixed_merge_distance=args.fixed_merge_distance,
            merge_coeff=args.merge_coeff,
        )
        pred_boxes = create_bounding_boxes(
            groups,
            sample["image_shape"],
            margin=args.edge_margin,
        )
        gt_boxes = get_gt_boxes_from_mask(sample["mask"])
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    return compute_metrics(all_tp, all_fp, all_fn)


def evaluate_adaptive(samples, args):
    all_tp, all_fp, all_fn = 0, 0, 0
    for sample in tqdm(samples, desc="Adaptive lock eval"):
        result = detect_with_adaptive_box(
            sample["dapi"],
            sample["actn2"],
            min_nucleus_area=args.adaptive_min_nucleus_area,
            max_nucleus_area=args.adaptive_max_nucleus_area,
            search_radius=args.adaptive_search_radius,
            min_zlines=args.adaptive_min_zlines,
            zline_threshold=args.adaptive_zline_threshold,
            margin=args.edge_margin,
            size_ratio_threshold=args.size_ratio_threshold,
            use_relative_distance=args.use_relative_distance,
            fixed_merge_distance=args.fixed_merge_distance,
            merge_coeff=args.merge_coeff,
        )
        pred_boxes = result[0] if isinstance(result, tuple) else result
        gt_boxes = get_gt_boxes_from_mask(sample["mask"])
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn
    return compute_metrics(all_tp, all_fp, all_fn)


def main():
    parser = argparse.ArgumentParser(description="E34 test lock detection evaluation")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_DIR / "experiments" / "ablation_detection_lock" / "results.json"),
    )

    parser.add_argument("--dapi-min-nucleus-area", type=int, default=1500)
    parser.add_argument("--dapi-max-nucleus-area", type=int, default=20000)
    parser.add_argument("--edge-margin", type=int, default=32)
    parser.add_argument("--size-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--merge-coeff", type=float, default=1.2)
    parser.add_argument("--fixed-merge-distance", type=int, default=373)
    parser.add_argument("--use-relative-distance", dest="use_relative_distance", action="store_true")
    parser.add_argument("--use-fixed-distance", dest="use_relative_distance", action="store_false")
    parser.set_defaults(use_relative_distance=True)

    parser.add_argument("--adaptive-min-nucleus-area", type=int, default=1500)
    parser.add_argument("--adaptive-max-nucleus-area", type=int, default=20000)
    parser.add_argument("--adaptive-search-radius", type=int, default=200)
    parser.add_argument("--adaptive-min-zlines", type=int, default=5)
    parser.add_argument("--adaptive-zline-threshold", type=float, default=0.01)
    args = parser.parse_args()

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.force:
        raise RuntimeError(
            "Lock file already exists. Use --force only when you intentionally overwrite."
        )

    print("=" * 72)
    print("E34 detection lock evaluation")
    print("=" * 72)
    print(f"split={args.split}, output={output_file}")

    data_dir = PROJECT_DIR / "data" / "processed"
    split_file = PROJECT_DIR / "data" / "splits" / f"{args.split}_ids.txt"
    n_samples = args.n_samples if args.n_samples > 0 else None
    samples = load_samples(data_dir, split_file, n_samples=n_samples)
    print(f"loaded_samples={len(samples)}")

    dapi_metrics = evaluate_dapi(samples, args)
    adaptive_metrics = evaluate_adaptive(samples, args)

    delta_f1 = round(adaptive_metrics["f1"] - dapi_metrics["f1"], 4)
    winner = "adaptive" if delta_f1 > 0 else "dapi"

    results = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "n_samples": len(samples),
        "policy": "single_run_lockdown_no_reverse_tuning",
        "dapi_params": {
            "min_nucleus_area": args.dapi_min_nucleus_area,
            "max_nucleus_area": args.dapi_max_nucleus_area,
            "edge_margin": args.edge_margin,
            "size_ratio_threshold": args.size_ratio_threshold,
            "merge_coeff": args.merge_coeff,
            "use_relative_distance": args.use_relative_distance,
            "fixed_merge_distance": args.fixed_merge_distance,
        },
        "adaptive_params": {
            "min_nucleus_area": args.adaptive_min_nucleus_area,
            "max_nucleus_area": args.adaptive_max_nucleus_area,
            "search_radius": args.adaptive_search_radius,
            "min_zlines": args.adaptive_min_zlines,
            "zline_threshold": args.adaptive_zline_threshold,
            "edge_margin": args.edge_margin,
            "size_ratio_threshold": args.size_ratio_threshold,
            "merge_coeff": args.merge_coeff,
            "use_relative_distance": args.use_relative_distance,
            "fixed_merge_distance": args.fixed_merge_distance,
        },
        "dapi": dapi_metrics,
        "adaptive": adaptive_metrics,
        "delta_f1": delta_f1,
        "winner": winner,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 72)
    print(
        "dapi:     F1={}, P={}, R={}, TP/FP/FN={}/{}/{}".format(
            dapi_metrics["f1"],
            dapi_metrics["precision"],
            dapi_metrics["recall"],
            dapi_metrics["tp"],
            dapi_metrics["fp"],
            dapi_metrics["fn"],
        )
    )
    print(
        "adaptive: F1={}, P={}, R={}, TP/FP/FN={}/{}/{}".format(
            adaptive_metrics["f1"],
            adaptive_metrics["precision"],
            adaptive_metrics["recall"],
            adaptive_metrics["tp"],
            adaptive_metrics["fp"],
            adaptive_metrics["fn"],
        )
    )
    print(f"winner={winner}, delta_f1={delta_f1}")
    print(f"saved={output_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
