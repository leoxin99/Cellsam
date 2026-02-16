#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E34b: DAPI edge/binuclear joint ablation on a fixed split (default val=71).

Search space (default):
  - edge_margin: [20, 32, 50]
  - size_ratio_threshold: [2.0, 2.5, 3.0, 3.5]
  - merge_coeff: [1.0, 1.2, 1.4, 1.5]

Protocol:
  - Fixed detector core params: min/max nucleus area + relative/fixed merge mode
  - Metric: micro Precision / Recall / F1 with IoU threshold 0.3
"""

import argparse
import json
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from detection.dapi import detect_nuclei, merge_close_nuclei, create_bounding_boxes


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def evaluate_combo(
    samples,
    min_nucleus_area: int,
    max_nucleus_area: int,
    use_relative_distance: bool,
    fixed_merge_distance: int,
    edge_margin: int,
    size_ratio_threshold: float,
    merge_coeff: float,
):
    all_tp, all_fp, all_fn = 0, 0, 0

    for sample in samples:
        regions = detect_nuclei(
            sample["dapi"],
            min_area=min_nucleus_area,
            max_area=max_nucleus_area,
        )
        groups = merge_close_nuclei(
            regions,
            size_ratio_threshold=size_ratio_threshold,
            use_relative_distance=use_relative_distance,
            fixed_merge_distance=fixed_merge_distance,
            merge_coeff=merge_coeff,
        )
        pred_boxes = create_bounding_boxes(
            groups,
            sample["image_shape"],
            margin=edge_margin,
        )

        gt_boxes = get_gt_boxes_from_mask(sample["mask"])
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.3)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(all_tp),
        "fp": int(all_fp),
        "fn": int(all_fn),
    }


def combo_key(edge_margin: int, size_ratio_threshold: float, merge_coeff: float):
    return f"{edge_margin}|{size_ratio_threshold:.4f}|{merge_coeff:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="E34b joint ablation for edge_margin/size_ratio_threshold/merge_coeff"
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--min-nucleus-area", type=int, default=1500)
    parser.add_argument("--max-nucleus-area", type=int, default=20000)
    parser.add_argument("--fixed-merge-distance", type=int, default=373)
    parser.add_argument("--use-relative-distance", dest="use_relative_distance", action="store_true")
    parser.add_argument("--use-fixed-distance", dest="use_relative_distance", action="store_false")
    parser.set_defaults(use_relative_distance=True)
    parser.add_argument("--edge-margins", type=str, default="20,32,50")
    parser.add_argument("--size-ratios", type=str, default="2.0,2.5,3.0,3.5")
    parser.add_argument("--merge-coeffs", type=str, default="1.0,1.2,1.4,1.5")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    edge_margins = parse_int_list(args.edge_margins)
    size_ratios = parse_float_list(args.size_ratios)
    merge_coeffs = parse_float_list(args.merge_coeffs)
    all_combos = list(itertools.product(edge_margins, size_ratios, merge_coeffs))

    print("=" * 72)
    print("E34b joint ablation (edge_margin, size_ratio_threshold, merge_coeff)")
    print("=" * 72)
    print(f"split={args.split}, combos={len(all_combos)}")
    print(
        "detector_core: min={}, max={}, mode={}, fixed_dist={}".format(
            args.min_nucleus_area,
            args.max_nucleus_area,
            "relative" if args.use_relative_distance else "fixed",
            args.fixed_merge_distance,
        )
    )

    data_dir = PROJECT_DIR / "data" / "processed"
    split_file = PROJECT_DIR / "data" / "splits" / f"{args.split}_ids.txt"
    output_dir = PROJECT_DIR / "experiments" / "ablation_detection_e34b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    n_samples = args.n_samples if args.n_samples > 0 else None
    samples = load_samples(data_dir, split_file, n_samples=n_samples)
    print(f"loaded_samples={len(samples)}")

    if args.resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"resume_from={output_file}")
    else:
        results = {
            "timestamp": datetime.now().isoformat(),
            "split": args.split,
            "n_samples": len(samples),
            "detector_core": {
                "min_nucleus_area": args.min_nucleus_area,
                "max_nucleus_area": args.max_nucleus_area,
                "use_relative_distance": args.use_relative_distance,
                "fixed_merge_distance": args.fixed_merge_distance,
            },
            "search_space": {
                "edge_margins": edge_margins,
                "size_ratios": size_ratios,
                "merge_coeffs": merge_coeffs,
            },
            "experiments": [],
        }

    seen = {
        combo_key(
            int(row["edge_margin"]),
            float(row["size_ratio_threshold"]),
            float(row["merge_coeff"]),
        )
        for row in results.get("experiments", [])
    }

    for edge_margin, size_ratio, merge_coeff in tqdm(all_combos, desc="E34b"):
        key = combo_key(edge_margin, size_ratio, merge_coeff)
        if key in seen:
            continue

        metrics = evaluate_combo(
            samples=samples,
            min_nucleus_area=args.min_nucleus_area,
            max_nucleus_area=args.max_nucleus_area,
            use_relative_distance=args.use_relative_distance,
            fixed_merge_distance=args.fixed_merge_distance,
            edge_margin=edge_margin,
            size_ratio_threshold=size_ratio,
            merge_coeff=merge_coeff,
        )
        row = {
            "edge_margin": edge_margin,
            "size_ratio_threshold": size_ratio,
            "merge_coeff": merge_coeff,
        }
        row.update(metrics)
        results["experiments"].append(row)

    sorted_rows = sorted(results["experiments"], key=lambda x: x["f1"], reverse=True)
    best = sorted_rows[0] if sorted_rows else None
    top2 = sorted_rows[:2] if len(sorted_rows) >= 2 else sorted_rows
    results["best"] = best
    results["top2"] = top2
    results["updated_at"] = datetime.now().isoformat()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 72)
    if best is None:
        print("no_result")
    else:
        print(
            "best: edge_margin={}, size_ratio_threshold={}, merge_coeff={}, F1={}, P={}, R={}".format(
                best["edge_margin"],
                best["size_ratio_threshold"],
                best["merge_coeff"],
                best["f1"],
                best["precision"],
                best["recall"],
            )
        )
    if len(top2) > 1:
        second = top2[1]
        print(
            "second: edge_margin={}, size_ratio_threshold={}, merge_coeff={}, F1={}".format(
                second["edge_margin"],
                second["size_ratio_threshold"],
                second["merge_coeff"],
                second["f1"],
            )
        )
    print(f"saved={output_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
