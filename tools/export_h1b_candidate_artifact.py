#!/usr/bin/env python3
"""Export frozen H1bA candidate artifacts for CellFinder prior experiments."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from augmented_dataset import load_split_ids
from detection.h1b_priors import _detect_h1b_candidates_with_stats
from detection.profiles import get_detection_profile


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def match_boxes_to_gt(pred_boxes, gt_boxes, gt_cell_ids, iou_threshold=0.3):
    """Run Hungarian matching between predicted candidate boxes and GT boxes."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return [], []
    if n_gt == 0:
        return [-1] * n_pred, [0.0] * n_pred

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou_matrix[pred_index, gt_index] = compute_box_iou(pred_box, gt_box)

    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    matched_gt_ids = [-1] * n_pred
    match_ious = [0.0] * n_pred
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        iou = float(iou_matrix[pred_index, gt_index])
        if iou >= iou_threshold:
            matched_gt_ids[pred_index] = int(gt_cell_ids[gt_index])
            match_ious[pred_index] = iou

    return matched_gt_ids, match_ious


def extract_gt_boxes_from_mask(mask_np):
    """Extract GT boxes and instance ids from an instance mask."""
    boxes = []
    cell_ids = []
    for cell_id in np.unique(mask_np):
        if cell_id <= 0:
            continue
        ys, xs = np.where(mask_np == cell_id)
        if len(ys) == 0:
            continue
        boxes.append(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ]
        )
        cell_ids.append(int(cell_id))
    return boxes, cell_ids


def extract_gt_centroids_from_mask(mask_np):
    """Extract GT cell centroids from an instance mask."""
    centroids = {}
    for cell_id in np.unique(mask_np):
        if cell_id <= 0:
            continue
        ys, xs = np.where(mask_np == cell_id)
        if len(ys) == 0:
            continue
        centroids[int(cell_id)] = [float(xs.mean()), float(ys.mean())]
    return centroids


def evaluate_candidate_centers(candidates, gt_mask):
    """Evaluate candidate centers against GT masks without using box IoU."""
    gt_centroids = extract_gt_centroids_from_mask(gt_mask)
    candidate_inside_gt_cell_ids = []
    gt_best_distance = {}

    for candidate in candidates:
        center_x, center_y = candidate["center_xy"]
        x_index = min(max(int(round(center_x)), 0), gt_mask.shape[1] - 1)
        y_index = min(max(int(round(center_y)), 0), gt_mask.shape[0] - 1)
        gt_id = int(gt_mask[y_index, x_index])
        if gt_id <= 0:
            candidate_inside_gt_cell_ids.append(-1)
            continue

        candidate_inside_gt_cell_ids.append(gt_id)
        gt_center_x, gt_center_y = gt_centroids[gt_id]
        distance = float(
            ((center_x - gt_center_x) ** 2 + (center_y - gt_center_y) ** 2) ** 0.5
        )
        if gt_id not in gt_best_distance or distance < gt_best_distance[gt_id]:
            gt_best_distance[gt_id] = distance

    n_gt = len(gt_centroids)
    n_candidates = len(candidates)
    n_inside_any_gt = sum(1 for gt_id in candidate_inside_gt_cell_ids if gt_id != -1)
    n_unique_gt_hit = len(gt_best_distance)
    precision = n_unique_gt_hit / max(n_candidates, 1)
    recall = n_unique_gt_hit / max(n_gt, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_distance = sum(gt_best_distance.values()) / max(len(gt_best_distance), 1)

    return {
        "candidate_inside_gt_cell_ids": candidate_inside_gt_cell_ids,
        "n_candidate_inside_any_gt": int(n_inside_any_gt),
        "n_unique_gt_hit": int(n_unique_gt_hit),
        "center_precision": round(precision, 4),
        "center_recall": round(recall, 4),
        "center_f1": round(f1, 4),
        "candidate_inside_gt_rate": round(n_inside_any_gt / max(n_candidates, 1), 4),
        "avg_center_to_gt_centroid_distance_px": round(avg_distance, 3),
    }


def get_raw_channels(sample_id):
    """Load processed raw channels in (C, H, W) order."""
    raw_path = PROJECT_ROOT / "data" / "processed" / "images" / f"{sample_id}.npy"
    raw = np.load(str(raw_path))
    if raw.ndim == 3 and raw.shape[0] <= 8 and raw.shape[-1] > 8:
        return raw
    if raw.ndim == 3 and raw.shape[-1] <= 8 and raw.shape[0] > 8:
        return raw.transpose(2, 0, 1)
    if raw.ndim == 3 and raw.shape[0] in (3, 4, 5):
        return raw
    if raw.ndim == 3 and raw.shape[-1] in (3, 4, 5):
        return raw.transpose(2, 0, 1)
    raise ValueError(f"Unsupported raw image shape for sample {sample_id}: {raw.shape}")


def get_gt_mask(sample_id):
    """Load one GT instance mask."""
    mask_path = PROJECT_ROOT / "data" / "processed" / "masks" / f"{sample_id}.npy"
    return np.load(str(mask_path))


def _build_summary(counter):
    total_images = counter["total_images"]
    total_candidates = counter["total_candidates"]
    total_groups = counter["total_groups"]
    counts = counter["candidate_counts"]

    if counts:
        median_candidates = float(np.median(counts))
        p90_candidates = float(np.percentile(counts, 90))
        min_candidates = int(min(counts))
        max_candidates = int(max(counts))
    else:
        median_candidates = 0.0
        p90_candidates = 0.0
        min_candidates = 0
        max_candidates = 0

    return {
        "total_images": int(total_images),
        "total_candidates": int(total_candidates),
        "total_gt": int(counter["total_gt"]),
        "total_matched": int(counter["total_matched"]),
        "total_fp": int(counter["total_fp"]),
        "avg_candidates_per_image": round(total_candidates / max(total_images, 1), 2),
        "avg_gt_per_image": round(counter["total_gt"] / max(total_images, 1), 2),
        "avg_matched_per_image": round(counter["total_matched"] / max(total_images, 1), 2),
        "avg_fp_per_image": round(counter["total_fp"] / max(total_images, 1), 2),
        "avg_match_iou": round(counter["total_iou"] / max(counter["matched_iou_count"], 1), 4),
        "gt_match_recall": round(counter["total_matched"] / max(counter["total_gt"], 1), 4),
        "binuclear_ratio": round(counter["total_binuclear"] / max(total_candidates, 1), 4),
        "adaptive_ratio": round(counter["total_adaptive"] / max(total_candidates, 1), 4),
        "edge_filtered_ratio": round(counter["total_edge_filtered"] / max(total_groups, 1), 4),
        "candidate_count_stats": {
            "min": min_candidates,
            "max": max_candidates,
            "median": round(median_candidates, 2),
            "p90": round(p90_candidates, 2),
        },
        "center_only_metrics": {
            "candidate_inside_gt_rate": round(
                counter["total_center_inside_gt"] / max(total_candidates, 1),
                4,
            ),
            "one_to_one_center_precision": round(
                counter["total_center_unique_gt_hit"] / max(total_candidates, 1),
                4,
            ),
            "one_to_one_center_recall": round(
                counter["total_center_unique_gt_hit"] / max(counter["total_gt"], 1),
                4,
            ),
            "one_to_one_center_f1": round(
                (
                    2
                    * (counter["total_center_unique_gt_hit"] / max(total_candidates, 1))
                    * (counter["total_center_unique_gt_hit"] / max(counter["total_gt"], 1))
                    / max(
                        (
                            counter["total_center_unique_gt_hit"] / max(total_candidates, 1)
                            + counter["total_center_unique_gt_hit"] / max(counter["total_gt"], 1)
                        ),
                        1e-8,
                    )
                ),
                4,
            ),
            "avg_abs_count_error_per_image": round(
                counter["count_abs_error_sum"] / max(total_images, 1),
                3,
            ),
            "avg_matched_center_to_gt_centroid_distance_px": round(
                counter["matched_center_distance_sum"] / max(counter["matched_center_distance_n"], 1),
                3,
            ),
        },
    }


def _new_counter():
    return {
        "total_images": 0,
        "total_candidates": 0,
        "total_gt": 0,
        "total_matched": 0,
        "total_fp": 0,
        "total_iou": 0.0,
        "matched_iou_count": 0,
        "total_groups": 0,
        "total_edge_filtered": 0,
        "total_binuclear": 0,
        "total_adaptive": 0,
        "candidate_counts": [],
        "total_center_inside_gt": 0,
        "total_center_unique_gt_hit": 0,
        "count_abs_error_sum": 0,
        "matched_center_distance_sum": 0.0,
        "matched_center_distance_n": 0,
    }


def _update_counter(
    counter,
    candidates,
    stats,
    gt_boxes,
    matched_gt_ids,
    match_ious,
    center_metrics,
):
    n_matched = sum(1 for gt_id in matched_gt_ids if gt_id != -1)
    n_candidates = len(candidates)

    counter["total_images"] += 1
    counter["total_candidates"] += n_candidates
    counter["total_gt"] += len(gt_boxes)
    counter["total_matched"] += n_matched
    counter["total_fp"] += n_candidates - n_matched
    counter["total_groups"] += int(stats.get("n_groups", 0))
    counter["total_edge_filtered"] += int(stats.get("n_edge_filtered", 0))
    counter["total_binuclear"] += sum(candidate["is_binuclear"] for candidate in candidates)
    counter["total_adaptive"] += sum(candidate["adaptive"] for candidate in candidates)
    counter["candidate_counts"].append(n_candidates)
    counter["total_center_inside_gt"] += int(center_metrics["n_candidate_inside_any_gt"])
    counter["total_center_unique_gt_hit"] += int(center_metrics["n_unique_gt_hit"])
    counter["count_abs_error_sum"] += abs(n_candidates - len(gt_boxes))
    counter["matched_center_distance_sum"] += (
        float(center_metrics["avg_center_to_gt_centroid_distance_px"])
        * int(center_metrics["n_unique_gt_hit"])
    )
    counter["matched_center_distance_n"] += int(center_metrics["n_unique_gt_hit"])

    matched_iou_values = [iou for iou in match_ious if iou > 0]
    counter["total_iou"] += float(sum(matched_iou_values))
    counter["matched_iou_count"] += len(matched_iou_values)


def _resolve_output_path(output_path, split, multiple_splits):
    output_path = Path(output_path)
    if output_path.suffix and not output_path.is_dir():
        if multiple_splits:
            return output_path.with_name(f"{output_path.stem}_{split}{output_path.suffix}")
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"h1b_candidates_{split}.json"


def export_candidate_artifact(
    split,
    candidate_mode,
    profile_name,
    output_path,
    adaptive_box_mode="adaptive_zline",
    match_iou=0.3,
    limit=0,
):
    """Export one split worth of H1bA candidates to JSON."""
    sample_ids = load_split_ids(split, str(PROJECT_ROOT / "data" / "splits"))
    if limit > 0:
        sample_ids = sample_ids[:limit]

    result = {
        "metadata": {
            "artifact_type": "h1b_candidate_artifact",
            "candidate_mode": candidate_mode,
            "adaptive_box_mode": adaptive_box_mode,
            "detection_profile": profile_name,
            "profile_snapshot": get_detection_profile(profile_name),
            "match_iou_threshold": match_iou,
            "box_match_fields_purpose": "artifact_audit_only_not_runtime_input",
            "center_metrics_purpose": "candidate_identity_and_center_alignment_audit",
            "timestamp": datetime.now().isoformat(),
            "source_script": "tools/export_h1b_candidate_artifact.py",
            "split": split,
        },
        "images": {},
        "summary": {},
    }

    counter = _new_counter()
    print(f"\nProcessing split: {split} ({len(sample_ids)} images)")

    for index, sample_id in enumerate(sample_ids):
        raw_image = get_raw_channels(sample_id)
        gt_mask = get_gt_mask(sample_id)
        gt_boxes, gt_cell_ids = extract_gt_boxes_from_mask(gt_mask)
        candidates, stats = _detect_h1b_candidates_with_stats(
            raw_image=raw_image,
            profile_name=profile_name,
            candidate_mode=candidate_mode,
            adaptive_box_mode=adaptive_box_mode,
        )
        candidate_boxes = [candidate["box_xyxy"] for candidate in candidates]
        matched_gt_ids, match_ious = match_boxes_to_gt(
            candidate_boxes,
            gt_boxes,
            gt_cell_ids,
            iou_threshold=match_iou,
        )
        center_metrics = evaluate_candidate_centers(candidates, gt_mask)

        n_matched = sum(1 for gt_id in matched_gt_ids if gt_id != -1)
        result["images"][sample_id] = {
            "split": split,
            "image_shape": [int(raw_image.shape[-2]), int(raw_image.shape[-1])],
            "candidates": candidates,
            "candidate_stats": stats,
            "center_audit": center_metrics,
            "matched_gt_cell_ids": matched_gt_ids,
            "match_ious": match_ious,
            "n_gt": len(gt_boxes),
            "n_candidates": len(candidates),
            "n_matched": n_matched,
            "n_fp": len(candidates) - n_matched,
        }

        _update_counter(
            counter=counter,
            candidates=candidates,
            stats=stats,
            gt_boxes=gt_boxes,
            matched_gt_ids=matched_gt_ids,
            match_ious=match_ious,
            center_metrics=center_metrics,
        )

        if (index + 1) % 50 == 0:
            print(f"  [{split}] {index + 1}/{len(sample_ids)}")

    result["summary"] = _build_summary(counter)

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"Saved artifact to: {resolved_output_path}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return result["summary"]


def main():
    parser = argparse.ArgumentParser(
        description="Export H1bA candidate artifact for CellFinder prior experiments"
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["adaptive", "dapi_cm"],
        required=True,
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--profile", default="locked_eval")
    parser.add_argument(
        "--adaptive-box-mode",
        choices=["center_only", "adaptive_zline"],
        default="adaptive_zline",
        help=(
            "Only affects candidate_mode=adaptive. "
            "Use adaptive_zline to keep legacy z-line adaptive/fallback boxes for export/visual audit."
        ),
    )
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-split sample cap for smoke tests (0 = all)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    multiple_splits = len(args.splits) > 1
    summaries = {}
    for split in args.splits:
        split_output_path = _resolve_output_path(args.output, split, multiple_splits)
        summaries[split] = export_candidate_artifact(
            split=split,
            candidate_mode=args.candidate_mode,
            profile_name=args.profile,
            output_path=split_output_path,
            adaptive_box_mode=args.adaptive_box_mode,
            match_iou=args.match_iou,
            limit=args.limit,
        )

    if multiple_splits:
        print("\nPer-split summaries:")
        print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
