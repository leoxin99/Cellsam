#!/usr/bin/env python3
"""Formal detector-level evaluation for H1bA recall-recovery variants.

Evaluates the adaptive-source H1bA recall-recovery variants requested for
the current iteration:
- raw CellFinder baseline
- H1bA strict with fixed thresholds 0.30 / 0.28 / 0.25
- H1bA candidate-aligned nodrop
- H1bA hybrid-open with fixed thresholds 0.30 / 0.28 / 0.25

Also exports a small sample audit on 4 fixed test images covering:
- strict_fixed0.30
- masked hybrid_fixed0.30
- hybrid_open_fixed0.30
- candidate_aligned_nodrop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import load_split_ids
from cellSAM import get_model
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates


PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
PROCESSED_MASKS_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
T33C_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T33c_CellFinder_NoES_seed123"
    / "best_cellfinder.pt"
)
DEFAULT_AUDIT_SAMPLE_IDS = [
    "5bf3e826_5500000014_63X_20190816_S1_P12_B3",
    "06433a48_5500000014_63X_20190816_S1_P16_B5",
    "bcd6a7c4_5500000013_63X_20190807_S1_P23_B4",
    "f573852d_5500000014_63X_20190816_S1_P29_B5",
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    prior_mode: str | None
    score_filter_mode: str | None
    score_threshold: float | None
    apply_candidate_mask: bool
    query_output_mode: str
    use_priors: bool


DETECTOR_VARIANTS = [
    VariantConfig(
        name="raw_cellfinder",
        prior_mode=None,
        score_filter_mode="dynamic",
        score_threshold=None,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=False,
    ),
    VariantConfig(
        name="h1ba_adaptive_strict_fixed0.30",
        prior_mode="strict",
        score_filter_mode="fixed",
        score_threshold=0.30,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_strict_fixed0.28",
        prior_mode="strict",
        score_filter_mode="fixed",
        score_threshold=0.28,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_strict_fixed0.25",
        prior_mode="strict",
        score_filter_mode="fixed",
        score_threshold=0.25,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_candidate_aligned_nodrop",
        prior_mode="strict",
        score_filter_mode=None,
        score_threshold=None,
        apply_candidate_mask=True,
        query_output_mode="candidate_aligned",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_hybrid_open_fixed0.30",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.30,
        apply_candidate_mask=False,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_hybrid_open_fixed0.28",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.28,
        apply_candidate_mask=False,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="h1ba_adaptive_hybrid_open_fixed0.25",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.25,
        apply_candidate_mask=False,
        query_output_mode="filtered",
        use_priors=True,
    ),
]

AUDIT_VARIANTS = [
    VariantConfig(
        name="strict_fixed0.30",
        prior_mode="strict",
        score_filter_mode="fixed",
        score_threshold=0.30,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="masked_hybrid_fixed0.30",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.30,
        apply_candidate_mask=True,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="hybrid_open_fixed0.30",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.30,
        apply_candidate_mask=False,
        query_output_mode="filtered",
        use_priors=True,
    ),
    VariantConfig(
        name="candidate_aligned_nodrop",
        prior_mode="strict",
        score_filter_mode=None,
        score_threshold=None,
        apply_candidate_mask=True,
        query_output_mode="candidate_aligned",
        use_priors=True,
    ),
]


def to_chw(raw_image: np.ndarray) -> np.ndarray:
    if raw_image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={raw_image.shape}")
    if raw_image.shape[0] in (3, 4, 5):
        return raw_image
    if raw_image.shape[-1] in (3, 4, 5):
        return raw_image.transpose(2, 0, 1)
    raise ValueError(f"Could not infer channel axis for shape={raw_image.shape}")


def load_raw_image(sample_id: str) -> np.ndarray:
    raw = np.load(PROCESSED_IMAGES_DIR / f"{sample_id}.npy")
    return to_chw(raw)


def load_mask(sample_id: str) -> np.ndarray:
    return np.load(PROCESSED_MASKS_DIR / f"{sample_id}.npy")


def extract_gt_boxes_from_mask(mask_np: np.ndarray):
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


def compute_box_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return float(inter / (union + 1e-6))


def match_boxes_to_gt(pred_boxes, gt_boxes, gt_cell_ids, iou_threshold):
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return [], []
    if n_gt == 0:
        return [-1] * n_pred, [0.0] * n_pred

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou_matrix[pred_idx, gt_idx] = compute_box_iou(pred_box, gt_box)

    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    matched_gt_ids = [-1] * n_pred
    match_ious = [0.0] * n_pred
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = float(iou_matrix[pred_idx, gt_idx])
        if iou >= iou_threshold:
            matched_gt_ids[pred_idx] = int(gt_cell_ids[gt_idx])
            match_ious[pred_idx] = iou
    return matched_gt_ids, match_ious


def summarize_scores(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def tensor_boxes_to_numpy(boxes) -> np.ndarray:
    if isinstance(boxes, torch.Tensor):
        if boxes.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.detach().cpu().numpy().astype(np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def filter_postprocessed_result(model, result, score_filter_mode, score_threshold):
    boxes = result["boxes"]
    scores = result["scores"]
    filtered = model._filter_boxes_by_scores(
        [boxes],
        [scores],
        score_filter_mode=score_filter_mode,
        score_threshold=score_threshold,
    )[0]
    return tensor_boxes_to_numpy(filtered)


def setup_model(device: str, checkpoint_path: Path, num_queries: int = 50):
    model = get_model()
    model.adv_mode = True
    model = model.to(device)
    model.eval()

    cellfinder = model.cellfinder
    original_nq = int(cellfinder.args.num_query_position)
    if num_queries != original_nq:
        old_state = cellfinder.state_dict()
        cellfinder.args.num_query_position = num_queries

        from cellSAM.AnchorDETR.models.anchor_detr import AnchorDETR, PostProcess
        from cellSAM.AnchorDETR.models.backbone import SAMBackbone
        from cellSAM.AnchorDETR.models.transformer import build_transformer

        backbone = SAMBackbone(
            "SAM",
            train_backbone=False,
            return_interm_layers=False,
            dilation=False,
            only_neck=False,
            freeze_backbone=False,
            sam_vit="vit_b",
        )
        transformer = build_transformer(cellfinder.args)
        cellfinder.decode_head = AnchorDETR(
            backbone,
            transformer,
            num_feature_levels=cellfinder.args.num_feature_levels,
            aux_loss=True,
        )
        cellfinder.postprocessors = {"bbox": PostProcess()}
        new_state = cellfinder.state_dict()
        compatible = {
            key: value
            for key, value in old_state.items()
            if key in new_state and value.shape == new_state[key].shape
        }
        cellfinder.load_state_dict(compatible, strict=False)
        model.cellfinder = cellfinder.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "cellfinder_state_dict" in checkpoint:
        state_dict = checkpoint["cellfinder_state_dict"]
    else:
        state_dict = checkpoint
    model.cellfinder.load_state_dict(state_dict, strict=False)
    model.cellfinder.eval()
    return model


def build_adaptive_prior_payload(model, raw_image: np.ndarray):
    candidates = detect_h1b_candidates(
        raw_image,
        profile_name="locked_eval",
        candidate_mode="adaptive",
    )
    points, valid_mask, _ = candidates_to_query_priors(
        candidates=candidates,
        image_shape=raw_image.shape[-2:],
        max_queries=int(model.cellfinder.args.num_query_position),
    )
    return {
        "candidates": candidates,
        "points": points,
        "valid_mask": valid_mask,
    }


@torch.no_grad()
def run_detector_pass(
    model,
    raw_image: np.ndarray,
    *,
    prior_payload=None,
    prior_mode=None,
    apply_candidate_mask=True,
):
    image_tensor = torch.from_numpy(raw_image).float()
    transformed = model.sam_bbox_preprocessing([image_tensor], percentile=False)

    if prior_payload is None:
        candidate_points = None
        candidate_valid_mask = None
    else:
        candidate_points, candidate_valid_mask = model._prepare_candidate_prior_batch(
            candidate_points_per_image=[prior_payload["points"]],
            candidate_valid_masks=[prior_payload["valid_mask"]],
            batch_size=1,
            max_queries=model.cellfinder.args.num_query_position,
            device=transformed.device,
        )

    raw_outputs = model.cellfinder.forward_inference(
        transformed,
        candidate_points=candidate_points,
        candidate_valid_mask=candidate_valid_mask,
        prior_mode=prior_mode,
        apply_candidate_mask=apply_candidate_mask,
        return_raw_outputs=True,
    )
    target_sizes = model._target_sizes_from_batch(transformed)
    postprocessed = model.cellfinder.postprocessors["bbox"](raw_outputs, target_sizes)[0]
    query_result = model._raw_outputs_to_query_results(raw_outputs, target_sizes)[0]
    return postprocessed, query_result


def init_variant_aggregate(variant_name: str):
    return {
        "variant": variant_name,
        "n_images": 0,
        "total_gt": 0,
        "total_pred_boxes": 0,
        "total_candidates": 0,
        "n_candidate_images": 0,
        "count_abs_error_sum": 0.0,
        "score_collections": {
            "candidate_query_scores": [],
            "fallback_query_scores": [],
        },
        "metrics": {
            "0.3": {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0, "matched_iou_n": 0},
            "0.5": {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0, "matched_iou_n": 0},
        },
    }


def update_variant_aggregate(aggregate, pred_boxes, gt_boxes, gt_cell_ids, candidate_count, query_result):
    pred_boxes = tensor_boxes_to_numpy(pred_boxes)

    aggregate["n_images"] += 1
    aggregate["total_gt"] += len(gt_boxes)
    aggregate["total_pred_boxes"] += len(pred_boxes)
    aggregate["count_abs_error_sum"] += abs(len(pred_boxes) - len(gt_boxes))

    if candidate_count is not None:
        aggregate["total_candidates"] += int(candidate_count)
        aggregate["n_candidate_images"] += 1

    valid_mask = query_result.get("effective_candidate_valid_mask")
    if valid_mask is not None:
        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        scores_np = query_result["scores"].detach().cpu().numpy()
        aggregate["score_collections"]["candidate_query_scores"].extend(
            scores_np[valid_mask_np].tolist()
        )
        aggregate["score_collections"]["fallback_query_scores"].extend(
            scores_np[~valid_mask_np].tolist()
        )

    for iou_threshold in (0.3, 0.5):
        matched_gt_ids, match_ious = match_boxes_to_gt(
            pred_boxes,
            gt_boxes,
            gt_cell_ids,
            iou_threshold=iou_threshold,
        )
        tp = sum(1 for gt_id in matched_gt_ids if gt_id != -1)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        bucket = aggregate["metrics"][f"{iou_threshold:.1f}"]
        bucket["tp"] += tp
        bucket["fp"] += fp
        bucket["fn"] += fn
        matched_ious = [iou for iou in match_ious if iou > 0]
        bucket["matched_iou_sum"] += float(sum(matched_ious))
        bucket["matched_iou_n"] += len(matched_ious)


def finalize_variant_aggregate(aggregate):
    def _metric_summary(bucket):
        tp = bucket["tp"]
        fp = bucket["fp"]
        fn = bucket["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        avg_iou = bucket["matched_iou_sum"] / max(bucket["matched_iou_n"], 1)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "avg_matched_box_iou": round(avg_iou, 4),
        }

    n_images = aggregate["n_images"]
    avg_candidates = None
    if aggregate["n_candidate_images"] > 0:
        avg_candidates = round(
            aggregate["total_candidates"] / max(aggregate["n_candidate_images"], 1),
            3,
        )

    summary = {
        "variant": aggregate["variant"],
        "n_images": int(n_images),
        "avg_candidates_per_image": avg_candidates,
        "avg_pred_boxes_per_image": round(aggregate["total_pred_boxes"] / max(n_images, 1), 3),
        "avg_gt_boxes_per_image": round(aggregate["total_gt"] / max(n_images, 1), 3),
        "avg_abs_count_error_per_image": round(
            aggregate["count_abs_error_sum"] / max(n_images, 1),
            3,
        ),
        "metrics_at_0.3": _metric_summary(aggregate["metrics"]["0.3"]),
        "metrics_at_0.5": _metric_summary(aggregate["metrics"]["0.5"]),
        "avg_matched_box_iou": _metric_summary(aggregate["metrics"]["0.3"])["avg_matched_box_iou"],
        "query_score_quantiles": {
            "candidate_query_scores": summarize_scores(
                aggregate["score_collections"]["candidate_query_scores"]
            ),
            "fallback_query_scores": summarize_scores(
                aggregate["score_collections"]["fallback_query_scores"]
            ),
        },
    }
    return summary


def build_audit_variant_entry(postprocessed, query_result, model, variant, candidate_count):
    pred_boxes = tensor_boxes_to_numpy(
        filter_postprocessed_result(
            model,
            postprocessed,
            variant.score_filter_mode,
            variant.score_threshold,
        )
        if variant.query_output_mode == "filtered"
        else query_result["boxes"][query_result["effective_candidate_valid_mask"]]
    )

    entry = {
        "pred_boxes": int(len(pred_boxes)),
        "candidate_count": int(candidate_count),
    }

    scores = query_result["scores"].detach().cpu().numpy()
    valid_mask = query_result.get("effective_candidate_valid_mask")
    if valid_mask is not None:
        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        candidate_scores = scores[valid_mask_np]
        fallback_scores = scores[~valid_mask_np]
        entry["candidate_query_scores"] = summarize_scores(candidate_scores)
        entry["candidate_query_scores_top6"] = [
            round(float(v), 4) for v in sorted(candidate_scores.tolist(), reverse=True)[:6]
        ]
        entry["candidate_queries_gt_0.30"] = int(np.sum(candidate_scores > 0.30))
        entry["fallback_query_scores"] = summarize_scores(fallback_scores)
        entry["fallback_query_scores_top6"] = [
            round(float(v), 4) for v in sorted(fallback_scores.tolist(), reverse=True)[:6]
        ]
        entry["fallback_queries_gt_0.30"] = int(np.sum(fallback_scores > 0.30))
    else:
        entry["query_scores"] = summarize_scores(scores)
        entry["query_scores_top6"] = [
            round(float(v), 4) for v in sorted(scores.tolist(), reverse=True)[:6]
        ]
    return entry


def evaluate_split(model, split: str, variant_configs):
    sample_ids = load_split_ids(split, str(PROJECT_ROOT / "data" / "splits"))
    per_variant = {variant.name: init_variant_aggregate(variant.name) for variant in variant_configs}

    for sample_id in tqdm(sample_ids, desc=f"Eval {split}"):
        raw_image = load_raw_image(sample_id)
        gt_mask = load_mask(sample_id)
        gt_boxes, gt_cell_ids = extract_gt_boxes_from_mask(gt_mask)
        prior_payload = build_adaptive_prior_payload(model, raw_image)

        raw_post, raw_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=None,
            prior_mode=None,
            apply_candidate_mask=True,
        )
        strict_post, strict_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=prior_payload,
            prior_mode="strict",
            apply_candidate_mask=True,
        )
        hybrid_open_post, hybrid_open_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=prior_payload,
            prior_mode="hybrid",
            apply_candidate_mask=False,
        )

        mode_cache = {
            "raw": (raw_post, raw_query, None),
            "strict": (strict_post, strict_query, len(prior_payload["candidates"])),
            "hybrid_open": (
                hybrid_open_post,
                hybrid_open_query,
                len(prior_payload["candidates"]),
            ),
        }

        for variant in variant_configs:
            if variant.name == "raw_cellfinder":
                postprocessed, query_result, candidate_count = mode_cache["raw"]
                pred_boxes = filter_postprocessed_result(
                    model,
                    postprocessed,
                    variant.score_filter_mode,
                    variant.score_threshold,
                )
            elif variant.query_output_mode == "candidate_aligned":
                postprocessed, query_result, candidate_count = mode_cache["strict"]
                pred_boxes = tensor_boxes_to_numpy(
                    query_result["boxes"][query_result["effective_candidate_valid_mask"]]
                )
            elif variant.prior_mode == "strict":
                postprocessed, query_result, candidate_count = mode_cache["strict"]
                pred_boxes = filter_postprocessed_result(
                    model,
                    postprocessed,
                    variant.score_filter_mode,
                    variant.score_threshold,
                )
            elif variant.prior_mode == "hybrid" and not variant.apply_candidate_mask:
                postprocessed, query_result, candidate_count = mode_cache["hybrid_open"]
                pred_boxes = filter_postprocessed_result(
                    model,
                    postprocessed,
                    variant.score_filter_mode,
                    variant.score_threshold,
                )
            else:
                raise ValueError(f"Unsupported variant routing: {variant}")

            update_variant_aggregate(
                per_variant[variant.name],
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                gt_cell_ids=gt_cell_ids,
                candidate_count=candidate_count,
                query_result=query_result,
            )

    return {
        variant_name: finalize_variant_aggregate(aggregate)
        for variant_name, aggregate in per_variant.items()
    }


def build_sample_audit(model, sample_ids):
    audit = {}
    for sample_id in sample_ids:
        raw_image = load_raw_image(sample_id)
        gt_mask = load_mask(sample_id)
        gt_boxes, _ = extract_gt_boxes_from_mask(gt_mask)
        prior_payload = build_adaptive_prior_payload(model, raw_image)

        strict_post, strict_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=prior_payload,
            prior_mode="strict",
            apply_candidate_mask=True,
        )
        masked_hybrid_post, masked_hybrid_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=prior_payload,
            prior_mode="hybrid",
            apply_candidate_mask=True,
        )
        hybrid_open_post, hybrid_open_query = run_detector_pass(
            model,
            raw_image,
            prior_payload=prior_payload,
            prior_mode="hybrid",
            apply_candidate_mask=False,
        )

        variant_entries = {}
        for variant in AUDIT_VARIANTS:
            if variant.name == "strict_fixed0.30":
                variant_entries[variant.name] = build_audit_variant_entry(
                    strict_post,
                    strict_query,
                    model,
                    variant,
                    len(prior_payload["candidates"]),
                )
            elif variant.name == "masked_hybrid_fixed0.30":
                variant_entries[variant.name] = build_audit_variant_entry(
                    masked_hybrid_post,
                    masked_hybrid_query,
                    model,
                    variant,
                    len(prior_payload["candidates"]),
                )
            elif variant.name == "hybrid_open_fixed0.30":
                variant_entries[variant.name] = build_audit_variant_entry(
                    hybrid_open_post,
                    hybrid_open_query,
                    model,
                    variant,
                    len(prior_payload["candidates"]),
                )
            elif variant.name == "candidate_aligned_nodrop":
                variant_entries[variant.name] = build_audit_variant_entry(
                    strict_post,
                    strict_query,
                    model,
                    variant,
                    len(prior_payload["candidates"]),
                )
            else:
                raise ValueError(f"Unsupported audit variant: {variant.name}")

        audit[sample_id] = {
            "split": "test",
            "gt_boxes": int(len(gt_boxes)),
            "candidate_count": int(len(prior_payload["candidates"])),
            "variants": variant_entries,
        }
    return audit


def main():
    parser = argparse.ArgumentParser(description="Evaluate H1bA recall-recovery detector variants")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=T33C_CKPT)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "h1ba_recall_recovery_detector_eval_t33c.json",
    )
    parser.add_argument(
        "--sample-audit-output",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "h1ba_recall_recovery_sample_audit_t33c.json",
    )
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_AUDIT_SAMPLE_IDS)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Splits: {args.splits}")

    model = setup_model(device=device, checkpoint_path=args.checkpoint, num_queries=args.num_queries)

    started_at = time.time()
    results = {}
    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        split_t0 = time.time()
        results[split] = evaluate_split(model, split, DETECTOR_VARIANTS)
        results[split]["elapsed_seconds"] = round(time.time() - split_t0, 1)

    sample_audit = build_sample_audit(model, args.sample_ids)

    metadata = {
        "script": "tools/eval_h1ba_recall_recovery.py",
        "checkpoint": str(args.checkpoint.resolve()),
        "num_queries": int(args.num_queries),
        "candidate_source": "adaptive",
        "detection_profile": "locked_eval",
        "splits": args.splits,
        "sample_audit_ids": args.sample_ids,
        "elapsed_seconds_total": round(time.time() - started_at, 1),
    }

    output_payload = {
        "metadata": metadata,
        "variants": [variant.__dict__ for variant in DETECTOR_VARIANTS],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)

    sample_audit_payload = {
        "metadata": metadata,
        "variants": [variant.__dict__ for variant in AUDIT_VARIANTS],
        "samples": sample_audit,
    }
    args.sample_audit_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sample_audit_output, "w", encoding="utf-8") as handle:
        json.dump(sample_audit_payload, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved detector eval to: {args.output}")
    print(f"Saved sample audit to: {args.sample_audit_output}")

    print("\nSummary:")
    for split in args.splits:
        split_results = results[split]
        print(f"  [{split}]")
        for variant in DETECTOR_VARIANTS:
            metrics = split_results[variant.name]
            print(
                f"    {variant.name}: "
                f"F1@0.3={metrics['metrics_at_0.3']['f1']:.4f} | "
                f"R@0.3={metrics['metrics_at_0.3']['recall']:.4f} | "
                f"IoU={metrics['avg_matched_box_iou']:.4f} | "
                f"pred/img={metrics['avg_pred_boxes_per_image']:.2f}"
            )


if __name__ == "__main__":
    main()
