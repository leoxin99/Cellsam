#!/usr/bin/env python3
"""Formal E2E evaluation for promoted H1bA recall-recovery variants.

Protocol is locked to the prior formal H1bA E2E setup:
- detector input: raw processed images with [BF, DAPI, Actn2]
- segmentation input: T27a BF-only replicated to [BF, BF, BF]
- segmentation inference: InferenceConfig.default()

Evaluated methods:
- raw_cellfinder
- h1ba_adaptive_candidate_aligned_nodrop
- h1ba_adaptive_hybrid_open_fixed0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset, SemanticChannelMapper, load_split_ids
from cellSAM import get_model
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates
from inference.core import InferenceConfig, segment_with_boxes
from metrics.instance_metrics import compute_all_metrics


PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
T33C_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T33c_CellFinder_NoES_seed123"
    / "best_cellfinder.pt"
)
T27A_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T27a_PlanB_DecoderOnly_20260302_033621"
    / "best_model.pt"
)
OUTPUT_PATH = PROJECT_ROOT / "tmp" / "h1ba_recall_recovery_e2e_t33c_t27a.json"


@dataclass(frozen=True)
class MethodConfig:
    name: str
    candidate_mode: str | None
    prior_mode: str | None
    score_filter_mode: str | None
    score_threshold: float | None
    query_output_mode: str
    apply_candidate_mask: bool


METHODS = [
    MethodConfig(
        name="raw_cellfinder",
        candidate_mode=None,
        prior_mode=None,
        score_filter_mode=None,
        score_threshold=None,
        query_output_mode="filtered",
        apply_candidate_mask=True,
    ),
    MethodConfig(
        name="h1ba_adaptive_candidate_aligned_nodrop",
        candidate_mode="adaptive",
        prior_mode="strict",
        score_filter_mode=None,
        score_threshold=None,
        query_output_mode="candidate_aligned",
        apply_candidate_mask=True,
    ),
    MethodConfig(
        name="h1ba_adaptive_hybrid_open_fixed0.25",
        candidate_mode="adaptive",
        prior_mode="hybrid",
        score_filter_mode="fixed",
        score_threshold=0.25,
        query_output_mode="filtered",
        apply_candidate_mask=False,
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


def build_seg_input_tensor(raw_image: np.ndarray, seg_input_mode: str) -> torch.Tensor:
    """Build segmentation input tensor from raw [BF, DAPI, Actn2]."""
    if seg_input_mode == "t27a_bf3":
        bf = raw_image[0].astype(np.float32)
        return torch.from_numpy(np.stack([bf, bf, bf], axis=0)).float()

    if seg_input_mode == "t28_legacy3ch":
        # Legacy semantic mapping used in T28: R=BF, G=Actn2, B=DAPI
        mapper = SemanticChannelMapper(
            use_2ch=False,
            use_official_encoding=False,
        )
        image_hwc = raw_image.transpose(1, 2, 0)  # [H, W, 3] = [BF, DAPI, Actn2]
        mapped_hwc = mapper(image_hwc)            # [H, W, 3] float32 [0, 1]
        mapped_chw = mapped_hwc.transpose(2, 0, 1)
        return torch.from_numpy(mapped_chw).float()

    raise ValueError(
        f"Unsupported seg_input_mode={seg_input_mode}. "
        "Expected 't27a_bf3' or 't28_legacy3ch'."
    )


def setup_detection_model(device: str, checkpoint_path: Path, num_queries: int = 50):
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


def setup_t27a_model(device: str, checkpoint_path: Path):
    model = get_model()
    model.adv_mode = True
    model = model.to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def build_prior_payload(detector_model, raw_image: np.ndarray, candidate_mode: str):
    candidates = detect_h1b_candidates(
        raw_image,
        profile_name="locked_eval",
        candidate_mode=candidate_mode,
    )
    points, valid_mask, _ = candidates_to_query_priors(
        candidates=candidates,
        image_shape=raw_image.shape[-2:],
        max_queries=int(detector_model.cellfinder.args.num_query_position),
    )
    return {
        "candidates": candidates,
        "points": points,
        "valid_mask": valid_mask,
    }


@torch.no_grad()
def detect_boxes(detector_model, raw_image: np.ndarray, method: MethodConfig):
    image_tensor = torch.from_numpy(raw_image).float()
    candidate_count = 0
    candidate_points_per_image = None
    candidate_valid_masks = None

    if method.candidate_mode is not None:
        prior_payload = build_prior_payload(detector_model, raw_image, method.candidate_mode)
        candidate_count = len(prior_payload["candidates"])
        candidate_points_per_image = [prior_payload["points"]]
        candidate_valid_masks = [prior_payload["valid_mask"]]

    boxes_list = detector_model.generate_bounding_boxes(
        [image_tensor],
        candidate_points_per_image=candidate_points_per_image,
        candidate_valid_masks=candidate_valid_masks,
        prior_mode=method.prior_mode,
        score_filter_mode=method.score_filter_mode,
        score_threshold=method.score_threshold,
        query_output_mode=method.query_output_mode,
        apply_candidate_mask=method.apply_candidate_mask,
    )
    if not boxes_list or len(boxes_list[0]) == 0:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        pred_boxes = boxes_list[0].detach().cpu().numpy().astype(np.float32)
    return pred_boxes, candidate_count


def aggregate_metrics(per_image, total_detected, total_candidates, n_samples):
    result = {}
    for key in [
        "pq",
        "sq",
        "rq",
        "bm_1to1_dice",
        "bm_coverage_dice",
        "aji",
        "semantic_dice",
    ]:
        values = [record[key] for record in per_image]
        result[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        result[f"{key}_std"] = float(np.std(values)) if values else 0.0

    total_tp = sum(int(record["tp"]) for record in per_image)
    total_fp = sum(int(record["fp"]) for record in per_image)
    total_fn = sum(int(record["fn"]) for record in per_image)
    total_gt = sum(int(record["n_gt_cells"]) for record in per_image)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_total": int(total_tp),
            "fp_total": int(total_fp),
            "fn_total": int(total_fn),
            "total_detected_boxes": int(total_detected),
            "total_candidates": int(total_candidates),
            "avg_detected_per_image": round(total_detected / max(n_samples, 1), 3),
            "avg_candidates_per_image": round(total_candidates / max(n_samples, 1), 3),
            "avg_gt_per_image": round(total_gt / max(n_samples, 1), 3),
            "n_samples": int(n_samples),
        }
    )
    return result


def evaluate_method(
    t27a_model,
    detector_model,
    dataset,
    split: str,
    method: MethodConfig,
    infer_cfg: InferenceConfig,
    device: str,
    seg_input_mode: str,
):
    per_image = []
    total_detected = 0
    total_candidates = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        gt_mask = sample["mask"].numpy().astype(np.int32)
        raw_image = load_raw_image(sample_id)
        pred_boxes, candidate_count = detect_boxes(detector_model, raw_image, method)
        total_detected += len(pred_boxes)
        total_candidates += candidate_count

        input_tensor = build_seg_input_tensor(raw_image, seg_input_mode=seg_input_mode)
        if len(pred_boxes) == 0:
            pred_mask = np.zeros_like(gt_mask)
        else:
            seg_result = segment_with_boxes(
                model=t27a_model,
                image=input_tensor,
                boxes=torch.tensor(pred_boxes, dtype=torch.float32),
                config=infer_cfg,
                device=device,
            )
            pred_mask = seg_result.instance_mask

        metrics = compute_all_metrics(pred_mask, gt_mask, iou_threshold=0.5)
        metrics["sample_id"] = sample_id
        metrics["split"] = split
        metrics["n_input_boxes"] = int(len(pred_boxes))
        metrics["n_candidates"] = int(candidate_count)
        per_image.append(metrics)

        if (index + 1) % 10 == 0:
            print(f"  [{method.name}] {split} {index + 1}/{len(dataset)}")

    return aggregate_metrics(per_image, total_detected, total_candidates, len(dataset))


def build_comparison_summary(results_by_split):
    summary = {}
    for split, split_results in results_by_split.items():
        raw = split_results["raw_cellfinder"]
        for variant_name in (
            "h1ba_adaptive_candidate_aligned_nodrop",
            "h1ba_adaptive_hybrid_open_fixed0.25",
        ):
            variant = split_results[variant_name]
            summary[f"{split}:{variant_name}"] = {
                "delta_pq_mean_vs_raw": round(variant["pq_mean"] - raw["pq_mean"], 4),
                "delta_f1_vs_raw": round(variant["f1"] - raw["f1"], 4),
                "delta_avg_detected_per_image_vs_raw": round(
                    variant["avg_detected_per_image"] - raw["avg_detected_per_image"],
                    3,
                ),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Formal E2E evaluation for H1bA recall-recovery variants")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--detector-checkpoint", type=Path, default=T33C_CKPT)
    parser.add_argument("--t27a-checkpoint", type=Path, default=T27A_CKPT)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument(
        "--seg-input-mode",
        type=str,
        default="t27a_bf3",
        choices=["t27a_bf3", "t28_legacy3ch"],
        help="Segmentation model input protocol",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.detector_checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")
    if not args.t27a_checkpoint.exists():
        raise FileNotFoundError(f"T27a checkpoint not found: {args.t27a_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Detector checkpoint: {args.detector_checkpoint}")
    print(f"T27a checkpoint: {args.t27a_checkpoint}")
    print(f"Segmentation input mode: {args.seg_input_mode}")

    detector_model = setup_detection_model(
        device=device,
        checkpoint_path=args.detector_checkpoint,
        num_queries=args.num_queries,
    )
    t27a_model = setup_t27a_model(device=device, checkpoint_path=args.t27a_checkpoint)
    infer_cfg = InferenceConfig.default()

    results = {}
    started_at = time.time()
    for split in args.splits:
        split_ids = load_split_ids(split, str(PROJECT_ROOT / "data" / "splits"))
        dataset = AugmentedAllenDataset(
            data_dir=str(PROJECT_ROOT / "data" / "processed"),
            sample_ids=split_ids,
            is_training=False,
            use_bf_only=True,
        )
        print(f"\n=== Split: {split} ({len(dataset)} images) ===")

        split_results = {}
        for method in METHODS:
            print(f"\n--- {method.name} ---")
            t0 = time.time()
            split_results[method.name] = evaluate_method(
                t27a_model=t27a_model,
                detector_model=detector_model,
                dataset=dataset,
                split=split,
                method=method,
                infer_cfg=infer_cfg,
                device=device,
                seg_input_mode=args.seg_input_mode,
            )
            split_results[method.name]["elapsed_seconds"] = round(time.time() - t0, 1)
            print(
                f"  PQ={split_results[method.name]['pq_mean']:.4f}, "
                f"F1={split_results[method.name]['f1']:.4f}, "
                f"detected/img={split_results[method.name]['avg_detected_per_image']:.3f}, "
                f"candidates/img={split_results[method.name]['avg_candidates_per_image']:.3f}"
            )

        results[split] = split_results

    metadata = {
        "script": "tools/eval_h1ba_recall_recovery_e2e.py",
        "t27a_checkpoint": str(args.t27a_checkpoint.resolve()),
        "cellfinder_checkpoint": str(args.detector_checkpoint.resolve()),
        "num_queries": int(args.num_queries),
        "detector_input_protocol": "raw data/processed/images/*.npy with [BF, DAPI, Actn2]",
        "segmentation_input_protocol": (
            "T27a BF-only -> [BF, BF, BF]"
            if args.seg_input_mode == "t27a_bf3"
            else "T28 legacy semantic mapping -> [BF, Actn2, DAPI]"
        ),
        "seg_input_mode": args.seg_input_mode,
        "apply_box_clipping": bool(infer_cfg.apply_box_clipping),
        "box_expand": float(infer_cfg.box_expand),
        "methods": [asdict(method) for method in METHODS],
        "splits": args.splits,
        "elapsed_seconds_total": round(time.time() - started_at, 1),
    }
    comparison_summary = build_comparison_summary(results)

    payload = {
        "metadata": metadata,
        "results": results,
        "comparison_summary": comparison_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved E2E results to: {args.output}")
    print("\nShort comparison vs raw:")
    for key, value in comparison_summary.items():
        print(
            f"  {key}: ΔPQ={value['delta_pq_mean_vs_raw']:+.4f}, "
            f"ΔF1={value['delta_f1_vs_raw']:+.4f}, "
            f"Δdet/img={value['delta_avg_detected_per_image_vs_raw']:+.3f}"
        )


if __name__ == "__main__":
    main()
