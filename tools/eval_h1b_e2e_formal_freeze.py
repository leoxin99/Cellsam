#!/usr/bin/env python3
"""Single-source formal E2E freeze for H1b paper table.

This script evaluates three detector->segmentation arms under one locked protocol:
1) raw_cellfinder_t33b_s42_t28
2) t33f_adaptive_candidate_aligned_t28
3) t33g_dapicm_candidate_aligned_t28

Outputs:
- one JSON artifact (authoritative source)
- one markdown summary table for A3 integration
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
DEFAULT_SPLITS = ["val", "test"]

DEFAULT_T28_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T28_PlanB_3ch_seed123_20260302_173912"
    / "best_model.pt"
)

DEFAULT_ARMS = [
    {
        "name": "raw_cellfinder_t33b_s42_t28",
        "detector_checkpoint": str(
            PROJECT_ROOT
            / "checkpoints"
            / "T33b_CellFinder_AP50_seed42_20260309_045037"
            / "best_cellfinder.pt"
        ),
        "num_queries": 50,
        "candidate_mode": None,
        "prior_mode": None,
        "score_filter_mode": None,
        "score_threshold": None,
        "query_output_mode": "filtered",
        "apply_candidate_mask": True,
    },
    {
        "name": "t33f_adaptive_candidate_aligned_t28",
        "detector_checkpoint": str(
            PROJECT_ROOT
            / "checkpoints"
            / "T33f_CandidateAware_adaptive_strict_q35_f1p03_seed123_20260318_032125"
            / "best_cellfinder.pt"
        ),
        "num_queries": 35,
        "candidate_mode": "adaptive",
        "prior_mode": "strict",
        "score_filter_mode": None,
        "score_threshold": None,
        "query_output_mode": "candidate_aligned",
        "apply_candidate_mask": True,
    },
    {
        "name": "t33g_dapicm_candidate_aligned_t28",
        "detector_checkpoint": str(
            PROJECT_ROOT
            / "checkpoints"
            / "T33g_CandidateAware_dapicm_strict_q35_f1p03_seed123_20260318_032132"
            / "best_cellfinder.pt"
        ),
        "num_queries": 35,
        "candidate_mode": "dapi_cm",
        "prior_mode": "strict",
        "score_filter_mode": None,
        "score_threshold": None,
        "query_output_mode": "candidate_aligned",
        "apply_candidate_mask": True,
    },
]


@dataclass(frozen=True)
class ArmConfig:
    name: str
    detector_checkpoint: Path
    num_queries: int
    candidate_mode: str | None
    prior_mode: str | None
    score_filter_mode: str | None
    score_threshold: float | None
    query_output_mode: str
    apply_candidate_mask: bool


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


def build_t28_input_tensor(raw_image: np.ndarray) -> torch.Tensor:
    # Legacy semantic mapping for T28: [R,G,B] = [BF, Actn2, DAPI]
    mapper = SemanticChannelMapper(
        use_2ch=False,
        use_official_encoding=False,
    )
    image_hwc = raw_image.transpose(1, 2, 0)  # [H, W, 3] = [BF, DAPI, Actn2]
    mapped_hwc = mapper(image_hwc)
    mapped_chw = mapped_hwc.transpose(2, 0, 1)
    return torch.from_numpy(mapped_chw).float()


def setup_detection_model(device: str, checkpoint_path: Path, num_queries: int):
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


def setup_seg_model(device: str, checkpoint_path: Path):
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
def detect_boxes(detector_model, raw_image: np.ndarray, arm: ArmConfig):
    image_tensor = torch.from_numpy(raw_image).float()
    candidate_count = 0
    candidate_points_per_image = None
    candidate_valid_masks = None

    if arm.candidate_mode is not None:
        prior_payload = build_prior_payload(detector_model, raw_image, arm.candidate_mode)
        candidate_count = len(prior_payload["candidates"])
        candidate_points_per_image = [prior_payload["points"]]
        candidate_valid_masks = [prior_payload["valid_mask"]]

    boxes_list = detector_model.generate_bounding_boxes(
        [image_tensor],
        candidate_points_per_image=candidate_points_per_image,
        candidate_valid_masks=candidate_valid_masks,
        prior_mode=arm.prior_mode,
        score_filter_mode=arm.score_filter_mode,
        score_threshold=arm.score_threshold,
        query_output_mode=arm.query_output_mode,
        apply_candidate_mask=arm.apply_candidate_mask,
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


def evaluate_arm(
    seg_model,
    detector_model,
    dataset,
    split: str,
    arm: ArmConfig,
    infer_cfg: InferenceConfig,
    device: str,
):
    per_image = []
    total_detected = 0
    total_candidates = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        gt_mask = sample["mask"].numpy().astype(np.int32)
        raw_image = load_raw_image(sample_id)

        pred_boxes, candidate_count = detect_boxes(detector_model, raw_image, arm)
        total_detected += len(pred_boxes)
        total_candidates += candidate_count

        input_tensor = build_t28_input_tensor(raw_image)
        if len(pred_boxes) == 0:
            pred_mask = np.zeros_like(gt_mask)
        else:
            seg_result = segment_with_boxes(
                model=seg_model,
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
            print(f"  [{arm.name}] {split} {index + 1}/{len(dataset)}")

    return aggregate_metrics(per_image, total_detected, total_candidates, len(dataset))


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# H1b Formal E2E Freeze (Single Source, Same Protocol)")
    lines.append("")
    lines.append("This file is generated from one run and is the sole source for A3 table entry.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- script: `{payload['metadata']['script']}`")
    lines.append(f"- created_at: `{payload['metadata']['created_at']}`")
    lines.append(f"- detector_input: `{payload['metadata']['detector_input_protocol']}`")
    lines.append(f"- segmentation_input: `{payload['metadata']['segmentation_input_protocol']}`")
    lines.append(f"- segmentation_checkpoint: `{payload['metadata']['segmentation_checkpoint']}`")
    lines.append(f"- apply_box_clipping: `{payload['metadata']['apply_box_clipping']}`")
    lines.append("")

    for split, split_results in payload["results"].items():
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append("| Method | P | R | F1 | PQ | TP | FP | FN | det/img | cand/img |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method in payload["metadata"]["arm_order"]:
            row = split_results[method]
            lines.append(
                "| "
                + method
                + f" | {row['precision']:.4f}"
                + f" | {row['recall']:.4f}"
                + f" | {row['f1']:.4f}"
                + f" | {row['pq_mean']:.4f}"
                + f" | {row['tp_total']}"
                + f" | {row['fp_total']}"
                + f" | {row['fn_total']}"
                + f" | {row['avg_detected_per_image']:.3f}"
                + f" | {row['avg_candidates_per_image']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_arms() -> list[ArmConfig]:
    arms: list[ArmConfig] = []
    for item in DEFAULT_ARMS:
        arms.append(
            ArmConfig(
                name=item["name"],
                detector_checkpoint=Path(item["detector_checkpoint"]),
                num_queries=int(item["num_queries"]),
                candidate_mode=item["candidate_mode"],
                prior_mode=item["prior_mode"],
                score_filter_mode=item["score_filter_mode"],
                score_threshold=item["score_threshold"],
                query_output_mode=item["query_output_mode"],
                apply_candidate_mask=bool(item["apply_candidate_mask"]),
            )
        )
    return arms


def main():
    parser = argparse.ArgumentParser(description="Single-source formal E2E freeze for H1b")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--seg-checkpoint", type=Path, default=DEFAULT_T28_CKPT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "h1b_e2e_formal_t28_single_source.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "results" / "h1b_e2e_formal_t28_single_source.md",
    )
    args = parser.parse_args()

    if not args.seg_checkpoint.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {args.seg_checkpoint}")

    arms = parse_arms()
    for arm in arms:
        if not arm.detector_checkpoint.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {arm.detector_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seg checkpoint: {args.seg_checkpoint}")

    started_at = time.time()
    seg_model = setup_seg_model(device=device, checkpoint_path=args.seg_checkpoint)
    infer_cfg = InferenceConfig.default()

    detectors = {}
    for arm in arms:
        print(f"Loading detector arm: {arm.name}")
        detectors[arm.name] = setup_detection_model(
            device=device,
            checkpoint_path=arm.detector_checkpoint,
            num_queries=arm.num_queries,
        )

    results = {}
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
        for arm in arms:
            print(f"\n--- {arm.name} ---")
            t0 = time.time()
            split_results[arm.name] = evaluate_arm(
                seg_model=seg_model,
                detector_model=detectors[arm.name],
                dataset=dataset,
                split=split,
                arm=arm,
                infer_cfg=infer_cfg,
                device=device,
            )
            split_results[arm.name]["elapsed_seconds"] = round(time.time() - t0, 1)
            row = split_results[arm.name]
            print(
                f"  PQ={row['pq_mean']:.4f}, F1={row['f1']:.4f}, "
                f"det/img={row['avg_detected_per_image']:.3f}, cand/img={row['avg_candidates_per_image']:.3f}"
            )
        results[split] = split_results

    metadata = {
        "script": "tools/eval_h1b_e2e_formal_freeze.py",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds_total": round(time.time() - started_at, 1),
        "detector_input_protocol": "raw data/processed/images/*.npy with [BF, DAPI, Actn2]",
        "segmentation_input_protocol": "T28 legacy semantic mapping -> [BF, Actn2, DAPI]",
        "segmentation_checkpoint": str(args.seg_checkpoint.resolve()),
        "num_queries_per_arm": {arm.name: int(arm.num_queries) for arm in arms},
        "apply_box_clipping": bool(infer_cfg.apply_box_clipping),
        "box_expand": float(infer_cfg.box_expand),
        "splits": args.splits,
        "arm_order": [arm.name for arm in arms],
        "arms": [asdict(arm) | {"detector_checkpoint": str(arm.detector_checkpoint.resolve())} for arm in arms],
    }

    payload = {
        "metadata": metadata,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    md_text = build_markdown(payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as handle:
        handle.write(md_text)

    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved MD  : {args.output_md}")


if __name__ == "__main__":
    main()
