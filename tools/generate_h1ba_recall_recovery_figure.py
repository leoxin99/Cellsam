#!/usr/bin/env python3
"""Generate static H1bA recall-recovery comparison figures (no napari).

Figure layout (rows = samples, columns fixed):
  BF Input | Actn2 Channel | GT Mask | Before (T27a) | After (T27a) | Oracle GT-Box (T27a)

Compared pipelines on the same samples:
  1) before detector + candidate_aligned + T27a segmentation
  2) after detector  + candidate_aligned + T27a segmentation
  3) GT boxes + T27a segmentation (oracle reference)

This script follows the locked H1bA E2E protocol:
  - detector input: raw [BF, DAPI, Actn2] from data/processed/images/*.npy
  - segmentation input: BF-only replicated to [BF, BF, BF]
  - segmentation inference config: InferenceConfig.default()
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset, load_split_ids
from cellSAM import get_model
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates
from inference.core import InferenceConfig, segment_with_boxes
from metrics.instance_metrics import compute_all_metrics


PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "h1ba_recall_recovery"

DEFAULT_BEFORE_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T33c_CellFinder_NoES_seed123"
    / "best_cellfinder.pt"
)
DEFAULT_AFTER_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T33f_CandidateAware_adaptive_strict_q35_f1p03_seed123_20260318_032125"
    / "best_cellfinder.pt"
)
DEFAULT_T27A_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T27a_PlanB_DecoderOnly_20260302_033621"
    / "best_model.pt"
)


@dataclass(frozen=True)
class ArmResult:
    n_boxes: int
    n_instances: int
    n_candidates_total: int
    n_candidates_effective: int
    metrics: dict
    boxes: np.ndarray
    seg_mask: np.ndarray


def to_chw(raw_image: np.ndarray) -> np.ndarray:
    if raw_image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={raw_image.shape}")
    if raw_image.shape[0] in (3, 4, 5):
        return raw_image
    if raw_image.shape[-1] in (3, 4, 5):
        return raw_image.transpose(2, 0, 1)
    raise ValueError(f"Could not infer channel axis for shape={raw_image.shape}")


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    mn = float(channel.min())
    mx = float(channel.max())
    if mx - mn <= 1e-8:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - mn) / (mx - mn), 0.0, 1.0)


def load_raw_image(sample_id: str) -> np.ndarray:
    raw = np.load(PROCESSED_IMAGES_DIR / f"{sample_id}.npy")
    return to_chw(raw)


def extract_gt_boxes_from_mask(mask_np: np.ndarray) -> np.ndarray:
    from skimage.measure import regionprops

    boxes = []
    for prop in regionprops(mask_np.astype(np.int32)):
        y1, x1, y2, x2 = prop.bbox
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def mask_to_rgba(mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    max_id = int(mask.max())
    if max_id <= 0:
        return rgba

    for cell_id in range(1, max_id + 1):
        ys, xs = np.where(mask == cell_id)
        if ys.size == 0:
            continue
        # Deterministic per-instance color.
        rng = np.random.RandomState(cell_id * 1299721 % (2**31 - 1))
        color = np.asarray([rng.uniform(0.15, 1.0), rng.uniform(0.15, 1.0), rng.uniform(0.15, 1.0), alpha], dtype=np.float32)
        rgba[ys, xs] = color
    return rgba


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    from skimage.segmentation import find_boundaries

    if int(mask.max()) <= 0:
        return np.zeros_like(mask, dtype=bool)
    return find_boundaries(mask.astype(np.int32), mode="thick")


def choose_sample_ids(split: str, samples: int, sample_ids: list[str] | None) -> list[str]:
    if sample_ids:
        return sample_ids
    return load_split_ids(split, str(SPLITS_DIR))[:samples]


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


def setup_t27a_model(device: str, checkpoint_path: Path):
    model = get_model()
    model.adv_mode = True
    model = model.to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def build_candidates(raw_image: np.ndarray, candidate_mode: str, profile_name: str):
    return detect_h1b_candidates(
        raw_image=raw_image,
        profile_name=profile_name,
        candidate_mode=candidate_mode,
    )


@torch.no_grad()
def detect_candidate_aligned_boxes(detector_model, raw_image: np.ndarray, candidates: list[dict]):
    points, valid_mask, _ = candidates_to_query_priors(
        candidates=candidates,
        image_shape=raw_image.shape[-2:],
        max_queries=int(detector_model.cellfinder.args.num_query_position),
    )
    image_tensor = torch.from_numpy(raw_image).float()
    boxes_list = detector_model.generate_bounding_boxes(
        [image_tensor],
        candidate_points_per_image=[points],
        candidate_valid_masks=[valid_mask],
        prior_mode="strict",
        query_output_mode="candidate_aligned",
        apply_candidate_mask=True,
    )
    if not boxes_list or len(boxes_list[0]) == 0:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        pred_boxes = boxes_list[0].detach().cpu().numpy().astype(np.float32)
    return pred_boxes, int(len(candidates)), int(valid_mask.sum().item())


@torch.no_grad()
def run_t27a_segmentation(
    t27a_model,
    bf_source: torch.Tensor,
    boxes_xyxy: np.ndarray,
    infer_cfg: InferenceConfig,
    device: str,
):
    if isinstance(bf_source, torch.Tensor):
        bf_np = bf_source[0].detach().cpu().numpy().astype(np.float32)
    else:
        bf_np = np.asarray(bf_source, dtype=np.float32)
    h, w = bf_np.shape[-2], bf_np.shape[-1]

    if len(boxes_xyxy) == 0:
        return np.zeros((h, w), dtype=np.int32), 0

    bf_3ch = np.stack([bf_np, bf_np, bf_np], axis=0)
    image_tensor = torch.from_numpy(bf_3ch).float()
    boxes_tensor = torch.from_numpy(boxes_xyxy.astype(np.float32))
    result = segment_with_boxes(
        model=t27a_model,
        image=image_tensor,
        boxes=boxes_tensor,
        config=infer_cfg,
        device=device,
    )
    return result.instance_mask.astype(np.int32), int(result.n_instances)


def evaluate_arm(
    *,
    detector_model,
    t27a_model,
    sample_image: torch.Tensor,
    gt_mask: np.ndarray,
    raw_image: np.ndarray,
    candidates: list[dict],
    infer_cfg: InferenceConfig,
    device: str,
) -> ArmResult:
    pred_boxes, n_candidates_total, n_candidates_effective = detect_candidate_aligned_boxes(
        detector_model=detector_model,
        raw_image=raw_image,
        candidates=candidates,
    )
    pred_mask, n_instances = run_t27a_segmentation(
        t27a_model=t27a_model,
        bf_source=sample_image,
        boxes_xyxy=pred_boxes,
        infer_cfg=infer_cfg,
        device=device,
    )
    metrics = compute_all_metrics(pred_mask, gt_mask, iou_threshold=0.5)
    return ArmResult(
        n_boxes=int(len(pred_boxes)),
        n_instances=int(n_instances),
        n_candidates_total=n_candidates_total,
        n_candidates_effective=n_candidates_effective,
        metrics=metrics,
        boxes=pred_boxes,
        seg_mask=pred_mask,
    )


def evaluate_oracle_arm(
    *,
    t27a_model,
    sample_image: torch.Tensor,
    gt_mask: np.ndarray,
    infer_cfg: InferenceConfig,
    device: str,
) -> ArmResult:
    gt_boxes = extract_gt_boxes_from_mask(gt_mask)
    pred_mask, n_instances = run_t27a_segmentation(
        t27a_model=t27a_model,
        bf_source=sample_image,
        boxes_xyxy=gt_boxes,
        infer_cfg=infer_cfg,
        device=device,
    )
    metrics = compute_all_metrics(pred_mask, gt_mask, iou_threshold=0.5)
    return ArmResult(
        n_boxes=int(len(gt_boxes)),
        n_instances=int(n_instances),
        n_candidates_total=0,
        n_candidates_effective=0,
        metrics=metrics,
        boxes=gt_boxes,
        seg_mask=pred_mask,
    )


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {}

    def _mean_std(values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
        }

    summary = {}
    for arm in ("before", "after", "oracle"):
        pq = [r[arm]["metrics"]["pq"] for r in records]
        dice = [r[arm]["metrics"]["bm_1to1_dice"] for r in records]
        aji = [r[arm]["metrics"]["aji"] for r in records]
        n_pred = [r[arm]["metrics"]["n_pred_cells"] for r in records]
        summary[arm] = {
            "pq": _mean_std(pq),
            "bm_1to1_dice": _mean_std(dice),
            "aji": _mean_std(aji),
            "n_pred_cells": _mean_std(n_pred),
        }

    deltas = [r["delta_pq_after_before"] for r in records]
    summary["delta_pq_after_before"] = _mean_std(deltas)
    return summary


def evaluate_sample(
    *,
    sample_id: str,
    sample: dict,
    before_detector,
    after_detector,
    t27a_model,
    infer_cfg: InferenceConfig,
    candidate_mode: str,
    profile_name: str,
    device: str,
    include_masks: bool,
) -> dict:
    sample_image = sample["image"].float()
    gt_mask = sample["mask"].numpy().astype(np.int32)
    raw_image = load_raw_image(sample_id)
    candidates = build_candidates(raw_image, candidate_mode=candidate_mode, profile_name=profile_name)

    before = evaluate_arm(
        detector_model=before_detector,
        t27a_model=t27a_model,
        sample_image=sample_image,
        gt_mask=gt_mask,
        raw_image=raw_image,
        candidates=candidates,
        infer_cfg=infer_cfg,
        device=device,
    )
    after = evaluate_arm(
        detector_model=after_detector,
        t27a_model=t27a_model,
        sample_image=sample_image,
        gt_mask=gt_mask,
        raw_image=raw_image,
        candidates=candidates,
        infer_cfg=infer_cfg,
        device=device,
    )
    oracle = evaluate_oracle_arm(
        t27a_model=t27a_model,
        sample_image=sample_image,
        gt_mask=gt_mask,
        infer_cfg=infer_cfg,
        device=device,
    )

    result = {
        "sample_id": sample_id,
        "n_gt_cells": int(gt_mask.max()),
        "before": {
            "n_boxes": before.n_boxes,
            "n_instances": before.n_instances,
            "n_candidates_total": before.n_candidates_total,
            "n_candidates_effective": before.n_candidates_effective,
            "metrics": before.metrics,
        },
        "after": {
            "n_boxes": after.n_boxes,
            "n_instances": after.n_instances,
            "n_candidates_total": after.n_candidates_total,
            "n_candidates_effective": after.n_candidates_effective,
            "metrics": after.metrics,
        },
        "oracle": {
            "n_boxes": oracle.n_boxes,
            "n_instances": oracle.n_instances,
            "metrics": oracle.metrics,
        },
        "delta_pq_after_before": float(after.metrics["pq"] - before.metrics["pq"]),
        "delta_dice_after_before": float(after.metrics["bm_1to1_dice"] - before.metrics["bm_1to1_dice"]),
    }

    if include_masks:
        result["_plot"] = {
            "raw_image": raw_image,
            "gt_mask": gt_mask,
            "before_mask": before.seg_mask,
            "after_mask": after.seg_mask,
            "oracle_mask": oracle.seg_mask,
            "before_boxes": before.boxes,
            "after_boxes": after.boxes,
            "oracle_boxes": oracle.boxes,
        }
    return result


def select_by_mode(records: list[dict], mode: str, samples: int) -> list[dict]:
    if mode == "firstk":
        return records[:samples]

    sorted_by_delta = sorted(records, key=lambda x: x["delta_pq_after_before"], reverse=True)
    if mode == "delta_top":
        return sorted_by_delta[:samples]
    if mode == "delta_bottom":
        sorted_asc = sorted(records, key=lambda x: x["delta_pq_after_before"])
        return sorted_asc[:samples]
    if mode == "delta_median":
        sorted_asc = sorted(records, key=lambda x: x["delta_pq_after_before"])
        n = len(sorted_asc)
        if n == 0:
            return []
        start = max(0, n // 2 - samples // 2)
        end = min(n, start + samples)
        if end - start < samples:
            start = max(0, end - samples)
        return sorted_asc[start:end]
    raise ValueError(f"Unknown mode: {mode}")


def format_metric_box(metrics: dict) -> str:
    return (
        f"PQ={metrics['pq']:.3f}\n"
        f"Dice={metrics['bm_1to1_dice']:.3f}\n"
        f"AJI={metrics['aji']:.3f}\n"
        f"n={int(metrics['n_pred_cells'])}"
    )


def render_figure(records: list[dict], output_png: Path, output_pdf: Path, split: str, mode: str):
    if not records:
        raise RuntimeError("No records to render.")

    col_labels = [
        "BF Input",
        "Actn2 Channel",
        "GT Mask",
        "Before (T27a)",
        "After (T27a)",
        "Oracle GT-Box (T27a)",
    ]
    n_rows = len(records)
    n_cols = len(col_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, record in enumerate(records):
        sample_id = record["sample_id"]
        payload = record["_plot"]
        bf = normalize_channel(payload["raw_image"][0])
        actn2 = normalize_channel(payload["raw_image"][2])
        gt_mask = payload["gt_mask"]
        before_mask = payload["before_mask"]
        after_mask = payload["after_mask"]
        oracle_mask = payload["oracle_mask"]
        masks = [None, None, gt_mask, before_mask, after_mask, oracle_mask]

        delta_pq = record["delta_pq_after_before"]
        row_label = f"{sample_id}\nΔPQ={delta_pq:+.3f}"

        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 1:
                ax.imshow(actn2, cmap="Greens", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(bf, cmap="gray", vmin=0.0, vmax=1.0)

            mask = masks[col]
            if mask is not None and int(mask.max()) > 0:
                ax.imshow(mask_to_rgba(mask, alpha=0.55))
                boundaries = mask_boundaries(mask)
                boundary_overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
                boundary_overlay[boundaries] = [1.0, 1.0, 1.0, 0.9]
                ax.imshow(boundary_overlay)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=8, fontweight="bold")
            elif col == 2:
                ax.text(
                    0.02,
                    0.98,
                    f"n={record['n_gt_cells']}",
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.7},
                )
            elif col == 3:
                ax.text(
                    0.02,
                    0.98,
                    format_metric_box(record["before"]["metrics"]),
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.7},
                )
            elif col == 4:
                ax.text(
                    0.02,
                    0.98,
                    format_metric_box(record["after"]["metrics"]),
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.7},
                )
            elif col == 5:
                ax.text(
                    0.02,
                    0.98,
                    format_metric_box(record["oracle"]["metrics"]),
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.7},
                )

            if row == 0:
                ax.set_title(col_labels[col], fontsize=10, fontweight="bold", pad=6)

    fig.suptitle(f"H1bA Recall-Recovery Static Comparison | split={split} | mode={mode}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.6, h_pad=0.4, w_pad=0.35)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output_pdf), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate static H1bA recall-recovery comparison figures")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--mode", type=str, default="firstk", choices=["firstk", "delta_top", "delta_bottom", "delta_median"])

    parser.add_argument("--before-detector-checkpoint", type=Path, default=DEFAULT_BEFORE_CKPT)
    parser.add_argument("--before-num-queries", type=int, default=50)
    parser.add_argument("--after-detector-checkpoint", type=Path, default=DEFAULT_AFTER_CKPT)
    parser.add_argument("--after-num-queries", type=int, default=35)
    parser.add_argument("--seg-checkpoint", type=Path, default=DEFAULT_T27A_CKPT)

    parser.add_argument("--candidate-mode", type=str, default="adaptive")
    parser.add_argument("--profile-name", type=str, default="locked_eval")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if not args.before_detector_checkpoint.exists():
        raise FileNotFoundError(f"Before detector checkpoint not found: {args.before_detector_checkpoint}")
    if not args.after_detector_checkpoint.exists():
        raise FileNotFoundError(f"After detector checkpoint not found: {args.after_detector_checkpoint}")
    if not args.seg_checkpoint.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {args.seg_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()

    print(f"Device: {device}")
    print(f"Before detector: {args.before_detector_checkpoint} (num_queries={args.before_num_queries})")
    print(f"After detector:  {args.after_detector_checkpoint} (num_queries={args.after_num_queries})")
    print(f"Seg checkpoint:  {args.seg_checkpoint}")
    print(f"candidate_mode={args.candidate_mode}, profile_name={args.profile_name}")

    initial_ids = choose_sample_ids(args.split, args.samples, args.sample_ids)
    split_ids = load_split_ids(args.split, str(SPLITS_DIR))
    if args.sample_ids:
        selected_pool = initial_ids
        print(f"Using explicit sample_ids ({len(selected_pool)}). mode={args.mode} ignored for selection.")
    elif args.mode == "firstk":
        selected_pool = initial_ids
    else:
        selected_pool = split_ids

    dataset = AugmentedAllenDataset(
        data_dir=str(PROJECT_ROOT / "data" / "processed"),
        sample_ids=selected_pool,
        is_training=False,
        use_bf_only=True,
    )
    id_to_idx = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}

    before_detector = setup_detection_model(
        device=device,
        checkpoint_path=args.before_detector_checkpoint,
        num_queries=args.before_num_queries,
    )
    after_detector = setup_detection_model(
        device=device,
        checkpoint_path=args.after_detector_checkpoint,
        num_queries=args.after_num_queries,
    )
    t27a_model = setup_t27a_model(device=device, checkpoint_path=args.seg_checkpoint)
    infer_cfg = InferenceConfig.default()

    all_records = []
    for sample_id in tqdm(selected_pool, desc="Evaluating sample pool"):
        idx = id_to_idx[sample_id]
        sample = dataset[idx]
        record = evaluate_sample(
            sample_id=sample_id,
            sample=sample,
            before_detector=before_detector,
            after_detector=after_detector,
            t27a_model=t27a_model,
            infer_cfg=infer_cfg,
            candidate_mode=args.candidate_mode,
            profile_name=args.profile_name,
            device=device,
            include_masks=False,
        )
        all_records.append(record)

    if args.sample_ids:
        selected_records = all_records
    else:
        selected_records = select_by_mode(all_records, mode=args.mode, samples=args.samples)
    selected_ids = [r["sample_id"] for r in selected_records]

    print("\nSelected samples:")
    for r in selected_records:
        print(
            f"  {r['sample_id']} | ΔPQ={r['delta_pq_after_before']:+.4f} | "
            f"before PQ={r['before']['metrics']['pq']:.4f} -> after PQ={r['after']['metrics']['pq']:.4f}"
        )

    # Rebuild dataset for selected rows only (to keep plotting pass simple).
    plot_dataset = AugmentedAllenDataset(
        data_dir=str(PROJECT_ROOT / "data" / "processed"),
        sample_ids=selected_ids,
        is_training=False,
        use_bf_only=True,
    )
    plot_id_to_idx = {sample["sample_id"]: idx for idx, sample in enumerate(plot_dataset.samples)}
    plot_records = []
    for sample_id in tqdm(selected_ids, desc="Running plotting pass"):
        sample = plot_dataset[plot_id_to_idx[sample_id]]
        plot_record = evaluate_sample(
            sample_id=sample_id,
            sample=sample,
            before_detector=before_detector,
            after_detector=after_detector,
            t27a_model=t27a_model,
            infer_cfg=infer_cfg,
            candidate_mode=args.candidate_mode,
            profile_name=args.profile_name,
            device=device,
            include_masks=True,
        )
        plot_records.append(plot_record)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"h1ba_recall_recovery_{args.split}_{args.mode}_k{len(plot_records)}_{timestamp}"
    output_png = args.output_dir / f"{base_name}.png"
    output_pdf = args.output_dir / f"{base_name}.pdf"
    output_json = args.output_dir / f"{base_name}.json"

    render_figure(plot_records, output_png=output_png, output_pdf=output_pdf, split=args.split, mode=args.mode)

    summary = summarize_records(plot_records)
    payload = {
        "metadata": {
            "script": "tools/generate_h1ba_recall_recovery_figure.py",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds_total": round(time.time() - started_at, 2),
            "device": device,
            "split": args.split,
            "mode": args.mode,
            "samples_requested": int(args.samples),
            "sample_ids_override": args.sample_ids if args.sample_ids else None,
            "selected_sample_ids": selected_ids,
            "before_detector_checkpoint": str(args.before_detector_checkpoint.resolve()),
            "before_num_queries": int(args.before_num_queries),
            "after_detector_checkpoint": str(args.after_detector_checkpoint.resolve()),
            "after_num_queries": int(args.after_num_queries),
            "seg_checkpoint": str(args.seg_checkpoint.resolve()),
            "candidate_mode": args.candidate_mode,
            "profile_name": args.profile_name,
            "inference_config": {
                "mask_threshold": float(infer_cfg.mask_threshold),
                "apply_box_clipping": bool(infer_cfg.apply_box_clipping),
                "box_expand": float(infer_cfg.box_expand),
                "apply_postprocess": bool(infer_cfg.apply_postprocess),
            },
            "outputs": {
                "png": str(output_png.resolve()),
                "pdf": str(output_pdf.resolve()),
                "json": str(output_json.resolve()),
            },
        },
        "per_sample": [
            {k: v for k, v in record.items() if k != "_plot"}
            for record in plot_records
        ],
        "summary": summary,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nOutputs:")
    print(f"  PNG : {output_png}")
    print(f"  PDF : {output_pdf}")
    print(f"  JSON: {output_json}")


if __name__ == "__main__":
    main()
