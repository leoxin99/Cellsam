#!/usr/bin/env python3
"""Napari viewer for raw CellFinder vs H1bA box comparison.

Shows, for the same sample:
- GT instance mask
- GT boxes
- raw CellFinder detector boxes
- H1bA adaptive detector boxes
- H1bA dapi_cm detector boxes
- optional upstream candidate boxes / centers for both H1bA sources

Protocol is pinned to the current formal H1bA evaluation:
- detector checkpoint: T33c_CellFinder_NoES_seed123
- raw CellFinder: no prior + default dynamic score filtering
- H1bA adaptive: adaptive candidates + strict prior + fixed(0.3)
- H1bA dapi_cm: dapi_cm candidates + strict prior + fixed(0.3)

Usage:
    conda activate cellsam
    python tools/visualize_h1ba_box_comparison.py --split test --samples 3
    python tools/visualize_h1ba_box_comparison.py --sample-ids <id1> <id2>
    python tools/visualize_h1ba_box_comparison.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.measure import regionprops


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import load_split_ids
from cellSAM import get_model
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates


PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
PROCESSED_MASKS_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
T33C_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "T33c_CellFinder_NoES_seed123"
    / "best_cellfinder.pt"
)


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


def extract_gt_boxes(mask: np.ndarray) -> np.ndarray:
    boxes = []
    for prop in regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = prop.bbox
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def boxes_to_rects(boxes_xyxy: np.ndarray, y_offset: int = 0) -> list[np.ndarray]:
    rects = []
    for x1, y1, x2, y2 in boxes_xyxy:
        rects.append(
            np.asarray(
                [
                    [y1 + y_offset, x1],
                    [y1 + y_offset, x2],
                    [y2 + y_offset, x2],
                    [y2 + y_offset, x1],
                ],
                dtype=np.float32,
            )
        )
    return rects


def points_to_napari(points_xy: np.ndarray, y_offset: int = 0) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([points_xy[:, 1] + y_offset, points_xy[:, 0]], axis=1).astype(np.float32)


def load_raw_image(sample_id: str) -> np.ndarray:
    raw = np.load(PROCESSED_IMAGES_DIR / f"{sample_id}.npy")
    return to_chw(raw)


def load_mask(sample_id: str) -> np.ndarray:
    return np.load(PROCESSED_MASKS_DIR / f"{sample_id}.npy")


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


@torch.no_grad()
def run_raw_cellfinder(model, raw_image: np.ndarray) -> np.ndarray:
    image_tensor = torch.from_numpy(raw_image).float()
    boxes_list = model.generate_bounding_boxes([image_tensor])
    if not boxes_list or len(boxes_list[0]) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes_list[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_h1ba_detector(
    model,
    raw_image: np.ndarray,
    candidate_mode: str,
) -> dict[str, np.ndarray]:
    image_tensor = torch.from_numpy(raw_image).float()
    candidates = detect_h1b_candidates(
        raw_image,
        profile_name="locked_eval",
        candidate_mode=candidate_mode,
    )
    points, valid_mask, _ = candidates_to_query_priors(
        candidates=candidates,
        image_shape=raw_image.shape[-2:],
        max_queries=int(model.cellfinder.args.num_query_position),
    )
    boxes_list = model.generate_bounding_boxes(
        [image_tensor],
        candidate_points_per_image=[points],
        candidate_valid_masks=[valid_mask],
        prior_mode="strict",
        score_filter_mode="fixed",
        score_threshold=0.3,
    )
    pred_boxes = (
        np.zeros((0, 4), dtype=np.float32)
        if not boxes_list or len(boxes_list[0]) == 0
        else boxes_list[0].detach().cpu().numpy().astype(np.float32)
    )
    candidate_boxes = np.asarray(
        [candidate["box_xyxy"] for candidate in candidates],
        dtype=np.float32,
    )
    candidate_points = np.asarray(
        [candidate["center_xy"] for candidate in candidates],
        dtype=np.float32,
    )
    if candidate_boxes.size == 0:
        candidate_boxes = np.zeros((0, 4), dtype=np.float32)
    if candidate_points.size == 0:
        candidate_points = np.zeros((0, 2), dtype=np.float32)
    return {
        "pred_boxes": pred_boxes,
        "candidate_boxes": candidate_boxes,
        "candidate_points": candidate_points,
    }


def choose_sample_ids(split: str, samples: int, sample_ids: list[str] | None) -> list[str]:
    if sample_ids:
        return sample_ids
    return load_split_ids(split, str(SPLITS_DIR))[:samples]


def add_box_layer(viewer, name: str, boxes_xyxy: np.ndarray, color, y_offset: int, visible: bool):
    rects = boxes_to_rects(boxes_xyxy, y_offset=y_offset)
    if rects:
        viewer.add_shapes(
            rects,
            shape_type="polygon",
            edge_color=color,
            edge_width=2,
            face_color="transparent",
            name=name,
            visible=visible,
        )


def main():
    parser = argparse.ArgumentParser(description="Napari viewer for H1bA box comparison")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to show")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Explicit sample IDs")
    parser.add_argument("--checkpoint", type=Path, default=T33C_CKPT)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Run inference and print counts only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    sample_ids = choose_sample_ids(args.split, args.samples, args.sample_ids)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples ({len(sample_ids)}):")
    for sample_id in sample_ids:
        print(f"  {sample_id}")

    model = setup_model(device=device, checkpoint_path=args.checkpoint, num_queries=args.num_queries)

    sample_outputs = []
    for sample_id in sample_ids:
        raw_image = load_raw_image(sample_id)
        mask = load_mask(sample_id)
        gt_boxes = extract_gt_boxes(mask)
        raw_boxes = run_raw_cellfinder(model, raw_image)
        adaptive = run_h1ba_detector(model, raw_image, candidate_mode="adaptive")
        dapi_cm = run_h1ba_detector(model, raw_image, candidate_mode="dapi_cm")

        summary = {
            "sample_id": sample_id,
            "gt_cells": int(len(gt_boxes)),
            "raw_boxes": int(len(raw_boxes)),
            "adaptive_candidates": int(len(adaptive["candidate_boxes"])),
            "adaptive_boxes": int(len(adaptive["pred_boxes"])),
            "dapi_cm_candidates": int(len(dapi_cm["candidate_boxes"])),
            "dapi_cm_boxes": int(len(dapi_cm["pred_boxes"])),
        }
        print(
            "[sample] "
            f"{summary['sample_id']} | GT={summary['gt_cells']} | "
            f"raw={summary['raw_boxes']} | "
            f"adaptive cand/pred={summary['adaptive_candidates']}/{summary['adaptive_boxes']} | "
            f"dapi_cm cand/pred={summary['dapi_cm_candidates']}/{summary['dapi_cm_boxes']}"
        )
        sample_outputs.append(
            {
                "sample_id": sample_id,
                "raw_image": raw_image,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "raw_boxes": raw_boxes,
                "adaptive": adaptive,
                "dapi_cm": dapi_cm,
            }
        )

    if args.dry_run:
        return

    import napari

    viewer = napari.Viewer(title="H1bA box comparison: raw vs adaptive vs dapi_cm")
    for index, item in enumerate(sample_outputs):
        sample_id = item["sample_id"]
        raw_image = item["raw_image"]
        mask = item["mask"]
        y_offset = index * (raw_image.shape[1] + 20)
        visible = index == 0

        bf = normalize_channel(raw_image[0])
        dapi = normalize_channel(raw_image[1])
        actn2 = normalize_channel(raw_image[2])

        viewer.add_image(
            bf,
            name=f"[{index}] BF | {sample_id}",
            translate=[y_offset, 0],
            colormap="gray",
            visible=visible,
        )
        viewer.add_image(
            dapi,
            name=f"[{index}] DAPI | {sample_id}",
            translate=[y_offset, 0],
            colormap="blue",
            visible=False,
            blending="additive",
        )
        viewer.add_image(
            actn2,
            name=f"[{index}] Actn2 | {sample_id}",
            translate=[y_offset, 0],
            colormap="green",
            visible=False,
            blending="additive",
        )
        viewer.add_labels(
            mask.astype(np.int32),
            name=f"[{index}] GT mask | {sample_id}",
            translate=[y_offset, 0],
            visible=visible,
            opacity=0.35,
        )

        add_box_layer(
            viewer,
            name=f"[{index}] GT boxes ({len(item['gt_boxes'])}) | {sample_id}",
            boxes_xyxy=item["gt_boxes"],
            color="green",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] raw CellFinder ({len(item['raw_boxes'])}) | {sample_id}",
            boxes_xyxy=item["raw_boxes"],
            color="yellow",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] H1bA adaptive pred ({len(item['adaptive']['pred_boxes'])}) | {sample_id}",
            boxes_xyxy=item["adaptive"]["pred_boxes"],
            color="cyan",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] H1bA dapi_cm pred ({len(item['dapi_cm']['pred_boxes'])}) | {sample_id}",
            boxes_xyxy=item["dapi_cm"]["pred_boxes"],
            color="magenta",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] adaptive candidate boxes ({len(item['adaptive']['candidate_boxes'])}) | {sample_id}",
            boxes_xyxy=item["adaptive"]["candidate_boxes"],
            color="cyan",
            y_offset=y_offset,
            visible=False,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] dapi_cm candidate boxes ({len(item['dapi_cm']['candidate_boxes'])}) | {sample_id}",
            boxes_xyxy=item["dapi_cm"]["candidate_boxes"],
            color="magenta",
            y_offset=y_offset,
            visible=False,
        )

        adaptive_points = points_to_napari(item["adaptive"]["candidate_points"], y_offset=y_offset)
        if len(adaptive_points) > 0:
            viewer.add_points(
                adaptive_points,
                name=f"[{index}] adaptive candidate centers ({len(adaptive_points)}) | {sample_id}",
                size=5,
                face_color="cyan",
                border_color="black",
                visible=False,
            )
        dapi_cm_points = points_to_napari(item["dapi_cm"]["candidate_points"], y_offset=y_offset)
        if len(dapi_cm_points) > 0:
            viewer.add_points(
                dapi_cm_points,
                name=f"[{index}] dapi_cm candidate centers ({len(dapi_cm_points)}) | {sample_id}",
                size=5,
                face_color="magenta",
                border_color="black",
                visible=False,
            )

    print("Napari layers:")
    print("  green   = GT boxes / GT mask")
    print("  yellow  = raw CellFinder boxes (T33c fine-tuned detector)")
    print("  cyan    = H1bA adaptive predicted boxes")
    print("  magenta = H1bA dapi_cm predicted boxes")
    print("  hidden layers also include candidate boxes and candidate centers")
    napari.run()


if __name__ == "__main__":
    main()
