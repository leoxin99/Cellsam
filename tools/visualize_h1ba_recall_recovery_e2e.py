"""Napari viewer for H1bA recall-recovery E2E comparison.

Shows the current locked downstream protocol for:
- raw CellFinder -> T27a
- H1bA adaptive candidate_aligned_nodrop -> T27a

Visible layers focus on:
- BF / DAPI / Actn2 channels
- GT instance mask
- raw CellFinder boxes
- raw CellFinder -> T27a segmentation
- H1bA candidate-aligned boxes
- H1bA candidate-aligned -> T27a segmentation

Protocol is pinned to the formal E2E evaluation:
- detector checkpoint: T33c_CellFinder_NoES_seed123
- segmentation checkpoint: T27a_PlanB_DecoderOnly_20260302_033621
- detector input: raw processed [BF, DAPI, Actn2]
- segmentation input: BF-only replicated to [BF, BF, BF]

Usage:
    conda activate cellsam
    python tools/visualize_h1ba_recall_recovery_e2e.py --split test --samples 3
    python tools/visualize_h1ba_recall_recovery_e2e.py --sample-ids <id1> <id2>
    python tools/visualize_h1ba_recall_recovery_e2e.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import load_split_ids
from cellSAM import get_model
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates
from inference.core import InferenceConfig, segment_with_boxes


PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
PROCESSED_MASKS_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
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


def load_raw_image(sample_id: str) -> np.ndarray:
    raw = np.load(PROCESSED_IMAGES_DIR / f"{sample_id}.npy")
    return to_chw(raw)


def load_mask(sample_id: str) -> np.ndarray:
    return np.load(PROCESSED_MASKS_DIR / f"{sample_id}.npy")


def choose_sample_ids(split: str, samples: int, sample_ids: list[str] | None) -> list[str]:
    if sample_ids:
        return sample_ids
    return load_split_ids(split, str(SPLITS_DIR))[:samples]


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


def build_candidate_priors(detector_model, raw_image: np.ndarray, candidate_mode: str):
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
    return candidates, points, valid_mask


@torch.no_grad()
def detect_raw_boxes(detector_model, raw_image: np.ndarray) -> np.ndarray:
    image_tensor = torch.from_numpy(raw_image).float()
    boxes_list = detector_model.generate_bounding_boxes([image_tensor])
    if not boxes_list or len(boxes_list[0]) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes_list[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def detect_candidate_aligned_boxes(
    detector_model,
    raw_image: np.ndarray,
    candidate_mode: str,
):
    image_tensor = torch.from_numpy(raw_image).float()
    candidates, points, valid_mask = build_candidate_priors(
        detector_model,
        raw_image,
        candidate_mode=candidate_mode,
    )
    boxes_list = detector_model.generate_bounding_boxes(
        [image_tensor],
        candidate_points_per_image=[points],
        candidate_valid_masks=[valid_mask],
        prior_mode="strict",
        query_output_mode="candidate_aligned",
        apply_candidate_mask=True,
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
    if candidate_boxes.size == 0:
        candidate_boxes = np.zeros((0, 4), dtype=np.float32)
    return pred_boxes, candidate_boxes


@torch.no_grad()
def run_t27a_segmentation(t27a_model, raw_image: np.ndarray, boxes_xyxy: np.ndarray, device: str):
    if len(boxes_xyxy) == 0:
        h, w = raw_image.shape[-2:]
        return np.zeros((h, w), dtype=np.int32), 0

    bf = raw_image[0].astype(np.float32)
    bf_3ch = np.stack([bf, bf, bf], axis=0)
    image_tensor = torch.from_numpy(bf_3ch).float()
    boxes_tensor = torch.from_numpy(boxes_xyxy.astype(np.float32))
    result = segment_with_boxes(
        model=t27a_model,
        image=image_tensor,
        boxes=boxes_tensor,
        config=InferenceConfig.default(),
        device=device,
    )
    return result.instance_mask.astype(np.int32), int(result.n_instances)


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
    parser = argparse.ArgumentParser(description="Napari viewer for H1bA recall-recovery E2E comparison")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to show")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Explicit sample IDs")
    parser.add_argument("--detector-checkpoint", type=Path, default=T33C_CKPT)
    parser.add_argument("--seg-checkpoint", type=Path, default=T27A_CKPT)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--candidate-mode", type=str, default="adaptive", choices=["adaptive", "dapi_cm"])
    parser.add_argument("--dry-run", action="store_true", help="Run inference and print counts only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.detector_checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")
    if not args.seg_checkpoint.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {args.seg_checkpoint}")

    sample_ids = choose_sample_ids(args.split, args.samples, args.sample_ids)
    print(f"Device: {device}")
    print(f"Detector checkpoint: {args.detector_checkpoint}")
    print(f"Segmentation checkpoint: {args.seg_checkpoint}")
    print(f"Candidate mode: {args.candidate_mode}")
    print(f"Samples ({len(sample_ids)}):")
    for sample_id in sample_ids:
        print(f"  {sample_id}")

    detector_model = setup_detection_model(
        device=device,
        checkpoint_path=args.detector_checkpoint,
        num_queries=args.num_queries,
    )
    t27a_model = setup_t27a_model(device=device, checkpoint_path=args.seg_checkpoint)

    sample_outputs = []
    for sample_id in sample_ids:
        raw_image = load_raw_image(sample_id)
        gt_mask = load_mask(sample_id)
        raw_boxes = detect_raw_boxes(detector_model, raw_image)
        candidate_boxes, prior_candidate_boxes = detect_candidate_aligned_boxes(
            detector_model,
            raw_image,
            candidate_mode=args.candidate_mode,
        )
        raw_seg, raw_n = run_t27a_segmentation(t27a_model, raw_image, raw_boxes, device=device)
        cand_seg, cand_n = run_t27a_segmentation(
            t27a_model,
            raw_image,
            candidate_boxes,
            device=device,
        )

        print(
            "[sample] "
            f"{sample_id} | GT={int(np.max(gt_mask))} | "
            f"raw boxes/seg={len(raw_boxes)}/{raw_n} | "
            f"cand boxes/seg={len(candidate_boxes)}/{cand_n}"
        )
        sample_outputs.append(
            {
                "sample_id": sample_id,
                "raw_image": raw_image,
                "gt_mask": gt_mask.astype(np.int32),
                "raw_boxes": raw_boxes,
                "candidate_aligned_boxes": candidate_boxes,
                "prior_candidate_boxes": prior_candidate_boxes,
                "raw_seg": raw_seg,
                "candidate_aligned_seg": cand_seg,
            }
        )

    if args.dry_run:
        return

    import napari

    viewer = napari.Viewer(
        title=f"H1bA recall-recovery E2E: raw vs candidate_aligned_nodrop ({args.candidate_mode})"
    )
    for index, item in enumerate(sample_outputs):
        sample_id = item["sample_id"]
        raw_image = item["raw_image"]
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
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            actn2,
            name=f"[{index}] Actn2 | {sample_id}",
            translate=[y_offset, 0],
            colormap="green",
            blending="additive",
            visible=False,
        )
        viewer.add_labels(
            item["gt_mask"],
            name=f"[{index}] GT mask | {sample_id}",
            translate=[y_offset, 0],
            opacity=0.35,
            visible=visible,
        )
        viewer.add_labels(
            item["raw_seg"],
            name=f"[{index}] raw -> T27a seg | {sample_id}",
            translate=[y_offset, 0],
            opacity=0.35,
            visible=False,
        )
        viewer.add_labels(
            item["candidate_aligned_seg"],
            name=f"[{index}] candidate_aligned -> T27a seg | {sample_id}",
            translate=[y_offset, 0],
            opacity=0.35,
            visible=False,
        )

        add_box_layer(
            viewer,
            name=f"[{index}] raw CellFinder boxes ({len(item['raw_boxes'])}) | {sample_id}",
            boxes_xyxy=item["raw_boxes"],
            color="yellow",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] candidate_aligned boxes ({len(item['candidate_aligned_boxes'])}) | {sample_id}",
            boxes_xyxy=item["candidate_aligned_boxes"],
            color="cyan",
            y_offset=y_offset,
            visible=visible,
        )
        add_box_layer(
            viewer,
            name=f"[{index}] {args.candidate_mode} candidate audit boxes ({len(item['prior_candidate_boxes'])}) | {sample_id}",
            boxes_xyxy=item["prior_candidate_boxes"],
            color="magenta",
            y_offset=y_offset,
            visible=False,
        )

    print("Napari layers:")
    print("  green/labels  = GT mask")
    print("  yellow boxes  = raw CellFinder boxes")
    print("  cyan boxes    = candidate_aligned_nodrop boxes")
    print("  hidden labels = raw->T27a seg / candidate_aligned->T27a seg")
    print(f"  hidden boxes  = {args.candidate_mode} candidate audit boxes")
    napari.run()


if __name__ == "__main__":
    main()
