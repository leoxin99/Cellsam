#!/usr/bin/env python3
"""Candidate-aware CellFinder fine-tuning on Allen splits.

This script reuses the existing T33 training stack and adds prior-conditioned
queries during both training and validation:
    candidate_points + candidate_valid_mask -> cellfinder(...)

Goal:
    Align detector training with H1bA inference regime instead of only
    injecting priors at inference time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_cellfinder as t33
from detection.h1b_priors import candidates_to_query_priors, detect_h1b_candidates


class AllenDetectionDatasetWithId(t33.AllenDetectionDataset):
    """T33 dataset + sample_id for prior-cache lookup."""

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        return image, target, self.ids[idx]


def collate_fn_with_ids(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    sample_ids = [item[2] for item in batch]
    return images, targets, sample_ids


def _to_chw(raw_image: np.ndarray) -> np.ndarray:
    if raw_image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={raw_image.shape}")
    if raw_image.shape[0] in (3, 4, 5):
        return raw_image
    if raw_image.shape[-1] in (3, 4, 5):
        return raw_image.transpose(2, 0, 1)
    raise ValueError(f"Could not infer channel axis for shape={raw_image.shape}")


def build_prior_cache(dataset, num_queries, profile_name, candidate_mode):
    """Precompute priors once per sample to avoid repeated CPU detection each epoch."""
    cache = {}
    candidate_counts = []
    truncated_counts = 0

    for sample_id in tqdm(dataset.ids, desc=f"Building prior cache ({candidate_mode})"):
        raw_image = np.load(dataset.image_dir / f"{sample_id}.npy")
        raw_image = _to_chw(raw_image)

        candidates = detect_h1b_candidates(
            raw_image=raw_image,
            profile_name=profile_name,
            candidate_mode=candidate_mode,
        )
        points, valid_mask, _ = candidates_to_query_priors(
            candidates=candidates,
            image_shape=raw_image.shape[-2:],
            max_queries=num_queries,
        )
        n_candidates = len(candidates)
        n_effective = int(valid_mask.sum().item())
        if n_candidates > num_queries:
            truncated_counts += 1

        cache[sample_id] = {
            "points": points,
            "valid_mask": valid_mask,
            "n_candidates": n_candidates,
            "n_effective": n_effective,
        }
        candidate_counts.append(n_candidates)

    mean_candidates = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    p95_candidates = float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0
    max_candidates = int(max(candidate_counts)) if candidate_counts else 0

    summary = {
        "n_samples": len(dataset.ids),
        "candidate_mode": candidate_mode,
        "profile_name": profile_name,
        "num_queries": num_queries,
        "mean_candidates_per_image": round(mean_candidates, 3),
        "p95_candidates_per_image": round(p95_candidates, 3),
        "max_candidates_per_image": max_candidates,
        "n_images_truncated": int(truncated_counts),
    }
    return cache, summary


def _gather_batch_priors(sample_ids, prior_cache, device):
    points = []
    masks = []
    effective_counts = []
    for sample_id in sample_ids:
        payload = prior_cache[sample_id]
        points.append(payload["points"])
        masks.append(payload["valid_mask"])
        effective_counts.append(payload["n_effective"])

    points = torch.stack(points, dim=0).to(device=device, dtype=torch.float32)
    masks = torch.stack(masks, dim=0).to(device=device, dtype=torch.bool)
    return points, masks, effective_counts


def compute_candidate_aligned_f1(preds, targets, iou_thresh=0.3):
    """One-candidate-one-box style F1 without score-threshold dropping."""
    from cellSAM.AnchorDETR.util.box_ops import box_cxcywh_to_xyxy
    from torchvision.ops import box_iou

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        gt_boxes = target["boxes"]

        if len(gt_boxes) == 0:
            all_fp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            all_fn += len(gt_boxes)
            continue

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        iou_matrix = box_iou(pred_xyxy, gt_xyxy)

        matched_gt = set()
        tp = 0
        for i in range(len(pred_xyxy)):
            if len(matched_gt) == len(gt_boxes):
                break
            best_j = -1
            best_iou = iou_thresh
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = float(iou_matrix[i, j].item())
                    best_j = j
            if best_j >= 0:
                tp += 1
                matched_gt.add(best_j)

        all_tp += tp
        all_fp += len(pred_boxes) - tp
        all_fn += len(gt_boxes) - len(matched_gt)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    tag = f"{iou_thresh:.1f}"
    return {
        f"candidate_aligned_precision@{tag}": precision,
        f"candidate_aligned_recall@{tag}": recall,
        f"candidate_aligned_f1@{tag}": f1,
        f"candidate_aligned_tp@{tag}": int(all_tp),
        f"candidate_aligned_fp@{tag}": int(all_fp),
        f"candidate_aligned_fn@{tag}": int(all_fn),
    }


def train_one_epoch_candidate_aware(
    model,
    cellfinder,
    criterion,
    weight_dict,
    dataloader,
    optimizer,
    device,
    epoch,
    prior_cache,
    prior_mode,
    apply_candidate_mask,
    max_train_batches=0,
):
    cellfinder.train()
    total_loss = 0.0
    n_batches = 0
    total_effective_candidates = 0

    for batch_index, (images, targets, sample_ids) in enumerate(
        tqdm(dataloader, desc=f"Epoch {epoch}")
    ):
        if max_train_batches > 0 and batch_index >= max_train_batches:
            break

        targets = [
            {
                "labels": t["labels"].to(device),
                "boxes": t["boxes"].to(device),
            }
            for t in targets
        ]
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        processed_imgs = t33.preprocess_for_cellfinder(model, images, device)
        candidate_points, candidate_valid_mask, effective_counts = _gather_batch_priors(
            sample_ids=sample_ids,
            prior_cache=prior_cache,
            device=device,
        )
        outputs = cellfinder(
            processed_imgs,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode,
            apply_candidate_mask=apply_candidate_mask,
        )

        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict if k in weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(cellfinder.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += float(losses.item())
        n_batches += 1
        total_effective_candidates += int(sum(effective_counts))

    avg_loss = total_loss / max(n_batches, 1)
    avg_effective_candidates_per_batch = total_effective_candidates / max(n_batches, 1)
    return avg_loss, avg_effective_candidates_per_batch


@torch.no_grad()
def evaluate_candidate_aware(
    model,
    cellfinder,
    criterion,
    weight_dict,
    dataloader,
    device,
    prior_cache,
    prior_mode,
    apply_candidate_mask,
    max_val_batches=0,
):
    cellfinder.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_candidate_aligned_preds = []
    all_targets = []
    total_effective_candidates = 0

    for batch_index, (images, targets, sample_ids) in enumerate(
        tqdm(dataloader, desc="Eval")
    ):
        if max_val_batches > 0 and batch_index >= max_val_batches:
            break

        targets = [
            {
                "labels": t["labels"].to(device),
                "boxes": t["boxes"].to(device),
            }
            for t in targets
        ]
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        processed_imgs = t33.preprocess_for_cellfinder(model, images, device)
        candidate_points, candidate_valid_mask, effective_counts = _gather_batch_priors(
            sample_ids=sample_ids,
            prior_cache=prior_cache,
            device=device,
        )
        outputs = cellfinder(
            processed_imgs,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode,
            apply_candidate_mask=apply_candidate_mask,
        )

        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict if k in weight_dict
        )
        total_loss += float(losses.item())
        n_batches += 1
        total_effective_candidates += int(sum(effective_counts))

        pred_logits = outputs["pred_logits"].sigmoid()
        pred_boxes = outputs["pred_boxes"]
        effective_valid_mask = outputs.get("effective_candidate_valid_mask")
        for b in range(pred_logits.shape[0]):
            scores = pred_logits[b, :, 0]
            all_preds.append(
                {
                    "scores": scores.detach().cpu(),
                    "boxes": pred_boxes[b].detach().cpu(),
                }
            )
            if effective_valid_mask is None:
                aligned_boxes = pred_boxes[b].detach().cpu()
            else:
                aligned_boxes = pred_boxes[b][effective_valid_mask[b]].detach().cpu()
            all_candidate_aligned_preds.append({"boxes": aligned_boxes})
            all_targets.append({"boxes": targets[b]["boxes"].detach().cpu()})

    avg_loss = total_loss / max(n_batches, 1)
    f1_metrics = t33.compute_simple_ap(all_preds, all_targets, iou_thresh=0.5)
    coco_metrics = t33.compute_coco_map(all_preds, all_targets)
    candidate_aligned_metrics_03 = compute_candidate_aligned_f1(
        all_candidate_aligned_preds,
        all_targets,
        iou_thresh=0.3,
    )
    candidate_aligned_metrics_05 = compute_candidate_aligned_f1(
        all_candidate_aligned_preds,
        all_targets,
        iou_thresh=0.5,
    )
    candidate_aligned_metrics_07 = compute_candidate_aligned_f1(
        all_candidate_aligned_preds,
        all_targets,
        iou_thresh=0.7,
    )
    merged = {
        **f1_metrics,
        **coco_metrics,
        **candidate_aligned_metrics_03,
        **candidate_aligned_metrics_05,
        **candidate_aligned_metrics_07,
    }
    merged["avg_effective_candidates_per_batch"] = total_effective_candidates / max(n_batches, 1)
    return avg_loss, merged


def main():
    parser = argparse.ArgumentParser(
        description="T33e: Candidate-aware CellFinder fine-tuning"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="coco_ap50",
        choices=[
            "f1",
            "coco_ap",
            "coco_ap50",
            "coco_ap75",
            "candidate_aligned_f1@0.3",
            "candidate_aligned_f1_0p3",
            "candidate_aligned_f1@0.5",
            "candidate_aligned_f1_0p5",
            "candidate_aligned_f1@0.7",
            "candidate_aligned_f1_0p7",
        ],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "dapi_cm"],
    )
    parser.add_argument("--profile-name", type=str, default="locked_eval")
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="strict",
        choices=["strict", "hybrid"],
    )
    parser.add_argument(
        "--disable-candidate-mask",
        action="store_true",
        help="If set, do not mask invalid slots during training/eval.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Apply LoRA to backbone ViT Q/V projections (same as T33).",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Debug/smoke: cap train batches per epoch (0=all).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Debug/smoke: cap val batches per eval (0=all).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    apply_candidate_mask = not args.disable_candidate_mask

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = str(
            PROJECT_ROOT
            / "checkpoints"
            / f"T33e_CellFinder_CandidateAware_{args.candidate_mode}_{args.prior_mode}_seed{args.seed}_{timestamp}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("T33e: Candidate-aware CellFinder fine-tuning")
    print("=" * 72)
    print(f"seed={args.seed} epochs={args.epochs} lr={args.lr} batch={args.batch_size}")
    print(f"candidate_mode={args.candidate_mode} profile={args.profile_name}")
    print(f"prior_mode={args.prior_mode} apply_candidate_mask={apply_candidate_mask}")
    print(f"num_queries={args.num_queries} early_stop_metric={args.early_stop_metric}")
    print(f"device={device}")
    print(f"output={output_dir}")
    print("=" * 72)

    config = vars(args).copy()
    config["apply_candidate_mask"] = apply_candidate_mask
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    data_root = PROJECT_ROOT / "data"
    train_ds = AllenDetectionDatasetWithId(
        data_root / "processed" / "images",
        data_root / "processed" / "masks",
        data_root / "splits" / "train_ids.txt",
    )
    val_ds = AllenDetectionDatasetWithId(
        data_root / "processed" / "images",
        data_root / "processed" / "masks",
        data_root / "splits" / "val_ids.txt",
    )

    train_prior_cache, train_prior_summary = build_prior_cache(
        dataset=train_ds,
        num_queries=args.num_queries,
        profile_name=args.profile_name,
        candidate_mode=args.candidate_mode,
    )
    val_prior_cache, val_prior_summary = build_prior_cache(
        dataset=val_ds,
        num_queries=args.num_queries,
        profile_name=args.profile_name,
        candidate_mode=args.candidate_mode,
    )
    with open(output_dir / "prior_cache_summary.json", "w") as f:
        json.dump(
            {"train": train_prior_summary, "val": val_prior_summary},
            f,
            indent=2,
        )
    print(f"train prior summary: {train_prior_summary}")
    print(f"val prior summary: {val_prior_summary}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_ids,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_ids,
        drop_last=False,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model, cellfinder = t33.setup_model(
        device=device,
        freeze_backbone=True,
        num_queries=args.num_queries,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )
    criterion, weight_dict = t33.setup_criterion(device, num_classes=2)

    trainable_params = [p for p in cellfinder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_metric = 0.0
    metric_alias_map = {
        "candidate_aligned_f1_0p3": "candidate_aligned_f1@0.3",
        "candidate_aligned_f1_0p5": "candidate_aligned_f1@0.5",
        "candidate_aligned_f1_0p7": "candidate_aligned_f1@0.7",
    }
    metric_name = metric_alias_map.get(args.early_stop_metric, args.early_stop_metric)
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, avg_train_candidates = train_one_epoch_candidate_aware(
            model=model,
            cellfinder=cellfinder,
            criterion=criterion,
            weight_dict=weight_dict,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            prior_cache=train_prior_cache,
            prior_mode=args.prior_mode,
            apply_candidate_mask=apply_candidate_mask,
            max_train_batches=args.max_train_batches,
        )
        val_loss, val_metrics = evaluate_candidate_aware(
            model=model,
            cellfinder=cellfinder,
            criterion=criterion,
            weight_dict=weight_dict,
            dataloader=val_loader,
            device=device,
            prior_cache=val_prior_cache,
            prior_mode=args.prior_mode,
            apply_candidate_mask=apply_candidate_mask,
            max_val_batches=args.max_val_batches,
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train={train_loss:.4f} Val={val_loss:.4f} | "
            f"F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} | "
            f"CandF1@0.3={val_metrics.get('candidate_aligned_f1@0.3', 0):.4f} "
            f"CandF1@0.5={val_metrics.get('candidate_aligned_f1@0.5', 0):.4f} "
            f"CandF1@0.7={val_metrics.get('candidate_aligned_f1@0.7', 0):.4f} "
            f"CandP@0.3={val_metrics.get('candidate_aligned_precision@0.3', 0):.4f} "
            f"CandR@0.3={val_metrics.get('candidate_aligned_recall@0.3', 0):.4f} | "
            f"mAP={val_metrics.get('coco_ap', 0):.4f} AP50={val_metrics.get('coco_ap50', 0):.4f} | "
            f"avg_train_candidates={avg_train_candidates:.1f} "
            f"avg_val_candidates={val_metrics.get('avg_effective_candidates_per_batch', 0):.1f} | "
            f"LR={lr:.6f} | {elapsed:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_candidate_aligned_precision_03": val_metrics.get(
                "candidate_aligned_precision@0.3", 0
            ),
            "val_candidate_aligned_recall_03": val_metrics.get(
                "candidate_aligned_recall@0.3", 0
            ),
            "val_candidate_aligned_f1_03": val_metrics.get(
                "candidate_aligned_f1@0.3", 0
            ),
            "val_candidate_aligned_precision_05": val_metrics.get(
                "candidate_aligned_precision@0.5", 0
            ),
            "val_candidate_aligned_recall_05": val_metrics.get(
                "candidate_aligned_recall@0.5", 0
            ),
            "val_candidate_aligned_f1_05": val_metrics.get(
                "candidate_aligned_f1@0.5", 0
            ),
            "val_candidate_aligned_tp_05": val_metrics.get(
                "candidate_aligned_tp@0.5", 0
            ),
            "val_candidate_aligned_fp_05": val_metrics.get(
                "candidate_aligned_fp@0.5", 0
            ),
            "val_candidate_aligned_fn_05": val_metrics.get(
                "candidate_aligned_fn@0.5", 0
            ),
            "val_candidate_aligned_precision_07": val_metrics.get(
                "candidate_aligned_precision@0.7", 0
            ),
            "val_candidate_aligned_recall_07": val_metrics.get(
                "candidate_aligned_recall@0.7", 0
            ),
            "val_candidate_aligned_f1_07": val_metrics.get(
                "candidate_aligned_f1@0.7", 0
            ),
            "val_candidate_aligned_tp_07": val_metrics.get(
                "candidate_aligned_tp@0.7", 0
            ),
            "val_candidate_aligned_fp_07": val_metrics.get(
                "candidate_aligned_fp@0.7", 0
            ),
            "val_candidate_aligned_fn_07": val_metrics.get(
                "candidate_aligned_fn@0.7", 0
            ),
            "val_candidate_aligned_tp_03": val_metrics.get(
                "candidate_aligned_tp@0.3", 0
            ),
            "val_candidate_aligned_fp_03": val_metrics.get(
                "candidate_aligned_fp@0.3", 0
            ),
            "val_candidate_aligned_fn_03": val_metrics.get(
                "candidate_aligned_fn@0.3", 0
            ),
            "val_coco_ap": val_metrics.get("coco_ap", 0),
            "val_coco_ap50": val_metrics.get("coco_ap50", 0),
            "val_coco_ap75": val_metrics.get("coco_ap75", 0),
            "val_coco_ap95": val_metrics.get("coco_ap95", 0),
            "avg_train_candidates_per_batch": avg_train_candidates,
            "avg_val_candidates_per_batch": val_metrics.get(
                "avg_effective_candidates_per_batch", 0
            ),
            "lr": lr,
            "elapsed": elapsed,
        }
        history.append(record)

        current_metric = val_metrics.get(metric_name, val_metrics.get("f1", 0))
        if current_metric > best_metric:
            best_metric = float(current_metric)
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "cellfinder_state_dict": cellfinder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    f"best_{metric_name}": best_metric,
                    "val_metrics": val_metrics,
                    "candidate_mode": args.candidate_mode,
                    "profile_name": args.profile_name,
                    "prior_mode": args.prior_mode,
                    "apply_candidate_mask": apply_candidate_mask,
                },
                output_dir / "best_cellfinder.pt",
            )
            print(f"  New best {metric_name}: {best_metric:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 72)
    print("Training complete")
    print(f"Best {metric_name}: {best_metric:.4f}")
    print(f"Output: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
