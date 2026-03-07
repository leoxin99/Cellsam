#!/usr/bin/env python3
"""T33: CellFinder Head-Only Fine-Tuning on Allen Data

Resource-constrained Allen adaptation inspired by CellSAM Stage 1.
Freezes ViT backbone, trains only CellFinder decoder head.

Usage:
  python tools/train_cellfinder.py --seed 42
  python tools/train_cellfinder.py --seed 123
"""

import sys
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source" / "cellSAM"))

from cellSAM.model import get_model
# Import SetCriterion and HungarianMatcher directly from AnchorDETR
# (not via cellSAM package) to avoid absolute import issues in matcher.py
from AnchorDETR.models.anchor_detr import SetCriterion
from AnchorDETR.models.matcher import HungarianMatcher


# ================================================================
# Dataset
# ================================================================

class AllenDetectionDataset(Dataset):
    """Allen cardiomyocyte dataset for detection (mask → boxes)."""

    def __init__(self, image_dir, mask_dir, ids_file, target_size=1024):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size

        with open(ids_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(self.ids)} samples from {ids_file}")

    def __len__(self):
        return len(self.ids)

    def _masks_to_cxcywh(self, instance_mask, H, W):
        """Convert instance mask to normalized [cx, cy, w, h] boxes."""
        from skimage.measure import regionprops
        boxes = []
        for prop in regionprops(instance_mask.astype(int)):
            y1, x1, y2, x2 = prop.bbox
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            boxes.append([cx, cy, w, h])
        return boxes

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = np.load(self.image_dir / f"{img_id}.npy")    # (C, H, W)
        mask = np.load(self.mask_dir / f"{img_id}.npy")     # (H, W)

        if img.ndim == 3 and img.shape[0] in (3, 4, 5):
            C, H, W = img.shape
        else:
            H, W = img.shape[:2]
            img = img.transpose(2, 0, 1)
            C = img.shape[0]

        boxes = self._masks_to_cxcywh(mask, H, W)
        n_boxes = len(boxes)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if n_boxes > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros(n_boxes, dtype=torch.int64)  # single class: "cell" = 0

        target = {
            "labels": labels_tensor,
            "boxes": boxes_tensor,
        }

        return img_tensor, target


def collate_fn(batch):
    """Custom collate for variable-length targets."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# ================================================================
# Model Setup
# ================================================================

def setup_model(device, freeze_backbone=True):
    """Load CellSAM, extract CellFinder, freeze backbone."""
    model = get_model()
    model.adv_mode = True
    model = model.to(device)

    cellfinder = model.cellfinder
    if cellfinder is None:
        raise RuntimeError("CellSAM model has no cellfinder — check get_model()")

    # Freeze backbone (ViT encoder inside CellFinder's SAMBackbone)
    if freeze_backbone:
        if hasattr(cellfinder, 'decode_head') and hasattr(cellfinder.decode_head, 'backbone'):
            for param in cellfinder.decode_head.backbone.parameters():
                param.requires_grad = False
            print("Froze CellFinder backbone (SAMBackbone)")
        else:
            print("WARNING: Could not find decode_head.backbone — freezing all cellfinder backbone params")
            for name, param in cellfinder.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False

    # Trainable param audit
    trainable = {n: p.numel() for n, p in cellfinder.named_parameters() if p.requires_grad}
    total_trainable = sum(trainable.values())
    total_all = sum(p.numel() for p in cellfinder.parameters())
    print(f"\n{'='*60}")
    print(f"CellFinder trainable: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.2f}%)")
    print(f"Trainable components:")
    # Group by top-level module
    groups = {}
    for name, count in trainable.items():
        top = name.split('.')[0]
        groups[top] = groups.get(top, 0) + count
    for g, c in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g}: {c:,}")
    print(f"{'='*60}\n")

    return model, cellfinder


def setup_criterion(device, num_classes=2):
    """Create SetCriterion + HungarianMatcher."""
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
    )
    weight_dict = {
        'loss_ce': 2.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
    }
    # Add aux loss weights (SetCriterion expects these for intermediate layers)
    for i in range(5):  # 6 decoder layers, 5 aux
        weight_dict[f'loss_ce_{i}'] = 2.0
        weight_dict[f'loss_bbox_{i}'] = 5.0
        weight_dict[f'loss_giou_{i}'] = 2.0

    losses = ['labels', 'boxes']
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=0.25,
    )
    criterion.to(device)
    return criterion, weight_dict


# ================================================================
# Training
# ================================================================

def preprocess_for_cellfinder(model, images, device):
    """Preprocess images through CellSAM's sam_bbox_preprocessing.

    This handles the official channel encoding + normalization.
    """
    # Stack into batch, move to device
    imgs = [img.to(device) for img in images]

    # Use official sam_bbox_preprocessing (handles resize, normalization, percentile, etc.)
    with torch.no_grad():
        processed = model.sam_bbox_preprocessing(imgs, percentile=True)
    return processed


def train_one_epoch(model, cellfinder, criterion, weight_dict,
                    dataloader, optimizer, device, epoch):
    """Single training epoch."""
    cellfinder.train()
    total_loss = 0
    n_batches = 0

    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}"):
        # Move targets to device
        targets = [{
            "labels": t["labels"].to(device),
            "boxes": t["boxes"].to(device),
        } for t in targets]

        # Skip empty batches
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        # Preprocess through CellSAM's pipeline
        processed_imgs = preprocess_for_cellfinder(model, images, device)

        # Forward through CellFinder
        outputs = cellfinder(processed_imgs)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * weight_dict.get(k, 1.0)
                     for k in loss_dict if k in weight_dict)

        # Backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(cellfinder.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, cellfinder, criterion, weight_dict, dataloader, device):
    """Validation evaluation — returns avg loss and detection metrics."""
    cellfinder.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc="Eval"):
        targets = [{
            "labels": t["labels"].to(device),
            "boxes": t["boxes"].to(device),
        } for t in targets]

        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        processed_imgs = preprocess_for_cellfinder(model, images, device)
        outputs = cellfinder(processed_imgs)

        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * weight_dict.get(k, 1.0)
                     for k in loss_dict if k in weight_dict)
        total_loss += losses.item()
        n_batches += 1

        # Collect predictions for simple AP calculation
        pred_logits = outputs["pred_logits"].sigmoid()  # [B, N, C]
        pred_boxes = outputs["pred_boxes"]              # [B, N, 4]

        for b in range(pred_logits.shape[0]):
            # Take top-scoring predictions (class 0 = cell)
            scores = pred_logits[b, :, 0]  # cell class scores
            all_preds.append({
                "scores": scores.cpu(),
                "boxes": pred_boxes[b].cpu(),
            })
            all_targets.append({
                "boxes": targets[b]["boxes"].cpu(),
            })

    avg_loss = total_loss / max(n_batches, 1)

    # Simple AP@0.5 calculation
    ap50 = compute_simple_ap(all_preds, all_targets, iou_thresh=0.5)

    return avg_loss, ap50


def compute_simple_ap(preds, targets, iou_thresh=0.5):
    """Simple AP@0.5 — not full COCO AP but sufficient for monitoring."""
    from AnchorDETR.util.box_ops import box_cxcywh_to_xyxy

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for pred, target in zip(preds, targets):
        # Filter by score threshold
        mask = pred["scores"] > 0.3
        pred_boxes = pred["boxes"][mask]
        gt_boxes = target["boxes"]

        if len(gt_boxes) == 0:
            all_fp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            all_fn += len(gt_boxes)
            continue

        # Convert to xyxy
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

        # IoU matrix
        from torchvision.ops import box_iou
        iou_matrix = box_iou(pred_xyxy, gt_xyxy)

        # Greedy matching
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
                    best_iou = iou_matrix[i, j].item()
                    best_j = j
            if best_j >= 0:
                tp += 1
                matched_gt.add(best_j)

        all_tp += tp
        all_fp += len(pred_boxes) - tp
        all_fn += len(gt_boxes) - len(matched_gt)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": all_tp, "fp": all_fp, "fn": all_fn}


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="T33: CellFinder fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "checkpoints" / f"T33_CellFinder_HeadOnly_seed{args.seed}_{timestamp}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T33: CellFinder Allen-Specific Adaptation (Head-Only)")
    print("=" * 60)
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patience: {args.patience}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    data_root = PROJECT_ROOT / "data"
    train_ds = AllenDetectionDataset(
        data_root / "processed" / "images",
        data_root / "processed" / "masks",
        data_root / "splits" / "train_ids.txt",
    )
    val_ds = AllenDetectionDataset(
        data_root / "processed" / "images",
        data_root / "processed" / "masks",
        data_root / "splits" / "val_ids.txt",
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model, cellfinder = setup_model(device, freeze_backbone=True)

    # Criterion
    criterion, weight_dict = setup_criterion(device, num_classes=2)

    # Optimizer — only trainable params
    trainable_params = [p for p in cellfinder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, cellfinder, criterion, weight_dict,
            train_loader, optimizer, device, epoch
        )

        # Eval
        val_loss, val_metrics = evaluate(
            model, cellfinder, criterion, weight_dict,
            val_loader, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | "
              f"R: {val_metrics['recall']:.4f} | LR: {lr:.6f} | {elapsed:.1f}s")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics['f1'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_tp": val_metrics['tp'],
            "val_fp": val_metrics['fp'],
            "val_fn": val_metrics['fn'],
            "lr": lr,
            "elapsed": elapsed,
        }
        history.append(record)

        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "cellfinder_state_dict": cellfinder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "val_metrics": val_metrics,
            }, output_dir / "best_cellfinder.pt")
            print(f"  ★ New best F1: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
