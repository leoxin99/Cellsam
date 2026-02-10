# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Training script using expanded Allen Cell dataset with augmentation.
Features:
- Proper train/val split (file-level, not random_split on augmented data)
- Learning rate scheduler (CosineAnnealingWarmRestarts)
- Mixed precision training (AMP)
- Early stopping
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, get_all_sample_ids


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """Dice loss with optional mask for region-specific computation."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        pred = pred.contiguous().reshape(-1)
        target = target.contiguous().reshape(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: focuses training on pixels near cell edges.
    
    This addresses the issue of high Dice but low instance IoU by emphasizing
    correct boundary prediction.
    
    Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
    """
    def __init__(self, boundary_width=3):
        super().__init__()
        self.boundary_width = boundary_width
    
    def get_boundary_mask(self, mask):
        """Extract boundary pixels from a mask using morphological operations."""
        # Convert to numpy for morphological operations
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        
        from scipy import ndimage
        from skimage import morphology
        
        # Ensure binary
        mask_binary = (mask_np > 0.5).astype(np.float32)
        
        # Erode to find interior
        struct = morphology.disk(self.boundary_width)
        eroded = ndimage.binary_erosion(mask_binary, struct).astype(np.float32)
        
        # Boundary = mask - eroded
        boundary = mask_binary - eroded
        
        return boundary
    
    def forward(self, pred, target):
        """
        Compute boundary-focused loss.
        
        Args:
            pred: (H, W) or (B, H, W) prediction probabilities (after sigmoid)
            target: (H, W) or (B, H, W) binary ground truth
        """
        # Get boundary of target
        if target.dim() == 2:
            boundary_mask = self.get_boundary_mask(target)
        else:
            # Batch processing
            boundaries = []
            for i in range(target.shape[0]):
                boundaries.append(self.get_boundary_mask(target[i]))
            boundary_mask = np.stack(boundaries)
        
        boundary_tensor = torch.from_numpy(boundary_mask).to(pred.device).float()
        
        # Weighted BCE on boundary region
        # Higher weight on boundary pixels
        n_boundary = boundary_tensor.sum()
        n_total = boundary_tensor.numel()
        
        if n_boundary > 0:
            # Loss on boundary pixels
            boundary_pred = pred[boundary_tensor > 0]
            boundary_target = target[boundary_tensor > 0].float()
            
            boundary_bce = F.binary_cross_entropy(
                boundary_pred.reshape(-1),
                boundary_target.reshape(-1),
                reduction='mean'
            )
            
            # Dice on boundary region
            intersection = (boundary_pred * boundary_target).sum()
            boundary_dice = 1 - (2. * intersection + 1) / (boundary_pred.sum() + boundary_target.sum() + 1)
            
            return 0.5 * boundary_bce + 0.5 * boundary_dice
        else:
            return torch.tensor(0.0, device=pred.device)


class CombinedLoss(nn.Module):
    """Combined Dice + BCE + Boundary loss with class imbalance handling."""
    def __init__(self, pos_weight=10.0, boundary_weight=0.3, use_boundary=True):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(boundary_width=3)
        # pos_weight balances foreground vs background in BCE
        self.pos_weight = pos_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary

    def forward(self, pred, target, box=None):
        """
        Compute loss within bounding box region to handle class imbalance.
        Args:
            pred: (H, W) prediction logits
            target: (H, W) binary ground truth
            box: [x1, y1, x2, y2] bounding box (optional)
        """
        if box is not None:
            # Compute loss only within expanded bounding box
            x1, y1, x2, y2 = box
            h, w = pred.shape[-2:]
            # Expand box by 20% for context
            bw, bh = x2 - x1, y2 - y1
            expand = 0.2
            x1 = max(0, int(x1 - bw * expand))
            y1 = max(0, int(y1 - bh * expand))
            x2 = min(w, int(x2 + bw * expand))
            y2 = min(h, int(y2 + bh * expand))

            pred_box = pred[..., y1:y2, x1:x2]
            target_box = target[..., y1:y2, x1:x2]
        else:
            pred_box = pred
            target_box = target

        # Compute class ratio for dynamic pos_weight
        n_pos = target_box.sum()
        n_neg = target_box.numel() - n_pos
        if n_pos > 0:
            dyn_pos_weight = min(n_neg / n_pos, self.pos_weight)
        else:
            dyn_pos_weight = self.pos_weight

        # BCE with pos_weight
        pos_weight_tensor = torch.as_tensor(dyn_pos_weight, dtype=pred.dtype, device=pred.device)
        bce = F.binary_cross_entropy_with_logits(
            pred_box.reshape(-1),
            target_box.reshape(-1).float(),
            pos_weight=pos_weight_tensor
        )

        # Dice loss
        pred_sigmoid = torch.sigmoid(pred_box)
        dice = self.dice(pred_sigmoid, target_box)

        # Base loss
        base_loss = 0.5 * dice + 0.5 * bce
        
        # Boundary loss (optional, helps with edge precision)
        if self.use_boundary and n_pos > 0:
            try:
                boundary_loss = self.boundary(pred_sigmoid, target_box)
                total_loss = (1 - self.boundary_weight) * base_loss + self.boundary_weight * boundary_loss
            except Exception:
                # Fallback if boundary computation fails
                total_loss = base_loss
        else:
            total_loss = base_loss

        return total_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    """Train one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['image'].to(device)
        boxes = batch['boxes'].to(device)
        cell_ids = batch['cell_ids'].to(device)
        box_counts = batch['box_counts']
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        batch_losses = []

        for i in range(images.shape[0]):
            img = images[i:i+1]
            num_boxes = box_counts[i].item()

            if num_boxes == 0:
                continue

            img_boxes = boxes[i, :num_boxes]
            img_cell_ids = cell_ids[i, :num_boxes]
            gt_mask = masks[i]

            # Image encoder (frozen, no grad) - use AMP for speed
            with torch.no_grad():
                if use_amp:
                    with autocast():
                        img_preprocessed = model.sam_preprocess(img)
                        embedding = model.model.image_encoder(img_preprocessed)
                else:
                    img_preprocessed = model.sam_preprocess(img)
                    embedding = model.model.image_encoder(img_preprocessed)

            cell_losses = []

            # Per-cell loss calculation
            for box_idx in range(min(num_boxes, 20)):
                box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                cell_id = img_cell_ids[box_idx].item()

                if use_amp:
                    with autocast():
                        sparse_emb, dense_emb = model.model.prompt_encoder(
                            points=None, boxes=box, masks=None
                        )

                        low_res_masks, _ = model.model.mask_decoder(
                            image_embeddings=embedding,
                            image_pe=model.model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )

                        pred_mask = F.interpolate(
                            low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
                            mode='bilinear', align_corners=False
                        ).squeeze()

                        gt_cell_mask = (gt_mask == cell_id).float()
                        # Pass box for region-based loss to handle class imbalance
                        box_coords = img_boxes[box_idx].tolist()
                        cell_loss = criterion(pred_mask, gt_cell_mask, box=box_coords)
                else:
                    sparse_emb, dense_emb = model.model.prompt_encoder(
                        points=None, boxes=box, masks=None
                    )

                    low_res_masks, _ = model.model.mask_decoder(
                        image_embeddings=embedding,
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )

                    pred_mask = F.interpolate(
                        low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
                        mode='bilinear', align_corners=False
                    ).squeeze()

                    gt_cell_mask = (gt_mask == cell_id).float()
                    # Pass box for region-based loss to handle class imbalance
                    box_coords = img_boxes[box_idx].tolist()
                    cell_loss = criterion(pred_mask, gt_cell_mask, box=box_coords)

                cell_losses.append(cell_loss)

            if cell_losses:
                img_loss = torch.stack(cell_losses).mean()
                batch_losses.append(img_loss)

        if batch_losses:
            batch_loss = torch.stack(batch_losses).mean()

            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validation with per-cell evaluation."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_cells = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            boxes = batch['boxes'].to(device)
            cell_ids = batch['cell_ids'].to(device)  # Added
            box_counts = batch['box_counts']
            masks = batch['mask'].to(device)
            
            for i in range(images.shape[0]):
                img = images[i:i+1]
                num_boxes = box_counts[i].item()
                
                if num_boxes == 0:
                    continue
                
                img_boxes = boxes[i, :num_boxes]
                img_cell_ids = cell_ids[i, :num_boxes]  # Added
                gt_mask = masks[i]
                
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
                
                for box_idx in range(min(num_boxes, 20)):
                    box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                    cell_id = img_cell_ids[box_idx].item()
                    
                    sparse_emb, dense_emb = model.model.prompt_encoder(
                        points=None, boxes=box, masks=None
                    )
                    
                    low_res_masks, _ = model.model.mask_decoder(
                        image_embeddings=embedding,
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    
                    pred_mask = F.interpolate(
                        low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
                        mode='bilinear', align_corners=False
                    ).squeeze()
                    
                    # Per-cell GT
                    gt_cell_mask = (gt_mask == cell_id).float()

                    # Loss (with box for consistent calculation)
                    box_coords = img_boxes[box_idx].tolist()
                    loss = criterion(pred_mask, gt_cell_mask, box=box_coords)
                    total_loss += loss.item()
                    
                    # Dice per cell
                    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                    intersection = (pred_binary * gt_cell_mask).sum()
                    dice = (2 * intersection) / (pred_binary.sum() + gt_cell_mask.sum() + 1e-8)
                    total_dice += dice.item()
                    
                    num_cells += 1
    
    return total_loss / max(num_cells, 1), total_dice / max(num_cells, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="d:/AI/paper/CellSam/data/processed")
    parser.add_argument("--output_dir", type=str, default="d:/AI/paper/CellSam/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    args = parser.parse_args()

    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"expanded_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading CellSAM...")
    model = get_model()
    model = model.to(device)

    # Freeze encoder
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # ========== FIXED: Proper train/val split at file level ==========
    print("\nLoading datasets...")
    all_sample_ids = get_all_sample_ids(args.data_dir)
    print(f"Total samples found: {len(all_sample_ids)}")

    if len(all_sample_ids) == 0:
        print(f"ERROR: No samples found in {args.data_dir}")
        print("Please run: python data/scripts/extract_expanded_pairs.py")
        return

    # Split sample IDs FIRST, then create separate datasets
    train_ids, val_ids = train_test_split(
        all_sample_ids,
        test_size=args.val_split,
        random_state=args.seed
    )

    print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

    # Create datasets with their own transforms
    train_dataset = AugmentedAllenDataset(
        data_dir=args.data_dir,
        is_training=True,  # With augmentation
        sample_ids=train_ids
    )
    val_dataset = AugmentedAllenDataset(
        data_dir=args.data_dir,
        is_training=False,  # No augmentation
        sample_ids=val_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    else:
        scheduler = None
        print("Scheduler: None")

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    criterion = CombinedLoss()

    print("\n" + "="*60)
    print(f"Training Configuration")
    print("="*60)
    print(f"  Data dir: {args.data_dir}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Output: {output_dir}")
    print("="*60)

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            if args.scheduler == "cosine":
                scheduler.step()
            elif args.scheduler == "plateau":
                scheduler.step(val_dice)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_dice={val_dice:.4f}, lr={current_lr:.2e}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'train_ids': train_ids,
            'val_ids': val_ids,
        }, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best! Dice={best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Checkpoints: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
