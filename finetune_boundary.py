"""
Fine-tuning script with Boundary Loss for improved edge precision.

Author: Deep Learning Model Optimization Engineer
Date: 2026-01-11

Purpose:
- Load pre-trained model (best_model.pt)
- Fine-tune with Boundary Loss to improve instance-level IoU
- Address the issue: high Dice but low PQ (poor boundary alignment)

Expected improvement:
- Instance IoU should increase from 0.1-0.3 to 0.5+
- PQ@0.5 should become > 0

Usage:
    python finetune_boundary.py --epochs 20 --lr 1e-5
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, get_all_sample_ids
from sklearn.model_selection import train_test_split


# Import losses from train_expanded
from train_expanded import CombinedLoss, DiceLoss, BoundaryLoss


def finetune_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Fine-tune one epoch with boundary loss focus."""
    model.train()
    total_loss = 0.0
    num_samples = 0

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

            # Image encoder (frozen)
            with torch.no_grad():
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)

            cell_losses = []

            # Per-cell loss calculation
            for box_idx in range(min(num_boxes, 20)):
                box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                cell_id = img_cell_ids[box_idx].item()

                if cell_id <= 0:
                    continue

                cell_gt = (gt_mask == cell_id).float()

                if cell_gt.sum() == 0:
                    continue

                # Forward pass through decoder
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
                    low_res_masks, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)

                # Resize GT to 1024x1024 if needed
                if cell_gt.shape != pred_mask.shape:
                    cell_gt = F.interpolate(
                        cell_gt.unsqueeze(0).unsqueeze(0).float(),
                        size=pred_mask.shape,
                        mode='nearest'
                    ).squeeze()

                # Compute combined loss with boundary
                box_coords = img_boxes[box_idx].cpu().numpy()
                loss = criterion(pred_mask, cell_gt, box=box_coords)
                cell_losses.append(loss)

            if cell_losses:
                img_loss = torch.stack(cell_losses).mean()
                batch_losses.append(img_loss)

        if batch_losses:
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_samples += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}" if batch_losses else "N/A"})

    return total_loss / max(num_samples, 1)


def validate(model, dataloader, criterion, device):
    """Validate model with boundary-aware metrics."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            boxes = batch['boxes'].to(device)
            cell_ids = batch['cell_ids'].to(device)
            box_counts = batch['box_counts']
            masks = batch['mask'].to(device)

            for i in range(images.shape[0]):
                img = images[i:i+1]
                num_boxes = box_counts[i].item()

                if num_boxes == 0:
                    continue

                img_boxes = boxes[i, :num_boxes]
                img_cell_ids = cell_ids[i, :num_boxes]
                gt_mask = masks[i]

                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)

                for box_idx in range(min(num_boxes, 20)):
                    box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                    cell_id = img_cell_ids[box_idx].item()

                    if cell_id <= 0:
                        continue

                    cell_gt = (gt_mask == cell_id).float()

                    if cell_gt.sum() == 0:
                        continue

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
                        low_res_masks, size=(1024, 1024),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)

                    if cell_gt.shape != pred_mask.shape:
                        cell_gt = F.interpolate(
                            cell_gt.unsqueeze(0).unsqueeze(0).float(),
                            size=pred_mask.shape,
                            mode='nearest'
                        ).squeeze()

                    # Compute loss
                    box_coords = img_boxes[box_idx].cpu().numpy()
                    loss = criterion(pred_mask, cell_gt, box=box_coords)
                    total_loss += loss.item()

                    # Compute Dice
                    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                    intersection = (pred_binary * cell_gt).sum()
                    dice = (2 * intersection + 1) / (pred_binary.sum() + cell_gt.sum() + 1)
                    total_dice += dice.item()

                    num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)
    avg_dice = total_dice / max(num_samples, 1)

    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Boundary Loss")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--boundary_weight', type=float, default=0.3, help='Weight for boundary loss')
    parser.add_argument('--model_path', type=str, 
                        default='d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt',
                        help='Path to pre-trained model')
    args = parser.parse_args()

    print("="*70)
    print("BOUNDARY LOSS FINE-TUNING")
    print("="*70)
    print(f"Pre-trained model: {args.model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Boundary weight: {args.boundary_weight}")
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading pre-trained model...")
    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Freeze image encoder, only train decoder
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    print("\nLoading dataset...")
    DATA_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
    
    # Get sample IDs directly from TIFF files
    tiff_files = list(DATA_DIR.glob("*.tiff"))
    all_ids = [f.stem for f in tiff_files]
    print(f"Found {len(all_ids)} TIFF files")
    
    if len(all_ids) == 0:
        print("ERROR: No TIFF files found!")
        return
    
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    train_ids = train_ids[:50]  # Use same 50 samples as before
    val_ids = val_ids[:10]

    train_dataset = AugmentedAllenDataset(DATA_DIR, sample_ids=train_ids, is_training=True)
    val_dataset = AugmentedAllenDataset(DATA_DIR, sample_ids=val_ids, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Loss with boundary
    criterion = CombinedLoss(pos_weight=10.0, boundary_weight=args.boundary_weight, use_boundary=True)

    # Optimizer (lower LR for fine-tuning)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=args.lr, weight_decay=1e-4)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Checkpoint dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f"d:/AI/paper/CellSam/checkpoints/boundary_finetune_{timestamp}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoint dir: {ckpt_dir}")

    # Training loop
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = finetune_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  Saved best model (Dice: {best_dice:.4f})")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    print("\n" + "="*70)
    print(f"FINE-TUNING COMPLETE")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Model saved to: {ckpt_dir / 'best_model.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()
