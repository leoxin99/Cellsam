"""
Training script for fine-tuning CellSAM on cardiomyocyte images.

Key features:
- Freezes Image Encoder (ViT backbone)
- Trains only Mask Decoder
- Bypasses CellFinder (uses external DAPI-derived prompts)
- Uses Dice + Focal loss

Usage:
    python train_cardiomyocyte.py --data_dir /path/to/processed_data --output_dir /path/to/checkpoints

Requirements:
    pip install torch torchvision numpy tqdm
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add CellSAM to path
sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))

from cellSAM import get_model
from cardiomyocyte_dataset import CardiomyocyteDataset, collate_fn


# ============== Loss Functions ==============

class DiceLoss(nn.Module):
    """Dice loss for binary/multi-class segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, H, W) predicted masks (after sigmoid)
            target: (B, H, W) ground truth masks (binary)
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1).float()
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, H, W) predicted masks (logits, before sigmoid)
            target: (B, H, W) ground truth masks (binary)
        """
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        pt = target * pred_prob + (1 - target) * (1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal loss."""
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred is logits, need sigmoid for dice
        pred_sigmoid = torch.sigmoid(pred)
        
        dice = self.dice_loss(pred_sigmoid, target)
        focal = self.focal_loss(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


# ============== Training Functions ==============

def freeze_image_encoder(model: nn.Module) -> None:
    """Freeze the SAM image encoder (ViT backbone)."""
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False
    print("Frozen: model.model.image_encoder")


def freeze_cellfinder(model: nn.Module) -> None:
    """Freeze the CellFinder detection head."""
    if hasattr(model, 'cellfinder'):
        for param in model.cellfinder.parameters():
            param.requires_grad = False
        print("Frozen: cellfinder")


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)  # (B, 3, H, W)
        boxes = batch['boxes'].to(device)    # (B, N, 4)
        box_counts = batch['box_counts']      # (B,)
        masks = batch['mask'].to(device)     # (B, H, W)
        
        optimizer.zero_grad()
        
        # Forward pass with external prompts (bypassing CellFinder)
        # We need to call SAM's mask decoder directly with the boxes
        
        batch_losses = []
        
        for i in range(images.shape[0]):
            img = images[i:i+1]  # (1, 3, H, W)
            num_boxes = box_counts[i].item()
            
            if num_boxes == 0:
                continue
            
            img_boxes = boxes[i, :num_boxes]  # (N, 4)
            gt_mask = masks[i]  # (H, W)
            
            # Get image embedding
            with torch.no_grad():
                # Preprocess image for SAM
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
            
            # Run mask decoder for each box
            pred_masks_list = []
            
            for box_idx in range(num_boxes):
                box = img_boxes[box_idx:box_idx+1].unsqueeze(0)  # (1, 1, 4)
                
                # SAM expects boxes in specific format
                sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None,
                )
                
                low_res_masks, iou_predictions = model.model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # Upscale mask to original size
                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_mask.shape[0], gt_mask.shape[1]),
                    mode='bilinear',
                    align_corners=False
                )
                
                pred_masks_list.append(pred_mask.squeeze())
            
            if pred_masks_list:
                # Combine predictions (take max across all boxes)
                pred_combined = torch.stack(pred_masks_list, dim=0).max(dim=0)[0]
                
                # Binary target
                gt_binary = (gt_mask > 0).float()
                
                # Compute loss
                loss = criterion(pred_combined.unsqueeze(0), gt_binary.unsqueeze(0))
                batch_losses.append(loss)
        
        if batch_losses:
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": batch_loss.item()})
    
    avg_loss = total_loss / max(num_batches, 1)
    return {"train_loss": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Run validation."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            boxes = batch['boxes'].to(device)
            box_counts = batch['box_counts']
            masks = batch['mask'].to(device)
            
            # Similar to training but without gradients
            for i in range(images.shape[0]):
                img = images[i:i+1]
                num_boxes = box_counts[i].item()
                
                if num_boxes == 0:
                    continue
                
                img_boxes = boxes[i, :num_boxes]
                gt_mask = masks[i]
                
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
                
                pred_masks_list = []
                
                for box_idx in range(min(num_boxes, 10)):  # Limit for speed
                    box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                    
                    sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                    
                    low_res_masks, _ = model.model.mask_decoder(
                        image_embeddings=embedding,
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(gt_mask.shape[0], gt_mask.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    pred_masks_list.append(pred_mask.squeeze())
                
                if pred_masks_list:
                    pred_combined = torch.stack(pred_masks_list, dim=0).max(dim=0)[0]
                    gt_binary = (gt_mask > 0).float()
                    
                    loss = criterion(pred_combined.unsqueeze(0), gt_binary.unsqueeze(0))
                    total_loss += loss.item()
                    num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return {"val_loss": avg_loss}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: str,
    filename: str = "checkpoint.pt"
) -> str:
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train CellSAM on cardiomyocyte data")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Model arguments
    parser.add_argument("--freeze_encoder", action="store_true", default=True, help="Freeze image encoder")
    parser.add_argument("--freeze_cellfinder", action="store_true", default=True, help="Freeze CellFinder")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model
    print("Loading CellSAM model...")
    model = get_model()
    model = model.to(device)
    
    # Freeze components
    if args.freeze_encoder:
        freeze_image_encoder(model)
    if args.freeze_cellfinder:
        freeze_cellfinder(model)
    
    # Print parameter counts
    param_counts = count_parameters(model)
    print(f"Parameters: {param_counts}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CardiomyocyteDataset(args.data_dir)
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        return
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    # Setup optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup loss
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, val_loss={val_metrics['val_loss']:.4f}")
        
        # Save checkpoint
        metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        save_checkpoint(model, optimizer, epoch, metrics, output_dir, f"checkpoint_epoch{epoch}.pt")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(model, optimizer, epoch, metrics, output_dir, "best_model.pt")
            print(f"  New best model saved!")
    
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
