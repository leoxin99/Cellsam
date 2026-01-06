"""
Training script using expanded Allen Cell dataset with augmentation.
50 samples, 539 cells, full Albumentations transforms.
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return 0.5 * self.dice(torch.sigmoid(pred), target) + 0.5 * self.bce(pred, target)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)
        boxes = batch['boxes'].to(device)
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
            gt_mask = masks[i]
            
            with torch.no_grad():
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
            
            pred_masks_list = []
            
            for box_idx in range(min(num_boxes, 20)):
                box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                
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
                )
                pred_masks_list.append(pred_mask.squeeze())
            
            if pred_masks_list:
                pred_combined = torch.stack(pred_masks_list, dim=0).max(dim=0)[0]
                gt_binary = (gt_mask > 0).float()
                loss = criterion(pred_combined.unsqueeze(0), gt_binary.unsqueeze(0))
                batch_losses.append(loss)
        
        if batch_losses:
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            boxes = batch['boxes'].to(device)
            box_counts = batch['box_counts']
            masks = batch['mask'].to(device)
            
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
                
                for box_idx in range(min(num_boxes, 20)):
                    box = img_boxes[box_idx:box_idx+1].unsqueeze(0)
                    
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
                    )
                    pred_masks_list.append(pred_mask.squeeze())
                
                if pred_masks_list:
                    pred_combined = torch.stack(pred_masks_list, dim=0).max(dim=0)[0]
                    gt_binary = (gt_mask > 0).float()
                    
                    loss = criterion(pred_combined.unsqueeze(0), gt_binary.unsqueeze(0))
                    total_loss += loss.item()
                    
                    pred_binary = (torch.sigmoid(pred_combined) > 0.5).float()
                    intersection = (pred_binary * gt_binary).sum()
                    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
                    total_dice += dice.item()
                    
                    num_samples += 1
    
    return total_loss / max(num_samples, 1), total_dice / max(num_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="d:/AI/paper/CellSam/training_pairs_expanded")
    parser.add_argument("--output_dir", type=str, default="d:/AI/paper/CellSam/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = AugmentedAllenDataset(args.data_dir, is_training=True)
    val_dataset = AugmentedAllenDataset(args.data_dir, is_training=False)
    
    # Split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    criterion = CombinedLoss()
    
    print("\n" + "="*60)
    print(f"Training with expanded dataset ({len(train_dataset)} samples)")
    print("="*60)
    
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
        }, output_dir / f"checkpoint_epoch{epoch}.pt")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best! Dice={best_dice:.4f}")
    
    print("\n" + "="*60)
    print(f"Training complete! Best Dice: {best_dice:.4f}")
    print(f"Checkpoints: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
