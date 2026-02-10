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
Fine-tuning script with Boundary Loss for improved edge precision.
SIMPLIFIED VERSION - loads directly from raw TIFF files.

Author: Deep Learning Model Optimization Engineer
Date: 2026-01-11

Usage:
    python finetune_boundary_simple.py --epochs 20 --lr 1e-5
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tifffile
from skimage import measure, morphology
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
from cellSAM import get_model

# Channel mapping
CH_BRIGHTFIELD = 0
CH_MASK = 9


class SimpleTiffDataset(Dataset):
    """Simple dataset that loads directly from TIFF files."""
    
    def __init__(self, data_dir, sample_ids, target_size=(1024, 1024), is_training=True):
        self.data_dir = Path(data_dir)
        self.sample_ids = sample_ids
        self.target_size = target_size
        self.is_training = is_training
        
        # Find actual files
        self.file_paths = []
        for sid in sample_ids:
            matching = list(self.data_dir.glob(f"{sid}*.tiff"))
            if matching:
                self.file_paths.append(matching[0])
        
        print(f"Found {len(self.file_paths)} files for {len(sample_ids)} sample IDs")
    
    def __len__(self):
        return len(self.file_paths)
    
    def _normalize(self, img):
        p2, p98 = np.percentile(img, [2, 98])
        if p98 > p2:
            return np.clip((img - p2) / (p98 - p2), 0, 1).astype(np.float32)
        return np.zeros_like(img, dtype=np.float32)
    
    def _get_boxes_and_ids(self, mask):
        """Extract bounding boxes and cell IDs from mask."""
        boxes = []
        cell_ids = []
        
        for region in measure.regionprops(mask):
            y1, x1, y2, x2 = region.bbox
            # Expand box
            h, w = mask.shape
            bh, bw = y2 - y1, x2 - x1
            expand = 0.1
            x1 = max(0, int(x1 - bw * expand))
            y1 = max(0, int(y1 - bh * expand))
            x2 = min(w, int(x2 + bw * expand))
            y2 = min(h, int(y2 + bh * expand))
            
            boxes.append([x1, y1, x2, y2])
            cell_ids.append(region.label)
        
        return boxes, cell_ids
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        with tifffile.TiffFile(path) as tif:
            data = np.squeeze(tif.asarray())
        
        bf = data[CH_BRIGHTFIELD]
        mask = data[CH_MASK].astype(np.int32)
        
        # Normalize and resize
        bf_norm = self._normalize(bf)
        
        # Resize to target size
        from skimage.transform import resize
        bf_resized = resize(bf_norm, self.target_size, preserve_range=True).astype(np.float32)
        mask_resized = resize(mask, self.target_size, order=0, preserve_range=True).astype(np.int32)
        
        # Convert to RGB
        bf_rgb = np.stack([bf_resized, bf_resized, bf_resized], axis=0)
        
        # Get boxes
        boxes, cell_ids = self._get_boxes_and_ids(mask_resized)
        
        # Scale boxes to 1024x1024
        scale_x = 1024 / self.target_size[1]
        scale_y = 1024 / self.target_size[0]
        scaled_boxes = [[int(b[0]*scale_x), int(b[1]*scale_y), 
                         int(b[2]*scale_x), int(b[3]*scale_y)] for b in boxes]
        
        return {
            'image': torch.from_numpy(bf_rgb),
            'mask': torch.from_numpy(mask_resized),
            'boxes': scaled_boxes,
            'cell_ids': cell_ids
        }


def collate_fn(batch):
    """Collate function for variable number of boxes."""
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    
    max_boxes = max(len(b['boxes']) for b in batch) if batch else 0
    max_boxes = max(max_boxes, 1)
    
    batch_size = len(batch)
    boxes = torch.zeros(batch_size, max_boxes, 4)
    cell_ids = torch.zeros(batch_size, max_boxes, dtype=torch.int64)
    box_counts = []
    
    for i, b in enumerate(batch):
        n = len(b['boxes'])
        box_counts.append(n)
        if n > 0:
            boxes[i, :n] = torch.tensor(b['boxes'])
            cell_ids[i, :n] = torch.tensor(b['cell_ids'])
    
    return {
        'image': images,
        'mask': masks,
        'boxes': boxes,
        'cell_ids': cell_ids,
        'box_counts': torch.tensor(box_counts)
    }


class CombinedBoundaryLoss(nn.Module):
    """Dice + BCE + Boundary loss."""
    
    def __init__(self, boundary_weight=0.3):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def get_boundary(self, mask):
        """Get boundary pixels."""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        
        mask_binary = (mask_np > 0.5).astype(np.float32)
        eroded = ndimage.binary_erosion(mask_binary, morphology.disk(2)).astype(np.float32)
        return mask_binary - eroded
    
    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        
        # Dice
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(pred_logits.reshape(-1), target.reshape(-1).float())
        
        base_loss = 0.5 * dice + 0.5 * bce
        
        # Boundary loss
        if target.sum() > 0:
            try:
                boundary = self.get_boundary(target)
                boundary_tensor = torch.from_numpy(boundary).to(pred.device).float()
                
                if boundary_tensor.sum() > 0:
                    boundary_pred = pred[boundary_tensor > 0]
                    boundary_target = target[boundary_tensor > 0].float()
                    boundary_loss = F.binary_cross_entropy(boundary_pred, boundary_target)
                    
                    return (1 - self.boundary_weight) * base_loss + self.boundary_weight * boundary_loss
            except:
                pass
        
        return base_loss


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    n_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        boxes = batch['boxes'].to(device)
        cell_ids = batch['cell_ids'].to(device)
        box_counts = batch['box_counts']
        
        optimizer.zero_grad()
        batch_losses = []
        
        for i in range(images.shape[0]):
            img = images[i:i+1]
            mask = masks[i]
            num_boxes = box_counts[i].item()
            
            if num_boxes == 0:
                continue
            
            with torch.no_grad():
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
            
            for j in range(min(num_boxes, 15)):
                box = boxes[i, j:j+1].unsqueeze(0)
                cell_id = cell_ids[i, j].item()
                
                if cell_id <= 0:
                    continue
                
                cell_gt = (mask == cell_id).float()
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
                
                pred = F.interpolate(
                    low_res_masks, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)
                
                if cell_gt.shape != pred.shape:
                    cell_gt = F.interpolate(
                        cell_gt.unsqueeze(0).unsqueeze(0),
                        size=pred.shape, mode='nearest'
                    ).squeeze()
                
                loss = criterion(pred, cell_gt)
                batch_losses.append(loss)
        
        if batch_losses:
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_samples += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / max(n_samples, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_dice = 0
    n_cells = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes'].to(device)
            cell_ids = batch['cell_ids'].to(device)
            box_counts = batch['box_counts']
            
            for i in range(images.shape[0]):
                img = images[i:i+1]
                mask = masks[i]
                num_boxes = box_counts[i].item()
                
                if num_boxes == 0:
                    continue
                
                img_preprocessed = model.sam_preprocess(img)
                embedding = model.model.image_encoder(img_preprocessed)
                
                for j in range(min(num_boxes, 15)):
                    box = boxes[i, j:j+1].unsqueeze(0)
                    cell_id = cell_ids[i, j].item()
                    
                    if cell_id <= 0:
                        continue
                    
                    cell_gt = (mask == cell_id).float()
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
                    
                    pred = F.interpolate(
                        low_res_masks, size=(1024, 1024),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    if cell_gt.shape != pred.shape:
                        cell_gt = F.interpolate(
                            cell_gt.unsqueeze(0).unsqueeze(0),
                            size=pred.shape, mode='nearest'
                        ).squeeze()
                    
                    pred_binary = (torch.sigmoid(pred) > 0.5).float()
                    intersection = (pred_binary * cell_gt).sum()
                    dice = (2 * intersection + 1) / (pred_binary.sum() + cell_gt.sum() + 1)
                    
                    total_dice += dice.item()
                    n_cells += 1
    
    return total_dice / max(n_cells, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--boundary_weight', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_path', type=str,
                        default='d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt')
    args = parser.parse_args()
    
    print("="*70)
    print("BOUNDARY LOSS FINE-TUNING (Simplified)")
    print("="*70)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Boundary weight: {args.boundary_weight}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Freeze encoder
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False
    
    # Dataset
    DATA_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
    tiff_files = sorted(DATA_DIR.glob("*.tiff"))
    all_ids = [f.stem[:40] for f in tiff_files]
    
    print(f"\nFound {len(all_ids)} samples")
    
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    train_ids = train_ids[:40]
    val_ids = val_ids[:8]
    
    train_dataset = SimpleTiffDataset(DATA_DIR, train_ids, is_training=True)
    val_dataset = SimpleTiffDataset(DATA_DIR, val_ids, is_training=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    criterion = CombinedBoundaryLoss(boundary_weight=args.boundary_weight)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f"d:/AI/paper/CellSam/checkpoints/boundary_{timestamp}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")
    
    best_dice = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_dice = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  Saved best model!")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
    
    print("\n" + "="*70)
    print(f"DONE! Best Val Dice: {best_dice:.4f}")
    print(f"Model: {ckpt_dir / 'best_model.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()
