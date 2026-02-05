"""
Unified training script for CellSAM.
Supports config-driven training for base model and fine-tuning.

Usage:
    python src/train.py --config src/config/base.yaml
    python src/train.py --config src/config/boundary.yaml
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from scipy import ndimage

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from losses.combined import CombinedLoss
from adapters import IndependentChannelAdapter, LightweightChannelAdapter


def compute_pq(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Compute Panoptic Quality (PQ) for instance segmentation.
    
    PQ = SQ × RQ
    - SQ (Segmentation Quality): average IoU of matched instances  
    - RQ (Recognition Quality): TP / (TP + 0.5*FP + 0.5*FN)
    
    Returns: pq, sq, rq
    """
    # Label connected components in prediction
    pred_binary = (pred_mask > 0.5).astype(np.int32)
    pred_labeled, n_pred = ndimage.label(pred_binary)
    
    # Get unique GT labels
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels > 0]
    n_gt = len(gt_labels)
    
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0
    
    # Match predictions to GT using IoU
    matched_gt = set()
    matched_ious = []
    
    for pred_id in range(1, n_pred + 1):
        pred_region = (pred_labeled == pred_id)
        best_iou = 0
        best_gt_id = -1
        
        for gt_id in gt_labels:
            if gt_id in matched_gt:
                continue
            gt_region = (gt_mask == gt_id)
            
            intersection = np.logical_and(pred_region, gt_region).sum()
            union = np.logical_or(pred_region, gt_region).sum()
            iou = intersection / (union + 1e-8)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_id)
            matched_ious.append(best_iou)
    
    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp
    
    sq = np.mean(matched_ious) if matched_ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq
    
    return pq, sq, rq


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train and validation dataloaders using fixed splits."""
    train_ids = load_split_ids("train", config['data']['splits_dir'])
    val_ids = load_split_ids("val", config['data']['splits_dir'])
    
    # Get flags from config
    use_bf_only = config['data'].get('use_bf_only', False)
    use_semantic_mapping = config['data'].get('use_semantic_mapping', False)
    
    train_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=True,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=train_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping
    )
    
    val_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=False,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=val_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def create_model(config: dict, device):
    """Create or load CellSAM model and optional channel adapter."""
    model = get_model()
    
    # Load checkpoint if specified
    if config['model']['checkpoint']:
        checkpoint_path = config['model']['checkpoint']
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle both dict format and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Freeze layers as specified
    if config['model']['freeze_encoder']:
        for param in model.model.image_encoder.parameters():
            param.requires_grad = False
        print("Froze image encoder")
    
    if config['model']['freeze_decoder']:
        for param in model.model.mask_decoder.parameters():
            param.requires_grad = False
        print("Froze mask decoder")
    
    # Create channel adapter if enabled
    adapter = None
    if config['model'].get('use_adapter', False):
        adapter_config = config['model'].get('adapter', {})
        adapter_type = adapter_config.get('type', 'independent')
        
        if adapter_type == 'independent':
            adapter = IndependentChannelAdapter(
                kernel_size=adapter_config.get('kernel_size', 3),
                use_relu=adapter_config.get('use_relu', True)
            )
        else:
            adapter = LightweightChannelAdapter()
        
        adapter = adapter.to(device)
        print(f"Created {adapter_type} channel adapter ({adapter.get_param_count()} params)")
    
    return model.to(device), adapter


def create_optimizer(model, config: dict, adapter=None):
    """Create optimizer and scheduler."""
    # Collect trainable parameters from model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Add adapter parameters if present
    if adapter is not None:
        adapter_params = list(adapter.parameters())
        trainable_params.extend(adapter_params)
        print(f"Adapter parameters: {sum(p.numel() for p in adapter_params):,}")
    
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['warmup_epochs'],
        T_mult=2
    )
    
    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, adapter=None, box_expand=0.1):
    """Train one epoch with optional mixed precision (AMP) and instance-level training.
    
    Key improvements:
    - box_expand: Constrain pred/target to expanded box region
    - Uses cell_id from batch to get specific cell mask (not entire semantic mask)
    """
    model.train()
    if adapter is not None:
        adapter.train()
    
    total_loss = 0
    num_batches = 0
    use_amp = scaler is not None
    
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        boxes = batch['boxes']
        cell_ids = batch.get('cell_ids', None)  # List of cell IDs per sample
        
        optimizer.zero_grad()
        
        # Apply channel adapter if present (TRAINABLE - keep gradients)
        if adapter is not None:
            images = adapter(images)
        
        # Preprocess images and get embeddings (frozen encoder - no grad needed)
        with torch.no_grad():
            if use_amp:
                with autocast():
                    img_preprocessed = model.sam_preprocess(images)
                    image_embedding = model.model.image_encoder(img_preprocessed)
            else:
                img_preprocessed = model.sam_preprocess(images)
                image_embedding = model.model.image_encoder(img_preprocessed)
        
        batch_loss = 0
        num_cells = 0
        
        for i in range(images.shape[0]):
            sample_boxes = boxes[i]
            sample_mask = masks[i]
            sample_cell_ids = cell_ids[i] if cell_ids is not None else None
            
            for j, box in enumerate(sample_boxes):
                if box.sum() == 0:
                    continue
                
                # Get specific cell ID for this box (instance-level training)
                if sample_cell_ids is not None and j < len(sample_cell_ids):
                    cell_id = sample_cell_ids[j].item() if hasattr(sample_cell_ids[j], 'item') else sample_cell_ids[j]
                else:
                    cell_id = None
                
                box_tensor = torch.tensor([box.tolist()], dtype=torch.float32).unsqueeze(0).to(device)
                
                try:
                    if use_amp:
                        with autocast():
                            sparse_emb, dense_emb = model.model.prompt_encoder(
                                points=None, boxes=box_tensor, masks=None
                            )
                            low_res_masks, _ = model.model.mask_decoder(
                                image_embeddings=image_embedding[i:i+1],
                                image_pe=model.model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=False,
                            )
                            
                            pred_mask = F.interpolate(
                                low_res_masks,
                                size=(1024, 1024),
                                mode="bilinear",
                                align_corners=False
                            )[0, 0]
                            
                            # Instance-level target: only this cell, not entire mask
                            if cell_id is not None:
                                target = (sample_mask == cell_id).float()
                            else:
                                target = (sample_mask > 0).float()
                            
                            # Box clipping for both pred and target
                            x1, y1, x2, y2 = [int(b) for b in box.tolist()]
                            h, w = pred_mask.shape
                            bw, bh = x2 - x1, y2 - y1
                            x1_clip = max(0, int(x1 - bw * box_expand))
                            y1_clip = max(0, int(y1 - bh * box_expand))
                            x2_clip = min(w, int(x2 + bw * box_expand))
                            y2_clip = min(h, int(y2 + bh * box_expand))
                            
                            # Zero out predictions outside box region
                            pred_clipped = torch.zeros_like(pred_mask)
                            pred_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = pred_mask[y1_clip:y2_clip, x1_clip:x2_clip]
                            
                            target_clipped = torch.zeros_like(target)
                            target_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = target[y1_clip:y2_clip, x1_clip:x2_clip]
                            
                            loss = criterion(pred_clipped, target_clipped, box=box.tolist())
                    else:
                        sparse_emb, dense_emb = model.model.prompt_encoder(
                            points=None, boxes=box_tensor, masks=None
                        )
                        low_res_masks, _ = model.model.mask_decoder(
                            image_embeddings=image_embedding[i:i+1],
                            image_pe=model.model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )
                        
                        pred_mask = F.interpolate(
                            low_res_masks,
                            size=(1024, 1024),
                            mode="bilinear",
                            align_corners=False
                        )[0, 0]
                        
                        # Instance-level target: only this cell, not entire mask
                        if cell_id is not None:
                            target = (sample_mask == cell_id).float()
                        else:
                            target = (sample_mask > 0).float()
                        
                        # Box clipping for both pred and target
                        x1, y1, x2, y2 = [int(b) for b in box.tolist()]
                        h, w = pred_mask.shape
                        bw, bh = x2 - x1, y2 - y1
                        x1_clip = max(0, int(x1 - bw * box_expand))
                        y1_clip = max(0, int(y1 - bh * box_expand))
                        x2_clip = min(w, int(x2 + bw * box_expand))
                        y2_clip = min(h, int(y2 + bh * box_expand))
                        
                        # Zero out predictions outside box region
                        pred_clipped = torch.zeros_like(pred_mask)
                        pred_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = pred_mask[y1_clip:y2_clip, x1_clip:x2_clip]
                        
                        target_clipped = torch.zeros_like(target)
                        target_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = target[y1_clip:y2_clip, x1_clip:x2_clip]
                        
                        loss = criterion(pred_clipped, target_clipped, box=box.tolist())
                    
                    batch_loss += loss
                    num_cells += 1
                    
                except Exception as e:
                    continue
        
        if num_cells > 0:
            avg_loss = batch_loss / num_cells
            
            if use_amp:
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                # Clip gradients for both model and adapter
                all_params = list(model.parameters())
                if adapter is not None:
                    all_params.extend(list(adapter.parameters()))
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                avg_loss.backward()
                # Clip gradients for both model and adapter
                all_params = list(model.parameters())
                if adapter is not None:
                    all_params.extend(list(adapter.parameters()))
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
            
            total_loss += avg_loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device, adapter=None, use_pq=False, box_expand=0.1):
    """Validation with instance-level Dice and optionally PQ score computation.
    
    Key improvements:
    - Instance-level Dice: compute Dice per cell, not semantic
    - box_expand: constrain predictions to box region
    """
    model.eval()
    if adapter is not None:
        adapter.eval()
    
    total_instance_dice = 0
    total_semantic_dice = 0
    total_pq = 0
    num_cells = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes']
            cell_ids = batch.get('cell_ids', None)
            
            # Apply channel adapter if present
            if adapter is not None:
                images = adapter(images)
            
            img_preprocessed = model.sam_preprocess(images)
            image_embedding = model.model.image_encoder(img_preprocessed)
            
            for i in range(images.shape[0]):
                sample_boxes = boxes[i]
                sample_mask = masks[i]
                sample_cell_ids = cell_ids[i] if cell_ids is not None else None
                combined_pred = torch.zeros_like(sample_mask, dtype=torch.float32)
                
                for j, box in enumerate(sample_boxes):
                    if box.sum() == 0:
                        continue
                    
                    # Get cell ID
                    if sample_cell_ids is not None and j < len(sample_cell_ids):
                        cell_id = sample_cell_ids[j].item() if hasattr(sample_cell_ids[j], 'item') else sample_cell_ids[j]
                    else:
                        cell_id = None
                    
                    box_tensor = torch.tensor([box.tolist()], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    try:
                        sparse_emb, dense_emb = model.model.prompt_encoder(
                            points=None, boxes=box_tensor, masks=None
                        )
                        low_res_masks, _ = model.model.mask_decoder(
                            image_embeddings=image_embedding[i:i+1],
                            image_pe=model.model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )
                        
                        pred_mask = F.interpolate(
                            low_res_masks,
                            size=(1024, 1024),
                            mode="bilinear",
                            align_corners=False
                        )[0, 0]
                        
                        pred_sigmoid = torch.sigmoid(pred_mask)
                        
                        # Box clipping
                        x1, y1, x2, y2 = [int(b) for b in box.tolist()]
                        h, w = pred_mask.shape
                        bw, bh = x2 - x1, y2 - y1
                        x1_clip = max(0, int(x1 - bw * box_expand))
                        y1_clip = max(0, int(y1 - bh * box_expand))
                        x2_clip = min(w, int(x2 + bw * box_expand))
                        y2_clip = min(h, int(y2 + bh * box_expand))
                        
                        pred_clipped = torch.zeros_like(pred_sigmoid)
                        pred_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = pred_sigmoid[y1_clip:y2_clip, x1_clip:x2_clip]
                        
                        # Compute instance-level Dice
                        if cell_id is not None and cell_id > 0:
                            target = (sample_mask == cell_id).float()
                            pred_binary = (pred_clipped > 0.5).float()
                            
                            intersection = (pred_binary * target).sum()
                            dice = (2 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
                            total_instance_dice += dice.item()
                            num_cells += 1
                        
                        combined_pred = torch.maximum(combined_pred, pred_clipped)
                        
                    except Exception:
                        continue
                
                # Semantic Dice (for backward compatibility logging)
                pred_binary = (combined_pred > 0.5).float()
                target_binary = (sample_mask > 0).float()
                intersection = (pred_binary * target_binary).sum()
                semantic_dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
                total_semantic_dice += semantic_dice.item()
                
                # Compute PQ if enabled
                if use_pq:
                    pred_np = combined_pred.cpu().numpy()
                    gt_np = sample_mask.cpu().numpy()
                    pq, _, _ = compute_pq(pred_np, gt_np, iou_threshold=0.5)
                    total_pq += pq
                
                num_samples += 1
    
    avg_instance_dice = total_instance_dice / max(num_cells, 1)
    avg_semantic_dice = total_semantic_dice / max(num_samples, 1)
    avg_pq = total_pq / max(num_samples, 1) if use_pq else 0.0
    
    # Return instance dice as primary metric
    return avg_instance_dice, avg_pq, avg_semantic_dice



def main():
    parser = argparse.ArgumentParser(description='CellSAM Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['output']['experiment_name']
    output_dir = Path(config['output']['checkpoint_dir']) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config to output
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create components
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    model, adapter = create_model(config, device)
    optimizer, scheduler = create_optimizer(model, config, adapter=adapter)
    
    criterion = CombinedLoss(
        pos_weight=config['loss']['pos_weight'],
        boundary_weight=config['loss']['boundary_weight'],
        aji_weight=config['loss'].get('aji_weight', 0.2),
        use_boundary=config['loss']['use_boundary'],
        use_aji=config['loss'].get('use_aji', True)
    )
    
    # Mixed precision scaler
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")
    
    # Training loop
    best_dice = 0
    best_pq = 0
    patience_counter = 0
    early_stop_patience = config['training'].get('early_stop_patience', 10)
    use_pq_early_stop = config['training'].get('use_pq_early_stop', False)
    
    if use_pq_early_stop:
        print("Using PQ (Panoptic Quality) for early stopping")
    else:
        print("Using Dice for early stopping")
    
    box_expand = config['loss'].get('box_expand', 0.1)
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, adapter=adapter, box_expand=box_expand)
        val_dice, val_pq, val_semantic_dice = validate(model, val_loader, criterion, device, adapter=adapter, use_pq=use_pq_early_stop, box_expand=box_expand)
        scheduler.step()
        
        # Select metric for early stopping
        if use_pq_early_stop:
            current_metric = val_pq
            best_metric = best_pq
            metric_name = "PQ"
        else:
            current_metric = val_dice
            best_metric = best_dice
            metric_name = "Dice"
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] "
              f"Train Loss: {train_loss:.4f}, Instance Dice: {val_dice:.4f}, "
              f"Semantic Dice: {val_semantic_dice:.4f}, PQ: {val_pq:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model (including adapter if present)
        if current_metric > best_metric:
            if use_pq_early_stop:
                best_pq = val_pq
            else:
                best_dice = val_dice
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'epoch': epoch + 1,
                'best_dice': val_dice,
                'best_pq': val_pq,
                'config': config,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  -> New best {metric_name}! Saved to {output_dir / 'best_model.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (patience={early_stop_patience})")
                break
        
        # Periodic checkpoints (including adapter)
        if (epoch + 1) % config['output']['save_every'] == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'epoch': epoch + 1,
                'val_dice': val_dice,
                'val_pq': val_pq,
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch+1}.pt")
    
    print(f"\nTraining complete! Best Val Dice: {best_dice:.4f}, Best Val PQ: {best_pq:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
