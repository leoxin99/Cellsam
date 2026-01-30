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

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids
from losses.combined import CombinedLoss
from adapters import IndependentChannelAdapter, LightweightChannelAdapter


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


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, adapter=None):
    """Train one epoch with optional mixed precision (AMP)."""
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
            
            for box in sample_boxes:
                if box.sum() == 0:
                    continue
                
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
                            
                            target = (sample_mask > 0).float()
                            loss = criterion(pred_mask, target, box=box.tolist())
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
                        
                        target = (sample_mask > 0).float()
                        loss = criterion(pred_mask, target, box=box.tolist())
                    
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


def validate(model, dataloader, criterion, device, adapter=None):
    """Validation with Dice score computation."""
    model.eval()
    if adapter is not None:
        adapter.eval()
    
    total_dice = 0
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes']
            
            # Apply channel adapter if present
            if adapter is not None:
                images = adapter(images)
            
            img_preprocessed = model.sam_preprocess(images)
            image_embedding = model.model.image_encoder(img_preprocessed)
            
            for i in range(images.shape[0]):
                sample_boxes = boxes[i]
                sample_mask = masks[i]
                combined_pred = torch.zeros_like(sample_mask)
                
                for box in sample_boxes:
                    if box.sum() == 0:
                        continue
                    
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
                        
                        combined_pred = torch.maximum(combined_pred, torch.sigmoid(pred_mask))
                        
                    except Exception:
                        continue
                
                # Compute Dice
                pred_binary = (combined_pred > 0.5).float()
                target_binary = (sample_mask > 0).float()
                
                intersection = (pred_binary * target_binary).sum()
                dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
                
                total_dice += dice.item()
                num_samples += 1
    
    return total_dice / max(num_samples, 1)


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
    patience_counter = 0
    early_stop_patience = config['training'].get('early_stop_patience', 10)
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, adapter=adapter)
        val_dice = validate(model, val_loader, criterion, device, adapter=adapter)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] "
              f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model (including adapter if present)
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'epoch': epoch + 1,
                'best_dice': best_dice,
                'config': config,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
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
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch+1}.pt")
    
    print(f"\nTraining complete! Best Val Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
