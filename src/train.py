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
from inference.core import segment_with_boxes, InferenceConfig
from metrics.instance_metrics import (
    compute_bm_1to1_dice,
    compute_bm_coverage_dice,
    compute_pq as compute_pq_unified,
    compute_semantic_dice as compute_semantic_dice_unified,
    compute_all_metrics,
)


# NOTE: Local compute_pq / compute_best_match_dice 已移除
# 统一使用 metrics.instance_metrics 中的实现
# - compute_bm_1to1_dice   (Hungarian 1对1，主指标)
# - compute_bm_coverage_dice (每GT取最大，辅助诊断)
# - compute_pq_unified      (PQ with Hungarian matching)


def set_seed(seed: int):
    """Set random seed for reproducibility across all randomness sources.
    
    Covers:
      1. torch.manual_seed — PyTorch CPU ops
      2. torch.cuda.manual_seed_all — PyTorch GPU ops
      3. numpy.random.seed — NumPy (used by Albumentations)
      4. random.seed — Python stdlib (box shuffle, box augmentation)
      5. cuDNN deterministic — disable nondeterministic algorithms
      6. DataLoader reproducibility via worker_init_fn (handled separately)
    """
    import random
    import numpy as np
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} (deterministic mode)")


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
    use_2ch = config['data'].get('use_2ch', False)
    
    train_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=True,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=train_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping,
        use_2ch=use_2ch
    )
    
    val_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=False,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=val_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping,
        use_2ch=use_2ch
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
        # Support both nested (model.adapter.type) and flat (model.adapter_type) config keys
        adapter_type = adapter_config.get('type', config['model'].get('adapter_type', 'independent'))
        
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
            
            # Phase 2: Shuffle boxes per image to reduce L_overlap approximation bias
            box_indices = list(range(len(sample_boxes)))
            import random
            random.shuffle(box_indices)
            
            # Phase 2: Initialize confidence map for L_overlap (accumulates per-cell predictions)
            # Shape from sample_mask to handle non-1024 inputs
            conf_h, conf_w = sample_mask.shape[-2:]
            confidence_map = torch.zeros(conf_h, conf_w, device=device)
            
            for j in box_indices:
                box = sample_boxes[j]
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
                            
                            loss = criterion(pred_clipped, target_clipped, box=box.tolist(),
                                             instance_mask=sample_mask.float(),
                                             confidence_map=confidence_map.detach())
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
                        
                        loss = criterion(pred_clipped, target_clipped, box=box.tolist(),
                                         instance_mask=sample_mask.float(),
                                         confidence_map=confidence_map.detach())
                    
                    batch_loss += loss
                    num_cells += 1
                    
                    # Phase 2: Accumulate confidence map for L_overlap
                    with torch.no_grad():
                        pred_sigmoid_full = torch.sigmoid(pred_clipped)
                        confidence_map = confidence_map + pred_sigmoid_full
                    
                except Exception as e:
                    import warnings
                    warnings.warn(f"Box {j} in sample {i} failed: {e}")
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
    """Validation using unified inference core and metrics.
    
    Uses segment_with_boxes for inference and compute_all_metrics for evaluation.
    Reports:
      - BM-1to1 Dice  (主指标, Hungarian 一对一)
      - BM-Coverage Dice (辅助, 每GT取最大)
      - Gap Dice (Coverage - 1to1, 诊断粘连)
      - PQ@0.5
      - Semantic Dice
    """
    model.eval()
    if adapter is not None:
        adapter.eval()
    
    # 累加器
    total_bm_1to1 = 0.0
    total_bm_coverage = 0.0
    total_pq = 0.0
    total_semantic_dice = 0.0
    total_conflict_pixels = 0
    num_samples = 0
    
    # 统一推理配置 (单一来源, 仅 override box_expand from training config)
    infer_cfg = InferenceConfig.default()
    infer_cfg.box_expand = box_expand
    infer_cfg.apply_postprocess = False
    infer_cfg.validate_size = False
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes']
            
            # Apply channel adapter if present
            if adapter is not None:
                images = adapter(images)
            
            for i in range(images.shape[0]):
                sample_boxes = boxes[i]
                sample_mask = masks[i]
                
                # 跳过空 box
                if sample_boxes.shape[0] == 0 or sample_boxes.sum() == 0:
                    continue
                
                # === 统一推理 ===
                result = segment_with_boxes(
                    model=model,
                    image=images[i],
                    boxes=sample_boxes,
                    config=infer_cfg,
                    device=str(device),
                )
                
                pred_np = result.instance_mask
                gt_np = sample_mask.cpu().numpy()
                
                # === 统一指标 ===
                metrics = compute_all_metrics(pred_np, gt_np, iou_threshold=0.5)
                
                total_bm_1to1 += metrics['bm_1to1_dice']
                total_bm_coverage += metrics['bm_coverage_dice']
                total_pq += metrics['pq']
                total_semantic_dice += metrics['semantic_dice']
                total_conflict_pixels += result.conflict_pixels
                num_samples += 1
    
    n = max(num_samples, 1)
    avg_bm_1to1 = total_bm_1to1 / n
    avg_bm_coverage = total_bm_coverage / n
    avg_pq = total_pq / n if use_pq else 0.0
    avg_semantic_dice = total_semantic_dice / n
    avg_conflict = total_conflict_pixels / n
    avg_gap = avg_bm_coverage - avg_bm_1to1
    
    # 返回完整诊断信息
    return {
        'bm_1to1': avg_bm_1to1,
        'bm_coverage': avg_bm_coverage,
        'gap': avg_gap,
        'pq': avg_pq,
        'semantic_dice': avg_semantic_dice,
        'conflict': avg_conflict,
    }



def main():
    parser = argparse.ArgumentParser(description='CellSAM Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=None, help='Override seed in config (for multi-seed ablation)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # CLI seed override (for running same config with different seeds)
    if args.seed is not None:
        config['training']['seed'] = args.seed
        print(f"Seed overridden by CLI: {args.seed}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed if specified (for reproducibility in ablation experiments)
    seed = config['training'].get('seed', None)
    if seed is not None:
        set_seed(seed)
    
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
        use_aji=config['loss'].get('use_aji', True),
        # Phase 2 losses (configurable)
        use_topology=config['loss'].get('use_topology', False),
        topology_weight=config['loss'].get('topology_weight', 0.1),
        use_size=config['loss'].get('use_size', False),
        size_weight=config['loss'].get('size_weight', 0.1),
        use_contour=config['loss'].get('use_contour', False),
        contour_weight=config['loss'].get('contour_weight', 0.1),
        # Phase 2 Step 3: L_neighbor and L_overlap
        use_neighbor=config['loss'].get('use_neighbor', False),
        neighbor_weight=config['loss'].get('neighbor_weight', 0.3),
        use_overlap=config['loss'].get('use_overlap', False),
        overlap_weight=config['loss'].get('overlap_weight', 0.1),
        neighbor_gamma=config['loss'].get('neighbor_gamma', 1.5),
        overlap_margin=config['loss'].get('overlap_margin', 0.05),
        # Fix3: Delayed loss enable (phase2_design.md §8.4)
        delay_epochs=config['loss'].get('delay_epochs', 0),
        ramp_epochs=config['loss'].get('ramp_epochs', 10),
    )
    
    # Log enabled losses
    enabled_losses = ["Dice", "BCE"]
    if config['loss']['use_boundary']:
        enabled_losses.append("Boundary")
    if config['loss'].get('use_aji', True):
        enabled_losses.append("AJI")
    if config['loss'].get('use_topology', False):
        enabled_losses.append("Topology")
    if config['loss'].get('use_size', False):
        enabled_losses.append("Size")
    if config['loss'].get('use_contour', False):
        enabled_losses.append("Contour")
    if config['loss'].get('use_neighbor', False):
        enabled_losses.append("Neighbor")
    if config['loss'].get('use_overlap', False):
        enabled_losses.append("Overlap")
    print(f"Enabled losses: {', '.join(enabled_losses)}")
    
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
        # Fix3: Update N/O loss weights based on delay schedule
        criterion.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, adapter=adapter, box_expand=box_expand)
        val_metrics = validate(model, val_loader, criterion, device, adapter=adapter, use_pq=use_pq_early_stop, box_expand=box_expand)
        val_dice = val_metrics['bm_1to1']
        val_pq = val_metrics['pq']
        val_semantic_dice = val_metrics['semantic_dice']
        val_coverage = val_metrics['bm_coverage']
        val_gap = val_metrics['gap']
        val_conflict = val_metrics['conflict']
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
              f"Train Loss: {train_loss:.4f}, "
              f"BM-1to1: {val_dice:.4f}, BM-Cov: {val_coverage:.4f}, Gap: {val_gap:.4f}, "
              f"PQ: {val_pq:.4f}, Sem: {val_semantic_dice:.4f}, "
              f"Conflict: {val_conflict:.0f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model (including adapter if present)
        if current_metric > best_metric:
            # Always track both metrics for accurate final reporting.
            # Note: under PQ early stop, best_dice is the dice at the best-PQ epoch,
            # not necessarily the global maximum dice across all epochs.
            best_dice = val_dice
            best_pq = val_pq
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
            
            # Checkpoint retention: keep only last N periodic saves to avoid disk full
            keep_last = config['output'].get('keep_last_checkpoints', 3)
            if keep_last > 0:
                import glob
                periodic_ckpts = sorted(glob.glob(str(output_dir / "epoch_*.pt")))
                if len(periodic_ckpts) > keep_last:
                    for old_ckpt in periodic_ckpts[:-keep_last]:
                        Path(old_ckpt).unlink()
                        print(f"  [Retention] Removed old checkpoint: {Path(old_ckpt).name}")
    
    print(f"\nTraining complete! Best Val Dice: {best_dice:.4f}, Best Val PQ: {best_pq:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
