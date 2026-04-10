"""
Unified training script for CellSAM.
Supports config-driven training for base model and fine-tuning.

Usage:
    python src/train.py --config src/config/base.yaml
    python src/train.py --config src/config/boundary.yaml
"""
import argparse
import json
import random
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


def _resolve_box_artifact_settings(data_config: dict) -> dict:
    """Resolve generic box-artifact settings with T36 backward compatibility."""
    use_box_artifact = data_config.get('use_box_artifact', False)
    artifact_json = data_config.get('box_artifact_json')
    artifact_strategy = data_config.get('box_artifact_strategy', 'filter')
    noisy_box_ratio = float(data_config.get('noisy_box_ratio', 0.5))
    artifact_name = data_config.get('box_artifact_name', 'artifact')
    legacy_cf_mode = False

    # Backward compatibility: T36 CF-box configs
    if not use_box_artifact and data_config.get('use_cf_boxes', False):
        use_box_artifact = True
        artifact_json = data_config.get('cf_boxes_json')
        artifact_strategy = data_config.get('cf_box_strategy', 'filter')
        artifact_name = 'cf_boxes'
        legacy_cf_mode = True

    return {
        'enabled': use_box_artifact,
        'json_path': artifact_json,
        'strategy': artifact_strategy,
        'noisy_ratio': noisy_box_ratio,
        'artifact_name': artifact_name,
        'legacy_cf_mode': legacy_cf_mode,
    }


def _load_box_artifact_json(project_root: Path, artifact_json_path: str):
    """Load one detector-box artifact JSON using project-root relative paths."""
    artifact_json = Path(artifact_json_path)
    if not artifact_json.is_absolute():
        artifact_json = project_root / artifact_json

    with open(artifact_json, 'r', encoding='utf8') as f:
        artifact_data = json.load(f)

    return artifact_json, artifact_data


def _get_artifact_boxes(entry: dict):
    """Read detector boxes from either generic or legacy T36 artifact entries."""
    return entry.get('boxes', entry.get('cf_boxes', []))


def _extract_best_matched_noisy_boxes(entry: dict):
    """Keep at most one noisy box per GT cell, choosing the highest-IoU match."""
    noisy_boxes = _get_artifact_boxes(entry)
    matched_ids = entry.get('matched_gt_cell_ids', [])
    match_ious = entry.get('match_ious', [])

    best_by_gt = {}
    for idx, (box, gt_id) in enumerate(zip(noisy_boxes, matched_ids)):
        if gt_id == -1:
            continue
        iou = float(match_ious[idx]) if idx < len(match_ious) else 0.0
        record = best_by_gt.get(int(gt_id))
        if record is None or iou > record['iou']:
            best_by_gt[int(gt_id)] = {
                'box': [float(v) for v in box],
                'iou': iou,
            }
    return best_by_gt


def _build_training_boxes(
    gt_boxes: torch.Tensor,
    gt_cell_ids: torch.Tensor,
    artifact_entry: dict,
    strategy: str,
    noisy_ratio: float,
    legacy_cf_mode: bool = False,
):
    """Build per-sample training prompts from GT and/or frozen detector artifact."""
    gt_pairs = []
    for box, cell_id in zip(gt_boxes, gt_cell_ids):
        cid = int(cell_id.item()) if hasattr(cell_id, 'item') else int(cell_id)
        if cid <= 0:
            continue
        gt_pairs.append(([float(v) for v in box.tolist()], cid))

    if artifact_entry is None:
        if not gt_pairs:
            return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.long)
        boxes_tensor = torch.tensor([box for box, _ in gt_pairs], dtype=torch.float32)
        ids_tensor = torch.tensor([cid for _, cid in gt_pairs], dtype=torch.long)
        return boxes_tensor, ids_tensor

    noisy_by_gt = _extract_best_matched_noisy_boxes(artifact_entry)
    strategy = strategy.lower()

    if legacy_cf_mode and strategy == 'filter':
        noisy_boxes = _get_artifact_boxes(artifact_entry)
        matched_ids = artifact_entry.get('matched_gt_cell_ids', [])
        filtered_pairs = []
        for box, gt_id in zip(noisy_boxes, matched_ids):
            if gt_id == -1:
                continue
            filtered_pairs.append(([float(v) for v in box], int(gt_id)))
        if not filtered_pairs:
            return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.long)
        boxes_tensor = torch.tensor([box for box, _ in filtered_pairs], dtype=torch.float32)
        ids_tensor = torch.tensor([cid for _, cid in filtered_pairs], dtype=torch.long)
        return boxes_tensor, ids_tensor

    if strategy in {'filter', 'replace', 'noisy_only'}:
        if not noisy_by_gt:
            return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.long)
        selected_ids = sorted(noisy_by_gt.keys())
        boxes_tensor = torch.tensor([noisy_by_gt[cid]['box'] for cid in selected_ids], dtype=torch.float32)
        ids_tensor = torch.tensor(selected_ids, dtype=torch.long)
        return boxes_tensor, ids_tensor

    if strategy == 'mixed':
        if not gt_pairs:
            return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        gt_by_id = {cid: box for box, cid in gt_pairs}
        available_noisy_ids = [cid for cid in noisy_by_gt.keys() if cid in gt_by_id]
        random.shuffle(available_noisy_ids)

        target_count = len(gt_pairs)
        num_noisy = min(len(available_noisy_ids), int(round(target_count * noisy_ratio)))
        selected_noisy_ids = available_noisy_ids[:num_noisy]

        remaining_gt_ids = [cid for _, cid in gt_pairs if cid not in set(selected_noisy_ids)]
        random.shuffle(remaining_gt_ids)
        selected_gt_ids = remaining_gt_ids[: target_count - num_noisy]

        final_pairs = []
        for cid in selected_gt_ids:
            final_pairs.append((gt_by_id[cid], cid))
        for cid in selected_noisy_ids:
            final_pairs.append((noisy_by_gt[cid]['box'], cid))

        if not final_pairs:
            return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.long)

        random.shuffle(final_pairs)
        boxes_tensor = torch.tensor([box for box, _ in final_pairs], dtype=torch.float32)
        ids_tensor = torch.tensor([cid for _, cid in final_pairs], dtype=torch.long)
        return boxes_tensor, ids_tensor

    raise ValueError(f"Unknown box artifact strategy: {strategy}")


def create_dataloaders(config: dict):
    """Create train and validation dataloaders using fixed splits."""
    project_root = Path(__file__).resolve().parent.parent
    train_ids = load_split_ids("train", config['data']['splits_dir'])
    val_ids = load_split_ids("val", config['data']['splits_dir'])
    
    # Get flags from config
    use_bf_only = config['data'].get('use_bf_only', False)
    use_semantic_mapping = config['data'].get('use_semantic_mapping', False)
    use_2ch = config['data'].get('use_2ch', False)
    use_official_encoding = config['data'].get('use_official_encoding', False)
    official_r_channel = config['data'].get('official_r_channel', 'blank')
    
    train_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=True,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=train_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping,
        use_2ch=use_2ch,
        use_official_encoding=use_official_encoding,
        official_r_channel=official_r_channel
    )
    
    val_dataset = AugmentedAllenDataset(
        data_dir=config['data']['processed_data_dir'],
        target_size=tuple(config['data']['target_size']),
        is_training=False,
        max_boxes_per_image=config['data']['max_boxes_per_image'],
        sample_ids=val_ids,
        use_bf_only=use_bf_only,
        use_semantic_mapping=use_semantic_mapping,
        use_2ch=use_2ch,
        use_official_encoding=use_official_encoding,
        official_r_channel=official_r_channel
    )
    
    # Generic detector artifact override / mixing for noisy-box training
    box_artifact_cfg = _resolve_box_artifact_settings(config['data'])
    box_artifact_data = None
    if box_artifact_cfg['enabled']:
        artifact_json, box_artifact_data = _load_box_artifact_json(
            project_root, box_artifact_cfg['json_path']
        )
        print(f"[BOX ARTIFACT] Loading {box_artifact_cfg['artifact_name']} from: {artifact_json}")
        print(
            f"[BOX ARTIFACT] Strategy={box_artifact_cfg['strategy']}, "
            f"noisy_ratio={box_artifact_cfg['noisy_ratio']:.2f}"
        )
        print(f"[BOX ARTIFACT] Summary: {box_artifact_data.get('summary', {})}")
    
    def artifact_collate_fn(batch):
        """Collate that optionally mixes GT boxes with frozen detector boxes."""
        result = collate_fn(batch)
        if box_artifact_data is None:
            return result
        
        new_boxes = []
        new_cell_ids = []
        new_counts = []
        for i, sample in enumerate(batch):
            sample_id = str(sample.get('sample_id', ''))
            artifact_entry = box_artifact_data.get('images', {}).get(sample_id)
            sample_boxes, sample_ids = _build_training_boxes(
                gt_boxes=sample['boxes'],
                gt_cell_ids=sample['cell_ids'],
                artifact_entry=artifact_entry,
                strategy=box_artifact_cfg['strategy'],
                noisy_ratio=box_artifact_cfg['noisy_ratio'],
                legacy_cf_mode=box_artifact_cfg['legacy_cf_mode'],
            )
            new_boxes.append(sample_boxes)
            new_cell_ids.append(sample_ids)
            new_counts.append(sample_boxes.shape[0])
        
        result['boxes'] = new_boxes
        result['cell_ids'] = new_cell_ids
        result['box_counts'] = torch.tensor(new_counts, dtype=torch.long)
        return result
    
    train_collate = artifact_collate_fn if box_artifact_cfg['enabled'] else collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate
    )
    
    # Validation always uses GT boxes (fair comparison)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def _maybe_clip_to_box(pred_mask, target, box, box_expand, apply_box_clipping):
    """Optionally clip prediction/target to an expanded box region."""
    if not apply_box_clipping:
        return pred_mask, target

    x1, y1, x2, y2 = [int(b) for b in box.tolist()]
    h, w = pred_mask.shape
    bw, bh = x2 - x1, y2 - y1
    x1_clip = max(0, int(x1 - bw * box_expand))
    y1_clip = max(0, int(y1 - bh * box_expand))
    x2_clip = min(w, int(x2 + bw * box_expand))
    y2_clip = min(h, int(y2 + bh * box_expand))

    pred_clipped = torch.zeros_like(pred_mask)
    pred_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = pred_mask[y1_clip:y2_clip, x1_clip:x2_clip]

    target_clipped = torch.zeros_like(target)
    target_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = target[y1_clip:y2_clip, x1_clip:x2_clip]
    return pred_clipped, target_clipped


def create_model(config: dict, device):
    """Create or load CellSAM model and optional channel adapter.
    
    Initialization order (critical for LoRA):
        1. get_model() → fresh SAM
        2. freeze_encoder → all encoder params requires_grad=False
        3. apply_lora → creates NEW LoRA params with requires_grad=True
        4. load_state_dict → loads checkpoint (including LoRA keys if present)
    
    This order ensures:
        - P1-1: LoRA params survive freeze (created AFTER freeze)
        - M3: LoRA keys in checkpoint are loaded into existing LoRA layers
               (not silently dropped by strict=False)
    """
    model = get_model()
    
    # Plan B: Use model_cp (Stage 2 weights, PQ=0.434) directly.
    # No weight copy needed — we operate on model_cp throughout.
    # Ensure adv_mode=True so forward()/prep_2 use model_cp branch.
    model.adv_mode = True
    print("★ Plan B: Using model_cp directly (adv_mode=True, official pipeline)")
    
    use_lora = config['model'].get('use_lora', False)
    
    # Step 0: Freeze ALL non-target branches (model + cellfinder)
    # These branches are not used in Plan B training, but model.parameters()
    # would otherwise include them in the optimizer (~200M unwanted params)
    for param in model.model.parameters():
        param.requires_grad = False
    if hasattr(model, 'cellfinder') and model.cellfinder is not None:
        for param in model.cellfinder.parameters():
            param.requires_grad = False
    print("Froze model (Stage 1) + cellfinder branches")
    
    # Step 1: Freeze layers as specified (on model_cp — the active branch)
    if config['model']['freeze_encoder']:
        for param in model.model_cp.image_encoder.parameters():
            param.requires_grad = False
        print("Froze image encoder (model_cp)")
    
    if config['model']['freeze_decoder']:
        for param in model.model_cp.mask_decoder.parameters():
            param.requires_grad = False
        print("Froze mask decoder")
    
    # T32: Neck-only training (Stage2-like surrogate)
    train_neck_only = config['model'].get('train_neck_only', False)
    if train_neck_only:
        # Unfreeze ONLY the neck within the (already frozen) encoder
        for param in model.model_cp.image_encoder.neck.parameters():
            param.requires_grad = True
        neck_params = sum(p.numel() for p in model.model_cp.image_encoder.neck.parameters())
        print(f"[T32] Neck-only mode: unfroze {neck_params:,} neck parameters")
        
        # If freeze_decoder is also set, decoder stays frozen
        # If NOT set, decoder is trainable alongside neck
        if config['model']['freeze_decoder']:
            print("[T32] Decoder FROZEN — pure neck-only baseline")
        else:
            print("[T32] Decoder TRAINABLE — neck + decoder mode")
    
    # Plan B: Always freeze prompt encoder (512 params, pure positional encoding)
    # Literature consensus: FSAM, ProMISe, Sam2Rad, MedSAM all freeze PE
    for param in model.model_cp.prompt_encoder.parameters():
        param.requires_grad = False
    print("Froze prompt encoder (model_cp, 512 params)")
    
    # Trainable parameter audit (always print for verification)
    trainable_params = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    total_trainable = sum(trainable_params.values())
    total_all = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Trainable parameters: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.2f}%)")
    if train_neck_only:
        print("[T32 AUDIT] Expected: only neck params trainable (+ decoder if unfrozen)")
        for name, count in sorted(trainable_params.items()):
            print(f"  {name}: {count:,}")
    print(f"{'='*60}\n")
    
    # Step 2: Apply LoRA BEFORE loading checkpoint (M3 fix)
    # This creates LoRA layers so checkpoint's LoRA keys have matching targets
    if use_lora:
        from lora import apply_lora_to_encoder
        lora_rank = config['model'].get('lora_rank', 4)
        apply_lora_to_encoder(model.model_cp.image_encoder, rank=lora_rank)
    
    # Step 3: Load checkpoint AFTER LoRA is applied
    if config['model']['checkpoint']:
        checkpoint_path = config['model']['checkpoint']
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle both dict format and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
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
    
    # Diagnostic: per-branch trainable param breakdown (A1 review fix)
    branches = {
        'model': model.model,
        'model_cp.image_encoder': model.model_cp.image_encoder,
        'model_cp.prompt_encoder': model.model_cp.prompt_encoder,
        'model_cp.mask_decoder': model.model_cp.mask_decoder,
    }
    if hasattr(model, 'cellfinder') and model.cellfinder is not None:
        branches['cellfinder'] = model.cellfinder
    for name, module in branches.items():
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in module.parameters())
        status = '[TRAIN]' if n_train > 0 else '[FROZEN]'
        print(f"  {status} {name}: {n_train:,} / {n_total:,} trainable")
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


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, adapter=None, box_expand=0.1, use_lora=False, iou_weight=0.0, train_neck_only=False, apply_box_clipping=True):
    """Train one epoch with optional mixed precision (AMP) and instance-level training.
    
    Key improvements:
    - box_expand: Constrain pred/target to expanded box region
    - Uses cell_id from batch to get specific cell mask (not entire semantic mask)
    - use_lora: When True, encoder forward runs WITH gradients for LoRA (P0-1 fix)
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
        
        # Plan B: Use official preprocessing pipeline (prep_2 + forward)
        # P0-1 fix: LoRA requires gradient flow through encoder
        # T32 fix: Neck-only also requires gradient flow (neck is inside encoder)
        from official_preprocess import official_preprocess_only, official_preprocess_and_encode
        if use_lora or train_neck_only:
            # LoRA/Neck-only: encoder forward WITH gradients
            # (LoRA params or neck params need autograd through the encoder graph)
            if use_amp:
                with autocast():
                    img_preprocessed = official_preprocess_only(model, images)
                    image_embedding = model.model_cp.image_encoder(img_preprocessed)
            else:
                img_preprocessed = official_preprocess_only(model, images)
                image_embedding = model.model_cp.image_encoder(img_preprocessed)
        else:
            # Standard: frozen encoder, no gradients needed
            with torch.no_grad():
                if use_amp:
                    with autocast():
                        image_embedding = official_preprocess_and_encode(model, images)
                else:
                    image_embedding = official_preprocess_and_encode(model, images)
        
        batch_loss = 0
        num_cells = 0
        
        # Fix-3: For LoRA, count expected cells upfront for loss scaling
        # (per-box backward needs the total count for correct averaging)
        if use_lora:
            num_cells_expected = 0
            for i in range(images.shape[0]):
                for j in range(len(boxes[i])):
                    if boxes[i][j].sum() != 0:
                        num_cells_expected += 1
            num_cells_expected = max(num_cells_expected, 1)
        
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
                            sparse_emb, dense_emb = model.model_cp.prompt_encoder(
                                points=None, boxes=box_tensor, masks=None
                            )
                            low_res_masks, iou_pred = model.model_cp.mask_decoder(
                                image_embeddings=image_embedding[i:i+1],
                                image_pe=model.model_cp.prompt_encoder.get_dense_pe(),
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
                            
                            pred_clipped, target_clipped = _maybe_clip_to_box(
                                pred_mask=pred_mask,
                                target=target,
                                box=box,
                                box_expand=box_expand,
                                apply_box_clipping=apply_box_clipping,
                            )
                            
                            loss = criterion(pred_clipped, target_clipped, box=box.tolist(),
                                             instance_mask=sample_mask.float(),
                                             confidence_map=confidence_map.detach())
                            
                            # IoU Head Loss: align quality prediction with actual IoU
                            if iou_weight > 0 and iou_pred is not None:
                                with torch.no_grad():
                                    pred_binary = (torch.sigmoid(pred_clipped) > 0.5).float()
                                    intersection = (pred_binary * target_clipped).sum()
                                    union = pred_binary.sum() + target_clipped.sum() - intersection
                                    actual_iou = intersection / (union + 1e-6)
                                iou_loss = F.mse_loss(iou_pred.squeeze(), actual_iou)
                                loss = loss + iou_weight * iou_loss
                    else:
                        sparse_emb, dense_emb = model.model_cp.prompt_encoder(
                            points=None, boxes=box_tensor, masks=None
                        )
                        low_res_masks, iou_pred = model.model_cp.mask_decoder(
                            image_embeddings=image_embedding[i:i+1],
                            image_pe=model.model_cp.prompt_encoder.get_dense_pe(),
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
                        
                        pred_clipped, target_clipped = _maybe_clip_to_box(
                            pred_mask=pred_mask,
                            target=target,
                            box=box,
                            box_expand=box_expand,
                            apply_box_clipping=apply_box_clipping,
                        )
                        
                        loss = criterion(pred_clipped, target_clipped, box=box.tolist(),
                                         instance_mask=sample_mask.float(),
                                         confidence_map=confidence_map.detach())
                        
                        # IoU Head Loss: align quality prediction with actual IoU
                        if iou_weight > 0 and iou_pred is not None:
                            with torch.no_grad():
                                pred_binary = (torch.sigmoid(pred_clipped) > 0.5).float()
                                intersection = (pred_binary * target_clipped).sum()
                                union = pred_binary.sum() + target_clipped.sum() - intersection
                                actual_iou = intersection / (union + 1e-6)
                            iou_loss = F.mse_loss(iou_pred.squeeze(), actual_iou)
                            loss = loss + iou_weight * iou_loss
                    
                    # Fix-3: Per-box backward for LoRA mode
                    # Each box's loss backward immediately → releases decoder computation graph
                    # Gradients accumulate in .grad (math-equivalent to batch backward)
                    if use_lora:
                        scaled_loss = loss / num_cells_expected
                        if use_amp:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        batch_loss += loss.item()  # Track scalar only
                    else:
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
            # Fix-3: LoRA already did per-box backward, just step optimizer
            if use_lora:
                if use_amp:
                    scaler.unscale_(optimizer)
                    all_params = list(model.parameters())
                    if adapter is not None:
                        all_params.extend(list(adapter.parameters()))
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    all_params = list(model.parameters())
                    if adapter is not None:
                        all_params.extend(list(adapter.parameters()))
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                
                total_loss += batch_loss / num_cells  # Already scalar
            else:
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


def validate(model, dataloader, criterion, device, adapter=None, use_pq=False, box_expand=0.1, apply_box_clipping=True):
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
    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_samples = 0
    
    # 统一推理配置 (单一来源, 仅 override box_expand from training config)
    # A1 review fix: use InferenceConfig.default() which has apply_postprocess=True
    # This ensures early-stop metric matches external evaluation metric
    infer_cfg = InferenceConfig.default()
    infer_cfg.box_expand = box_expand
    infer_cfg.validate_size = False
    infer_cfg.apply_box_clipping = apply_box_clipping
    
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
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']
                num_samples += 1
    
    n = max(num_samples, 1)
    avg_bm_1to1 = total_bm_1to1 / n
    avg_bm_coverage = total_bm_coverage / n
    avg_pq = total_pq / n if use_pq else 0.0
    avg_semantic_dice = total_semantic_dice / n
    avg_conflict = total_conflict_pixels / n
    avg_gap = avg_bm_coverage - avg_bm_1to1
    
    # Detection metrics: F1 (=RQ), Precision, Recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 返回完整诊断信息
    return {
        'bm_1to1': avg_bm_1to1,
        'bm_coverage': avg_bm_coverage,
        'gap': avg_gap,
        'pq': avg_pq,
        'semantic_dice': avg_semantic_dice,
        'conflict': avg_conflict,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
    }
def validate_with_box_artifact(model, dataloader, artifact_data, device, adapter=None,
                               box_expand=0.1, apply_box_clipping=True):
    """Validation on frozen detector boxes (E2E-style, detector fixed via artifact)."""
    model.eval()
    if adapter is not None:
        adapter.eval()

    total_bm_1to1 = 0.0
    total_bm_coverage = 0.0
    total_pq = 0.0
    total_sq = 0.0
    total_rq = 0.0
    total_aji = 0.0
    total_semantic_dice = 0.0
    total_conflict_pixels = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_samples = 0

    infer_cfg = InferenceConfig.default()
    infer_cfg.box_expand = box_expand
    infer_cfg.validate_size = False
    infer_cfg.apply_box_clipping = apply_box_clipping

    images_lookup = artifact_data.get('images', {})

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            sample_ids = batch['sample_ids']

            if adapter is not None:
                images = adapter(images)

            for i in range(images.shape[0]):
                sample_id = str(sample_ids[i])
                entry = images_lookup.get(sample_id, {})
                sample_boxes = _get_artifact_boxes(entry)
                sample_mask = masks[i]

                if not sample_boxes:
                    h, w = sample_mask.shape[-2:]
                    pred_np = np.zeros((h, w), dtype=np.int32)
                    gt_np = sample_mask.cpu().numpy()
                    metrics = compute_all_metrics(pred_np, gt_np, iou_threshold=0.5)
                    total_bm_1to1 += metrics['bm_1to1_dice']
                    total_bm_coverage += metrics['bm_coverage_dice']
                    total_pq += metrics['pq']
                    total_sq += metrics['sq']
                    total_rq += metrics['rq']
                    total_aji += metrics['aji']
                    total_semantic_dice += metrics['semantic_dice']
                    total_tp += metrics['tp']
                    total_fp += metrics['fp']
                    total_fn += metrics['fn']
                    num_samples += 1
                    continue

                boxes_tensor = torch.tensor(sample_boxes, dtype=torch.float32)
                result = segment_with_boxes(
                    model=model,
                    image=images[i],
                    boxes=boxes_tensor,
                    config=infer_cfg,
                    device=str(device),
                )

                pred_np = result.instance_mask
                gt_np = sample_mask.cpu().numpy()
                metrics = compute_all_metrics(pred_np, gt_np, iou_threshold=0.5)

                total_bm_1to1 += metrics['bm_1to1_dice']
                total_bm_coverage += metrics['bm_coverage_dice']
                total_pq += metrics['pq']
                total_sq += metrics['sq']
                total_rq += metrics['rq']
                total_aji += metrics['aji']
                total_semantic_dice += metrics['semantic_dice']
                total_conflict_pixels += result.conflict_pixels
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_fn += metrics['fn']
                num_samples += 1

    n = max(num_samples, 1)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'pq': total_pq / n,
        'sq': total_sq / n,
        'rq': total_rq / n,
        'bm_1to1': total_bm_1to1 / n,
        'bm_coverage': total_bm_coverage / n,
        'aji': total_aji / n,
        'semantic_dice': total_semantic_dice / n,
        'conflict': total_conflict_pixels / n,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'n_samples': num_samples,
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
    
    # Create output directory (include seed to prevent collisions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['output']['experiment_name']
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    output_dir = Path(config['output']['checkpoint_dir']) / f"{exp_name}{seed_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config to output
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    project_root = Path(__file__).resolve().parent.parent

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
        # Plan B: Focal Loss + IoU Head (A1 review: must be explicit in YAML)
        use_focal=config['loss'].get('use_focal', False),
        focal_weight=config['loss'].get('focal_weight', 0.3),
        focal_alpha=config['loss'].get('focal_alpha', 0.25),
        focal_gamma=config['loss'].get('focal_gamma', 2.0),
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
    if config['loss'].get('use_focal', False):
        enabled_losses.append(f"Focal(a={config['loss'].get('focal_alpha', 0.25)},g={config['loss'].get('focal_gamma', 2.0)})")
    iou_w = config['training'].get('iou_weight', 0.0)
    if iou_w > 0:
        enabled_losses.append(f"IoU_Head(w={iou_w})")
    print(f"Enabled losses: {', '.join(enabled_losses)}")
    
    # Mixed precision scaler
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")
    
    # Training loop
    best_dice = 0
    best_pq = 0
    best_e2e_f1 = 0
    patience_counter = 0
    early_stop_patience = config['training'].get('early_stop_patience', 10)
    early_stop_metric = config['training'].get('early_stop_metric')
    if early_stop_metric is None:
        use_pq_early_stop = config['training'].get('use_pq_early_stop', False)
        early_stop_metric = 'oracle_pq' if use_pq_early_stop else 'oracle_bm_1to1'
    else:
        use_pq_early_stop = early_stop_metric == 'oracle_pq'

    e2e_val_artifact_data = None
    e2e_val_artifact_name = config['training'].get('e2e_val_box_artifact_name', 'e2e_val_artifact')
    e2e_val_artifact_json = config['training'].get('e2e_val_box_artifact_json')
    if early_stop_metric == 'e2e_f1':
        if not e2e_val_artifact_json:
            raise ValueError("training.e2e_val_box_artifact_json is required when early_stop_metric=e2e_f1")
        e2e_val_artifact_path, e2e_val_artifact_data = _load_box_artifact_json(project_root, e2e_val_artifact_json)
        print(f"[EARLY STOP] Using E2E F1 with frozen val artifact: {e2e_val_artifact_name}")
        print(f"[EARLY STOP] Artifact path: {e2e_val_artifact_path}")
        print(f"[EARLY STOP] Artifact summary: {e2e_val_artifact_data.get('summary', {})}")
    elif early_stop_metric == 'oracle_pq':
        print("Using Oracle PQ for early stopping")
    elif early_stop_metric == 'oracle_bm_1to1':
        print("Using Oracle BM-1to1 Dice for early stopping")
    else:
        raise ValueError(f"Unknown training.early_stop_metric: {early_stop_metric}")
    
    box_expand = config['loss'].get('box_expand', 0.1)
    train_apply_box_clipping = config['training'].get('apply_train_box_clipping', True)
    val_apply_box_clipping = config['training'].get('apply_val_box_clipping', True)
    
    for epoch in range(config['training']['epochs']):
        # Fix3: Update N/O loss weights based on delay schedule
        criterion.set_epoch(epoch)
        iou_weight = config['training'].get('iou_weight', 0.0)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            adapter=adapter,
            box_expand=box_expand,
            use_lora=config['model'].get('use_lora', False),
            iou_weight=iou_weight,
            train_neck_only=config['model'].get('train_neck_only', False),
            apply_box_clipping=train_apply_box_clipping,
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            adapter=adapter,
            use_pq=True,
            box_expand=box_expand,
            apply_box_clipping=val_apply_box_clipping,
        )
        e2e_val_metrics = None
        if e2e_val_artifact_data is not None:
            e2e_val_metrics = validate_with_box_artifact(
                model,
                val_loader,
                artifact_data=e2e_val_artifact_data,
                device=device,
                adapter=adapter,
                box_expand=box_expand,
                apply_box_clipping=val_apply_box_clipping,
            )
        val_dice = val_metrics['bm_1to1']
        val_pq = val_metrics['pq']
        val_semantic_dice = val_metrics['semantic_dice']
        val_coverage = val_metrics['bm_coverage']
        val_gap = val_metrics['gap']
        val_conflict = val_metrics['conflict']
        val_f1 = val_metrics['f1']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        scheduler.step()
        
        # Select metric for early stopping
        if early_stop_metric == 'oracle_pq':
            current_metric = val_pq
            best_metric = best_pq
            metric_name = "Oracle_PQ"
        elif early_stop_metric == 'oracle_bm_1to1':
            current_metric = val_dice
            best_metric = best_dice
            metric_name = "Oracle_BM1to1"
        else:
            current_metric = e2e_val_metrics['f1']
            best_metric = best_e2e_f1
            metric_name = f"E2E_F1[{e2e_val_artifact_name}]"
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] "
              f"Train Loss: {train_loss:.4f}, "
              f"BM-1to1: {val_dice:.4f}, BM-Cov: {val_coverage:.4f}, Gap: {val_gap:.4f}, "
              f"PQ: {val_pq:.4f}, Sem: {val_semantic_dice:.4f}, "
              f"F1: {val_f1:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, "
              f"Conflict: {val_conflict:.0f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        if e2e_val_metrics is not None:
            print(
                f"  [E2E {e2e_val_artifact_name}] "
                f"PQ: {e2e_val_metrics['pq']:.4f}, SQ: {e2e_val_metrics['sq']:.4f}, "
                f"RQ: {e2e_val_metrics['rq']:.4f}, F1: {e2e_val_metrics['f1']:.4f}, "
                f"P: {e2e_val_metrics['precision']:.4f}, R: {e2e_val_metrics['recall']:.4f}"
            )
        
        # Save best model (including adapter if present)
        if current_metric > best_metric:
            # Always track both metrics for accurate final reporting.
            # Note: under PQ early stop, best_dice is the dice at the best-PQ epoch,
            # not necessarily the global maximum dice across all epochs.
            best_dice = val_dice
            best_pq = val_pq
            if e2e_val_metrics is not None:
                best_e2e_f1 = e2e_val_metrics['f1']
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'epoch': epoch + 1,
                'best_dice': val_dice,
                'best_pq': val_pq,
                'best_e2e_f1': e2e_val_metrics['f1'] if e2e_val_metrics is not None else None,
                'early_stop_metric': early_stop_metric,
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
    
    summary = f"\nTraining complete! Best Val Dice: {best_dice:.4f}, Best Val PQ: {best_pq:.4f}"
    if early_stop_metric == 'e2e_f1':
        summary += f", Best Val E2E F1: {best_e2e_f1:.4f}"
    print(summary)
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
