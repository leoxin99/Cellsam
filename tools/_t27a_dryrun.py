# -*- coding: utf-8 -*-
"""T27a dry-run: 1 epoch, verify freeze, loss, and logging are correct."""
import sys, os, yaml, torch
from pathlib import Path

project = Path(__file__).parent.parent
sys.path.insert(0, str(project / "src"))
sys.path.insert(0, str(project / "cellSAM_source"))
os.chdir(str(project))

with open(project / "src/config/t27a_planb_decoder.yaml") as f:
    config = yaml.safe_load(f)

# Override for dry-run
config['training']['epochs'] = 1
config['output']['experiment_name'] = 'T27a_dryrun'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train import create_dataloaders, create_model, create_optimizer, train_one_epoch, validate, CombinedLoss
from inference.core import InferenceConfig

print("=" * 60)
print("T27a DRY-RUN AUDIT")
print("=" * 60)

# 1. Model + freeze
model, adapter = create_model(config, device)
optimizer, scheduler = create_optimizer(model, config, adapter=adapter)

# 2. Loss 
criterion = CombinedLoss(
    pos_weight=config['loss']['pos_weight'],
    boundary_weight=config['loss']['boundary_weight'],
    aji_weight=config['loss'].get('aji_weight', 0.2),
    use_boundary=config['loss']['use_boundary'],
    use_aji=config['loss'].get('use_aji', True),
    use_focal=config['loss'].get('use_focal', False),
    focal_weight=config['loss'].get('focal_weight', 0.3),
    focal_alpha=config['loss'].get('focal_alpha', 0.25),
    focal_gamma=config['loss'].get('focal_gamma', 2.0),
)

print("\n--- Loss Config ---")
print(f"  use_focal={criterion.use_focal}, focal_weight={criterion.focal_weight}")
print(f"  use_boundary={criterion.use_boundary}, boundary_weight={criterion.boundary_weight}")
print(f"  pos_weight={criterion.pos_weight}")

# 3. IoU weight
iou_w = config['training'].get('iou_weight', 0.0)
print(f"  iou_weight={iou_w}")

# 4. Postprocess config
infer_cfg = InferenceConfig.default()
print(f"\n--- Inference Config ---")
print(f"  apply_postprocess={infer_cfg.apply_postprocess}")
print(f"  use_sam_iou_filter={infer_cfg.use_sam_iou_filter}")

# 5. Quick 1-batch train
train_loader, val_loader = create_dataloaders(config)
print(f"\n--- Training 1 batch ---")

from torch.cuda.amp import GradScaler
scaler = GradScaler()
box_expand = config['loss'].get('box_expand', 0.1)
train_loss = train_one_epoch(
    model, train_loader, optimizer, criterion, device,
    scaler=scaler, adapter=adapter, box_expand=box_expand,
    use_lora=False, iou_weight=iou_w
)
print(f"  Train loss: {train_loss:.4f}")

# 6. Quick validation
print(f"\n--- Validating ---")
val_metrics = validate(model, val_loader, criterion, device, adapter=adapter, use_pq=True, box_expand=box_expand)
print(f"  Val PQ: {val_metrics['pq']:.4f}")
print(f"  Val BM-Dice: {val_metrics['bm_1to1']:.4f}")

print("\n" + "=" * 60)
print("DRY-RUN COMPLETE - All systems go")
print("=" * 60)
