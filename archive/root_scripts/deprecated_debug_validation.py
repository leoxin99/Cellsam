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
Debug validation Dice calculation
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model = model.to(device)
model.eval()

# Load one validation sample
dataset = AugmentedAllenDataset(
    data_dir="D:/AI/paper/CellSam/data/processed",
    is_training=False
)

print(f"Total samples: {len(dataset)}")

# Get first sample
sample = dataset[0]
print(f"\nSample 0:")
print(f"  Image shape: {sample['image'].shape}")
print(f"  Mask shape: {sample['mask'].shape}")
print(f"  Boxes shape: {sample['boxes'].shape}")
print(f"  Cell IDs: {sample['cell_ids']}")
print(f"  Num boxes: {sample['num_boxes']}")

# Check mask values
mask = sample['mask'].numpy()
unique_labels = np.unique(mask)
print(f"\n  Mask unique labels: {unique_labels}")
print(f"  Number of cells: {len(unique_labels[unique_labels > 0])}")

# Run inference on first cell
if sample['num_boxes'] > 0:
    with torch.no_grad():
        img = sample['image'].unsqueeze(0).to(device)
        box = sample['boxes'][0:1].unsqueeze(0).to(device)
        cell_id = sample['cell_ids'][0].item()

        print(f"\n  Testing cell ID: {cell_id}")

        # Forward pass
        img_preprocessed = model.sam_preprocess(img)
        embedding = model.model.image_encoder(img_preprocessed)

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

        # Upscale prediction
        gt_mask = sample['mask'].to(device)
        pred_mask = F.interpolate(
            low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
            mode='bilinear', align_corners=False
        ).squeeze()

        # Get GT for this cell
        gt_cell_mask = (gt_mask == cell_id).float()

        # Calculate Dice
        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()

        print(f"\n  Prediction stats:")
        print(f"    pred_mask min/max: {pred_mask.min().item():.4f} / {pred_mask.max().item():.4f}")
        print(f"    After sigmoid: {torch.sigmoid(pred_mask).min().item():.4f} / {torch.sigmoid(pred_mask).max().item():.4f}")
        print(f"    pred_binary sum: {pred_binary.sum().item()}")
        print(f"    gt_cell_mask sum: {gt_cell_mask.sum().item()}")

        intersection = (pred_binary * gt_cell_mask).sum()
        dice = (2 * intersection) / (pred_binary.sum() + gt_cell_mask.sum() + 1e-8)

        print(f"\n  Dice Score: {dice.item():.4f}")
        print(f"    Intersection: {intersection.item()}")
        print(f"    Pred pixels: {pred_binary.sum().item()}")
        print(f"    GT pixels: {gt_cell_mask.sum().item()}")
