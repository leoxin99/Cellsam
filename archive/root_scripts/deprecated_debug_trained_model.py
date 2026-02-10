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
Compare base model vs trained checkpoint predictions
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

CHECKPOINT_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load one validation sample
dataset = AugmentedAllenDataset(
    data_dir="D:/AI/paper/CellSam/data/processed",
    is_training=False
)
sample = dataset[0]
print(f"\nSample: Image={sample['image'].shape}, Mask={sample['mask'].shape}")
print(f"Cell IDs: {sample['cell_ids']}")


def test_model(model, sample, model_name):
    """Test model prediction on a single cell."""
    model.eval()

    with torch.no_grad():
        img = sample['image'].unsqueeze(0).to(device)
        box = sample['boxes'][0:1].unsqueeze(0).to(device)
        cell_id = sample['cell_ids'][0].item()
        gt_mask = sample['mask'].to(device)

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
        pred_mask = F.interpolate(
            low_res_masks, size=(gt_mask.shape[0], gt_mask.shape[1]),
            mode='bilinear', align_corners=False
        ).squeeze()

        # Get GT for this cell
        gt_cell_mask = (gt_mask == cell_id).float()

        # Calculate Dice
        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
        intersection = (pred_binary * gt_cell_mask).sum()
        dice = (2 * intersection) / (pred_binary.sum() + gt_cell_mask.sum() + 1e-8)

        print(f"\n[{model_name}]")
        print(f"  pred_mask min/max: {pred_mask.min().item():.4f} / {pred_mask.max().item():.4f}")
        print(f"  After sigmoid: {torch.sigmoid(pred_mask).min().item():.4f} / {torch.sigmoid(pred_mask).max().item():.4f}")
        print(f"  pred_binary sum: {pred_binary.sum().item()}")
        print(f"  gt_cell_mask sum: {gt_cell_mask.sum().item()}")
        print(f"  Dice Score: {dice.item():.4f}")

        return dice.item()


# Test 1: Base model (pretrained)
print("\n" + "="*60)
print("TEST 1: Base pretrained model")
print("="*60)
base_model = get_model()
base_model = base_model.to(device)
dice_base = test_model(base_model, sample, "Base Model")

# Test 2: Load trained checkpoint
print("\n" + "="*60)
print("TEST 2: Trained checkpoint")
print("="*60)
trained_model = get_model()
trained_model = trained_model.to(device)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model_state_dict from checkpoint")
else:
    trained_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint directly as state_dict")

dice_trained = test_model(trained_model, sample, "Trained Model")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"  Base Model Dice:    {dice_base:.4f}")
print(f"  Trained Model Dice: {dice_trained:.4f}")
print(f"  Difference:         {dice_trained - dice_base:+.4f}")
