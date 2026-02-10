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
Analyze class imbalance in training data
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from augmented_dataset import AugmentedAllenDataset

# Load dataset
dataset = AugmentedAllenDataset(
    data_dir="D:/AI/paper/CellSam/data/processed",
    is_training=False
)

print("Analyzing class imbalance in training data...")
print("="*60)

total_pixels = 1024 * 1024
fg_ratios = []

for i in range(min(10, len(dataset))):
    sample = dataset[i]
    mask = sample['mask'].numpy()

    print(f"\nSample {i}: {sample['sample_id'][:30]}")

    for j, cell_id in enumerate(sample['cell_ids'][:5].numpy()):
        cell_mask = (mask == cell_id)
        fg_pixels = cell_mask.sum()
        fg_ratio = fg_pixels / total_pixels * 100
        fg_ratios.append(fg_ratio)
        print(f"  Cell {cell_id}: {fg_pixels:6d} pixels ({fg_ratio:.2f}% of image)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Mean foreground ratio: {np.mean(fg_ratios):.2f}%")
print(f"  Min foreground ratio:  {np.min(fg_ratios):.2f}%")
print(f"  Max foreground ratio:  {np.max(fg_ratios):.2f}%")
print(f"  Background ratio:      ~{100 - np.mean(fg_ratios):.2f}%")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print(f"  Class imbalance ratio: ~{(100 - np.mean(fg_ratios)) / np.mean(fg_ratios):.0f}:1 (bg:fg)")
print("  This severe imbalance causes BCE loss to push predictions toward 'all background'")
print("\nRECOMMENDED SOLUTIONS:")
print("  1. Use weighted BCE (pos_weight based on class ratio)")
print("  2. Compute loss only within bounding box region")
print("  3. Use Focal Loss to reduce easy negative impact")
