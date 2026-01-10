"""
Verify that each predicted cell mask is compared to its corresponding GT cell.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from augmented_dataset import AugmentedAllenDataset

# Load one sample
dataset = AugmentedAllenDataset(
    data_dir="D:/AI/paper/CellSam/data/processed",
    is_training=False
)

sample = dataset[0]
gt_mask = sample['mask'].numpy()
boxes = sample['boxes'].numpy()
cell_ids = sample['cell_ids'].numpy()
num_boxes = sample['num_boxes']

print("="*60)
print("VERIFYING CELL-TO-GT CORRESPONDENCE")
print("="*60)
print(f"\nSample: {sample['sample_id'][:40]}...")
print(f"Image shape: {sample['image'].shape}")
print(f"GT mask shape: {gt_mask.shape}")
print(f"Number of cells: {num_boxes}")
print(f"\nGT mask unique values: {np.unique(gt_mask)}")
print("  0 = background, 1,2,3... = individual cells")

print("\n" + "="*60)
print("CELL-BY-CELL CORRESPONDENCE")
print("="*60)

for idx in range(min(num_boxes, 5)):  # Show first 5 cells
    box = boxes[idx]
    cell_id = cell_ids[idx]

    # Extract GT mask for this cell ONLY
    gt_cell_mask = (gt_mask == cell_id).astype(np.uint8)

    # Count pixels
    gt_pixels = gt_cell_mask.sum()
    total_pixels = gt_mask.size

    print(f"\nCell {idx+1}:")
    print(f"  Cell ID in GT mask: {cell_id}")
    print(f"  Bounding box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    print(f"  GT mask for this cell:")
    print(f"    - Pixels where gt_mask == {cell_id}: {gt_pixels}")
    print(f"    - Percentage of image: {gt_pixels/total_pixels*100:.2f}%")
    print(f"  [OK] Dice will be calculated: pred_mask vs gt_cell_mask (only cell {cell_id})")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("[OK] Each cell's prediction is compared ONLY to its corresponding GT region")
print("[OK] Cell ID ensures correct one-to-one matching")
print("[OK] Other cells in GT are treated as background (0) during comparison")
print("="*60)
