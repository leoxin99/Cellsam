# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: One-off experiment/visualization script (Phase B cleanup)
# Replacement entry points:
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Baseline Test: GT Box + Original CellSAM (No Fine-tuning)
Created: 2026-02-06
Purpose: Evaluate pre-trained CellSAM with GT boxes as baseline for comparison

This establishes the baseline performance:
- If GT box + pre-trained CellSAM is already good → fine-tuning may have limited value
- If poor → fine-tuning is necessary
"""

import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cellSAM import get_model
    from cellSAM.model import segment_cellular_image
except ImportError:
    print("❌ Error: cellSAM not found. Please run in cellsam conda environment:")
    print("   conda activate cellsam")
    print("   python tools/baseline_gt_cellsam_20260206.py --samples 10")
    sys.exit(1)

from src.augmented_dataset import AugmentedAllenDataset, load_split_ids
from skimage.measure import regionprops
import tifffile


def compute_dice(pred, target):
    """Compute Dice coefficient."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return (2 * intersection) / (union + 1e-8)


def evaluate_baseline(num_samples=20, device='cuda'):
    """
    Run GT box + pre-trained CellSAM baseline evaluation.
    
    Args:
        num_samples: Number of samples to evaluate
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of metrics
    """
    print("="*60)
    print("GT Box + Pre-trained CellSAM Baseline Evaluation")
    print("="*60)
    
    # Load pre-trained CellSAM (no fine-tuning)
    print("\n1. Loading pre-trained CellSAM model...")
    model = get_model()
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded")
    
    # Load validation dataset
    print("\n2. Loading validation dataset...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",  # Parent dir - code auto-adds images/ and masks/
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=True,
        is_training=False
    )
    print(f"   ✅ Loaded {len(dataset)} samples")
    
    # Limit samples for quick evaluation
    sample_indices = list(range(min(num_samples, len(dataset))))
    
    # Metrics storage
    instance_dices = []
    semantic_dices = []
    cell_counts = {'gt': [], 'pred': []}
    
    print(f"\n3. Evaluating {len(sample_indices)} samples...")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices):
            sample = dataset[idx]
            
            image = sample['image'].unsqueeze(0).to(device)  # (1, 3, H, W)
            gt_mask = sample['mask'].numpy()  # (H, W) instance mask
            boxes = sample['boxes']  # List of [x1, y1, x2, y2]
            
            if len(boxes) == 0:
                continue
            
            # Combined prediction mask
            combined_pred = np.zeros_like(gt_mask)
            cell_id = 1
            
            # Run SAM inference for each GT box
            boxes_array = boxes.numpy() if torch.is_tensor(boxes) else boxes
            boxes_list = [box.tolist() if hasattr(box, 'tolist') else list(box) for box in boxes_array]
            
            # Get image as numpy array for segment_cellular_image
            img_np = sample['image'].numpy().transpose(1, 2, 0)  # (H, W, C)
            
            try:
                # Use segment_cellular_image with all boxes at once
                mask, _, _ = segment_cellular_image(
                    img_np, 
                    model, 
                    normalize=False,  # Already normalized by dataset
                    bounding_boxes=boxes_list,
                    device=device
                )
                combined_pred = mask
                
            except Exception as e:
                print(f"   Warning: Inference failed for sample {idx}: {e}")
                combined_pred = np.zeros_like(gt_mask)
            
            # Compute instance Dice for each GT cell
            gt_regions = regionprops(gt_mask)
            for region in gt_regions:
                gt_cell = (gt_mask == region.label)
                
                # Find overlapping predicted region
                overlap_ids = np.unique(combined_pred[gt_cell])
                overlap_ids = overlap_ids[overlap_ids > 0]
                
                if len(overlap_ids) > 0:
                    pred_cell = np.isin(combined_pred, overlap_ids)
                    dice = compute_dice(pred_cell.astype(float), gt_cell.astype(float))
                    instance_dices.append(dice)
                else:
                    instance_dices.append(0.0)  # False negative
            
            # Semantic Dice
            gt_binary = (gt_mask > 0).astype(float)
            pred_binary = (combined_pred > 0).astype(float)
            semantic_dice = compute_dice(pred_binary, gt_binary)
            semantic_dices.append(semantic_dice)
            
            # Cell counts
            cell_counts['gt'].append(len(gt_regions))
            cell_counts['pred'].append(len(np.unique(combined_pred)) - 1)
    
    # Compute summary metrics
    results = {
        'method': 'GT_Box_PretrainedCellSAM',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_samples': len(sample_indices),
        'num_cells_evaluated': len(instance_dices),
        'instance_dice_mean': float(np.mean(instance_dices)),
        'instance_dice_std': float(np.std(instance_dices)),
        'semantic_dice_mean': float(np.mean(semantic_dices)),
        'semantic_dice_std': float(np.std(semantic_dices)),
        'avg_gt_cells_per_image': float(np.mean(cell_counts['gt'])),
        'avg_pred_cells_per_image': float(np.mean(cell_counts['pred'])),
    }
    
    print("\n" + "="*60)
    print("Results: GT Box + Pre-trained CellSAM")
    print("="*60)
    print(f"Instance Dice: {results['instance_dice_mean']:.4f} ± {results['instance_dice_std']:.4f}")
    print(f"Semantic Dice: {results['semantic_dice_mean']:.4f} ± {results['semantic_dice_std']:.4f}")
    print(f"Cells evaluated: {results['num_cells_evaluated']}")
    print("="*60)
    
    # Save results
    output_path = Path('experiments/baseline_gt_cellsam_20260206.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    evaluate_baseline(num_samples=args.samples, device=args.device)
