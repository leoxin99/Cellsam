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
Napari Visualization for E29 Training Results
Created: 2026-02-06
Purpose: Visualize E29 (bf_instance_p1) segmentation results
"""

import napari
import numpy as np
import torch
from pathlib import Path
import tifffile

# Use existing latest_results if available
RESULTS_DIR = Path("docs/latest_results/images")

def load_and_visualize():
    """Load sample images and masks from latest_results."""
    
    viewer = napari.Viewer()
    
    # Find available samples
    samples = []
    for i in range(1, 10):
        prefix = f"{i:02d}_"
        img_files = list(RESULTS_DIR.glob(f"{prefix}*.png"))
        gt_mask = RESULTS_DIR / f"{prefix}gt_mask.npy"
        pred_mask = RESULTS_DIR / f"{prefix}pred_mask.npy"
        
        if img_files and gt_mask.exists():
            samples.append({
                'id': i,
                'image': img_files[0],
                'gt_mask': gt_mask,
                'pred_mask': pred_mask if pred_mask.exists() else None
            })
    
    if not samples:
        print("No samples found in docs/latest_results/images/")
        print("Please run inference first to generate results.")
        return
    
    print(f"Found {len(samples)} samples")
    
    # Load first 2 samples
    for idx, sample in enumerate(samples[:2]):
        print(f"\nLoading sample {sample['id']}...")
        
        # Load image
        import imageio
        img = imageio.imread(sample['image'])
        
        # Load GT mask
        gt_mask = np.load(sample['gt_mask'])
        
        # Add to viewer
        viewer.add_image(img, name=f"Sample_{sample['id']}_Image", 
                        visible=(idx == 0))
        viewer.add_labels(gt_mask.astype(np.int32), name=f"Sample_{sample['id']}_GT",
                         visible=(idx == 0))
        
        # Load pred mask if exists
        if sample['pred_mask']:
            pred_mask = np.load(sample['pred_mask'])
            viewer.add_labels(pred_mask.astype(np.int32), name=f"Sample_{sample['id']}_Pred",
                            visible=(idx == 0))
            print(f"  GT cells: {len(np.unique(gt_mask)) - 1}")
            print(f"  Pred cells: {len(np.unique(pred_mask)) - 1}")
        else:
            print(f"  GT cells: {len(np.unique(gt_mask)) - 1}")
            print("  No prediction mask found")
    
    print("\n✅ Visualization ready!")
    print("Use the layer list to toggle between samples and GT/Pred masks")
    
    napari.run()

if __name__ == "__main__":
    load_and_visualize()
