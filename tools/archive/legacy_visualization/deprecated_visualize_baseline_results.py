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
Visualize GT Box + Pretrained CellSAM Baseline Results
Created: 2026-02-06
Purpose: Show segmentation results from baseline_gt_cellsam_20260206.py
"""

import napari
import numpy as np
from pathlib import Path
import sys
import colorsys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.augmented_dataset import AugmentedAllenDataset, load_split_ids

try:
    from cellSAM import get_model
    from cellSAM.model import segment_cellular_image
except ImportError:
    print("❌ Error: cellSAM not found. Please run: conda activate cellsam")
    sys.exit(1)


def generate_distinct_colors(n):
    """Generate n maximally distinct colors."""
    colors = []
    golden_ratio = 0.618033988749895
    hue = 0.1
    for i in range(n):
        hue = (hue + golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
        colors.append(rgb)
    return colors


def relabel_for_contrast(mask):
    """Relabel mask for maximum color contrast."""
    if mask.max() == 0:
        return mask
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) < 2:
        return mask
    
    np.random.seed(42)
    shuffled = np.random.permutation(len(unique_labels)) + 1
    
    new_mask = np.zeros_like(mask)
    for old_id, new_id in zip(unique_labels, shuffled):
        new_mask[mask == old_id] = new_id
    
    return new_mask


def visualize_baseline_results(num_samples=2, device='cuda'):
    """
    Run GT box + CellSAM inference and visualize results in napari.
    
    Data source: data/processed/ (1024x1024, 3-channel: [BF, DAPI, Actn2])
    """
    print("="*60)
    print("GT Box + Pre-trained CellSAM Baseline Visualization")
    print("="*60)
    
    # Load model
    print("\n1. Loading CellSAM model...")
    model = get_model()
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded")
    
    # Load validation samples
    print("\n2. Loading validation dataset...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=True,
        is_training=False
    )
    print(f"   ✅ Loaded {len(dataset)} samples")
    
    # Create napari viewer
    viewer = napari.Viewer(title="GT Box + CellSAM Baseline Results")
    
    print(f"\n3. Processing {num_samples} samples...")
    
    import torch
    
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        
        # Get image and GT mask
        image = sample['image']  # (3, H, W) tensor
        gt_mask = sample['mask'].numpy()  # (H, W)
        boxes = sample['boxes']
        
        # Convert to numpy for visualization
        img_np = image.numpy().transpose(1, 2, 0)  # (H, W, 3)
        bf_channel = img_np[:, :, 0]  # BF is channel 0
        
        # Get boxes as list
        boxes_array = boxes.numpy() if torch.is_tensor(boxes) else boxes
        boxes_list = [box.tolist() if hasattr(box, 'tolist') else list(box) for box in boxes_array]
        
        # Run CellSAM inference with GT boxes
        print(f"   Processing sample {idx+1}/{num_samples}...")
        try:
            pred_mask, _, _ = segment_cellular_image(
                img_np,
                model,
                normalize=False,
                bounding_boxes=boxes_list,
                device=device
            )
        except Exception as e:
            print(f"   Warning: Inference failed: {e}")
            pred_mask = np.zeros_like(gt_mask)
        
        # Relabel for better visualization
        gt_relabeled = relabel_for_contrast(gt_mask.astype(np.int32))
        pred_relabeled = relabel_for_contrast(pred_mask.astype(np.int32))
        
        visible = (idx == 0)
        
        # Add layers
        viewer.add_image(bf_channel, name=f"S{idx+1}_BF", 
                        visible=visible, colormap='gray')
        viewer.add_labels(gt_relabeled, name=f"S{idx+1}_GT_Mask",
                         visible=visible)
        viewer.add_labels(pred_relabeled, name=f"S{idx+1}_Pred_CellSAM",
                         visible=False)
        
        n_gt = len(np.unique(gt_mask)) - 1
        n_pred = len(np.unique(pred_mask)) - 1
        print(f"   Sample {idx+1}: GT={n_gt} cells, Pred={n_pred} cells")
    
    print("\n" + "="*60)
    print("Data Source Information:")
    print("  - Images: data/processed/images/*.npy")
    print("  - Format: (3, 1024, 1024) = [BF, DAPI, Actn2]")
    print("  - GT Boxes: Extracted from GT mask using regionprops")
    print("  - Model: Pre-trained CellSAM (no fine-tuning)")
    print("="*60)
    print("\nVisualization Tips:")
    print("  - Toggle layers to compare GT vs Prediction")
    print("  - GT_Mask = Ground Truth")
    print("  - Pred_CellSAM = Pre-trained model output")
    print("="*60)
    
    napari.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=2, help='Number of samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    visualize_baseline_results(num_samples=args.samples, device=args.device)
