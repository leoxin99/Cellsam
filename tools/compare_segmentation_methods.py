"""
Napari Visualization: Compare Current Results with GT Box + CellSAM Baseline
Created: 2026-02-06
Purpose: 
- Show current E29 training results
- Compare with GT box + original cellSAM (no fine-tuning)
- Display Actn2, DAPI channels for context
"""

import napari
import numpy as np
from pathlib import Path
import tifffile
import imageio
from skimage.measure import regionprops, label

# Paths
RESULTS_DIR = Path("docs/latest_results/images")
DATA_DIR = Path("data/raw/allen_segmented_fields_full")
PROCESSED_DIR = Path("data/processed")

def get_distinct_colormap():
    """Create a colormap that maximally separates adjacent colors."""
    # Use a shuffled colormap to avoid similar colors for adjacent IDs
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    
    # Get a qualitative colormap with many distinct colors
    cmap = colormaps.get_cmap('tab20')
    colors = [cmap(i) for i in range(20)]
    
    # Shuffle to avoid adjacent similar colors
    np.random.seed(42)
    np.random.shuffle(colors)
    
    return colors

def relabel_with_spacing(mask):
    """Relabel masks to maximize color difference between adjacent cells."""
    if mask.max() == 0:
        return mask
    
    # Get regions and their neighbors
    regions = regionprops(mask)
    n_regions = len(regions)
    
    if n_regions < 2:
        return mask
    
    # Simple approach: renumber so adjacent IDs are far apart
    new_mask = np.zeros_like(mask)
    old_labels = [r.label for r in regions]
    
    # Interleave odd and even indices
    new_labels = []
    for i in range(0, n_regions, 2):
        new_labels.append(old_labels[i])
    for i in range(1, n_regions, 2):
        new_labels.append(old_labels[i])
    
    # Create mapping
    for new_id, old_id in enumerate(old_labels, 1):
        new_mask[mask == old_id] = new_id
    
    return new_mask

def load_sample_with_channels(sample_id):
    """Load sample with all channels: BF, Actn2, DAPI."""
    # Find the TIFF file
    tiff_pattern = f"*{sample_id}*.tiff"
    tiff_files = list(DATA_DIR.glob(tiff_pattern))
    
    if not tiff_files:
        # Try processed directory
        tiff_files = list(PROCESSED_DIR.glob(f"images/{sample_id}*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF found for sample {sample_id}")
        return None, None, None, None
    
    tiff_path = tiff_files[0]
    print(f"Loading: {tiff_path.name}")
    
    img = tifffile.imread(tiff_path)
    
    # Allen dataset channels (based on previous analysis)
    # Ch0: BF, Ch1-5: Fluorescence, Ch6: Actn2, Ch7: DAPI, Ch8: Binary, Ch9: Instance
    if img.ndim == 2:
        return img, None, None, None
    
    bf_channel = img[0] if img.shape[0] > 0 else None
    actn2_channel = img[6] if img.shape[0] > 6 else None  
    dapi_channel = img[7] if img.shape[0] > 7 else None
    gt_mask = img[9] if img.shape[0] > 9 else None
    
    return bf_channel, actn2_channel, dapi_channel, gt_mask

def visualize_comparison():
    """Main visualization function."""
    viewer = napari.Viewer()
    
    # Load sample 01 and 02 from latest_results
    for sample_num in [1, 2]:
        prefix = f"{sample_num:02d}_"
        
        # Load prediction mask
        pred_path = RESULTS_DIR / f"{prefix}pred_mask.npy"
        gt_path = RESULTS_DIR / f"{prefix}gt_mask.npy"
        
        if not gt_path.exists():
            print(f"Sample {sample_num}: No GT mask found")
            continue
        
        gt_mask = np.load(gt_path)
        pred_mask = np.load(pred_path) if pred_path.exists() else None
        
        # Try to load original channels
        img_files = list(RESULTS_DIR.glob(f"{prefix}*.png"))
        if img_files:
            bf_img = imageio.imread(img_files[0])
        else:
            bf_img = None
        
        # Relabel for better color separation
        gt_relabeled = relabel_with_spacing(gt_mask.astype(np.int32))
        
        visible = (sample_num == 1)
        
        # Add layers
        if bf_img is not None:
            viewer.add_image(bf_img, name=f"S{sample_num}_BF", 
                           visible=visible, colormap='gray')
        
        viewer.add_labels(gt_relabeled, name=f"S{sample_num}_GT_Mask",
                         visible=visible)
        
        if pred_mask is not None:
            pred_relabeled = relabel_with_spacing(pred_mask.astype(np.int32))
            viewer.add_labels(pred_relabeled, name=f"S{sample_num}_Pred_Mask",
                            visible=visible)
            
            # Calculate metrics
            n_gt = len(np.unique(gt_mask)) - 1
            n_pred = len(np.unique(pred_mask)) - 1
            print(f"\nSample {sample_num}:")
            print(f"  GT cells: {n_gt}")
            print(f"  Pred cells: {n_pred}")
    
    print("\n" + "="*50)
    print("Visualization Tips:")
    print("- Toggle layers to compare GT vs Pred")
    print("- Colors are shuffled to distinguish adjacent cells")
    print("- Zoom into boundaries to check jagged edges")
    print("="*50)
    
    napari.run()

if __name__ == "__main__":
    visualize_comparison()
