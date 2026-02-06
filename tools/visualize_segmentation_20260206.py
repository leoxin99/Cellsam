"""
Standard Napari Visualization Tool for Segmentation Comparison
Created: 2026-02-06
Purpose: Universal tool to compare GT masks with predicted masks
         with all available channels (BF, Actn2, DAPI)

Usage:
    python tools/visualize_segmentation_20260206.py --sample 01
    python tools/visualize_segmentation_20260206.py --all

Features:
    - Maximally distinct colors for adjacent cells
    - All channels: BF, Actn2, DAPI, GT mask, Pred mask
    - Toggle between samples and layers
"""

import napari
import numpy as np
from pathlib import Path
import argparse
import tifffile
import imageio.v2 as imageio
from skimage.measure import regionprops, label
import colorsys


def generate_distinct_colors(n):
    """Generate n maximally distinct colors using golden ratio."""
    colors = []
    golden_ratio = 0.618033988749895
    hue = 0
    for i in range(n):
        hue = (hue + golden_ratio) % 1.0
        # High saturation and value for visibility
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(rgb)
    return colors


def relabel_for_contrast(mask):
    """
    Relabel mask so adjacent cells have maximally different IDs.
    Uses random shuffle to avoid sequential similar colors.
    """
    if mask.max() == 0:
        return mask
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) < 2:
        return mask
    
    # Create random permutation for label mapping
    np.random.seed(42)  # Reproducible
    shuffled = np.random.permutation(len(unique_labels)) + 1
    
    new_mask = np.zeros_like(mask)
    for old_id, new_id in zip(unique_labels, shuffled):
        new_mask[mask == old_id] = new_id
    
    return new_mask


class SegmentationViewer:
    """Unified segmentation viewer with all channels."""
    
    def __init__(self):
        self.results_dir = Path("docs/latest_results/images")
        self.data_dir = Path("data/raw/allen_segmented_fields_full")
        self.processed_dir = Path("data/processed/images")
        
    def find_samples(self):
        """Find available samples in results directory."""
        samples = []
        for gt_file in self.results_dir.glob("*_gt_mask.npy"):
            sample_id = gt_file.name.split("_")[0]
            samples.append(sample_id)
        return sorted(samples)
    
    def load_full_channels(self, sample_id):
        """Load all channels from original TIFF if available."""
        # Try to find matching TIFF
        for tiff_dir in [self.data_dir, self.processed_dir]:
            tiff_files = list(tiff_dir.glob(f"*{sample_id}*.tiff")) + \
                        list(tiff_dir.glob(f"*{sample_id}*.tif"))
            if tiff_files:
                tiff_path = tiff_files[0]
                img = tifffile.imread(tiff_path)
                
                # Allen dataset channel layout
                if img.ndim >= 2 and img.shape[0] >= 10:
                    return {
                        'bf': img[0],
                        'actn2': img[6] if img.shape[0] > 6 else None,
                        'dapi': img[7] if img.shape[0] > 7 else None,
                        'gt_mask': img[9] if img.shape[0] > 9 else None
                    }
        return None
    
    def visualize(self, sample_ids=None):
        """
        Main visualization function.
        
        Args:
            sample_ids: List of sample IDs to show, or None for all
        """
        viewer = napari.Viewer(title="CellSAM Segmentation Comparison")
        
        if sample_ids is None:
            sample_ids = self.find_samples()[:5]  # Limit to 5 samples
        
        if not sample_ids:
            print("No samples found in docs/latest_results/images/")
            return
        
        print(f"Loading {len(sample_ids)} samples...")
        
        for i, sample_id in enumerate(sample_ids):
            visible = (i == 0)  # Only first sample visible initially
            
            # Load from results directory
            gt_path = self.results_dir / f"{sample_id}_gt_mask.npy"
            pred_path = self.results_dir / f"{sample_id}_pred_mask.npy"
            img_files = list(self.results_dir.glob(f"{sample_id}_*.png"))
            
            if not gt_path.exists():
                print(f"Sample {sample_id}: No GT mask found")
                continue
            
            gt_mask = np.load(gt_path)
            
            # Try to load full channels from TIFF
            channels = self.load_full_channels(sample_id)
            
            # Add BF image
            if img_files:
                bf_img = imageio.imread(img_files[0])
                viewer.add_image(bf_img, name=f"S{sample_id}_BF",
                               visible=visible, colormap='gray')
            elif channels and channels['bf'] is not None:
                viewer.add_image(channels['bf'], name=f"S{sample_id}_BF",
                               visible=visible, colormap='gray')
            
            # Add Actn2 if available
            if channels and channels['actn2'] is not None:
                viewer.add_image(channels['actn2'], name=f"S{sample_id}_Actn2",
                               visible=visible, colormap='green', 
                               blending='additive', opacity=0.5)
            
            # Add DAPI if available  
            if channels and channels['dapi'] is not None:
                viewer.add_image(channels['dapi'], name=f"S{sample_id}_DAPI",
                               visible=visible, colormap='blue',
                               blending='additive', opacity=0.5)
            
            # Add GT mask with contrast-enhanced labels
            gt_relabeled = relabel_for_contrast(gt_mask.astype(np.int32))
            viewer.add_labels(gt_relabeled, name=f"S{sample_id}_GT_Mask",
                            visible=visible)
            
            # Add prediction mask if exists
            if pred_path.exists():
                pred_mask = np.load(pred_path)
                pred_relabeled = relabel_for_contrast(pred_mask.astype(np.int32))
                viewer.add_labels(pred_relabeled, name=f"S{sample_id}_Pred_Mask",
                                visible=False)  # Hidden by default for comparison
                
                n_gt = len(np.unique(gt_mask)) - 1
                n_pred = len(np.unique(pred_mask)) - 1
                print(f"  Sample {sample_id}: GT={n_gt} cells, Pred={n_pred} cells")
            else:
                n_gt = len(np.unique(gt_mask)) - 1
                print(f"  Sample {sample_id}: GT={n_gt} cells (no prediction)")
        
        print("\n" + "="*50)
        print("Visualization Tips:")
        print("  - Toggle eye icon to show/hide layers")
        print("  - Labels are re-colored for max contrast")
        print("  - Use opacity slider to blend channels")
        print("  - Press 'L' for label picker tool")
        print("="*50)
        
        napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation visualization tool")
    parser.add_argument('--sample', type=str, help='Specific sample ID (e.g., 01)')
    parser.add_argument('--all', action='store_true', help='Show all available samples')
    args = parser.parse_args()
    
    viewer = SegmentationViewer()
    
    if args.sample:
        viewer.visualize([args.sample])
    else:
        viewer.visualize()
