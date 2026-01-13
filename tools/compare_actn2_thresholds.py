"""
Napari visualization to compare Actn2 threshold values.
Shows side-by-side comparison of 0.05, 0.1, 0.15, 0.2 thresholds.
"""
import numpy as np
import tifffile
import napari
from pathlib import Path
from skimage import morphology

RAW_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9

def normalize_channel(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)

def main():
    # Load first sample
    tiff_files = sorted(RAW_DIR.glob("*.tiff"))
    if not tiff_files:
        print("No TIFF files found")
        return
    
    sample = tiff_files[0]
    print(f"Loading: {sample.name}")
    
    data = tifffile.imread(sample)
    actn2 = data[CH_ACTN2]
    dapi = data[CH_DAPI]
    mask = data[CH_MASK]
    bf = data[CH_BRIGHTFIELD]
    
    # Normalize
    actn2_norm = normalize_channel(actn2)
    dapi_norm = normalize_channel(dapi)
    bf_norm = normalize_channel(bf)
    
    # Create threshold masks
    thresholds = [0.05, 0.1, 0.15, 0.2]
    actn2_masks = {}
    
    for thresh in thresholds:
        mask_binary = actn2_norm > thresh
        # Dilate slightly for visualization
        mask_dilated = morphology.binary_dilation(mask_binary, morphology.disk(3))
        actn2_masks[thresh] = mask_dilated
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Add base layers
    viewer.add_image(bf_norm, name='Brightfield', colormap='gray', visible=True)
    viewer.add_image(actn2_norm, name='Actn2 (raw)', colormap='green', 
                     blending='additive', visible=False)
    viewer.add_image(dapi_norm, name='DAPI', colormap='blue', 
                     blending='additive', visible=False)
    viewer.add_labels(mask.astype(np.int32), name='GT Mask', visible=False)
    
    # Add threshold comparison layers
    colors = ['cyan', 'yellow', 'magenta', 'red']
    for i, thresh in enumerate(thresholds):
        viewer.add_labels(
            actn2_masks[thresh].astype(np.int32), 
            name=f'Actn2 > {thresh}',
            visible=(i == 1),  # Only show 0.1 by default
            opacity=0.5
        )
    
    print("\n=== Actn2 Threshold Comparison ===")
    print("Toggle layers to compare different thresholds:")
    for thresh in thresholds:
        coverage = actn2_masks[thresh].sum() / actn2_masks[thresh].size * 100
        print(f"  Threshold {thresh}: {coverage:.1f}% coverage")
    
    print("\nRecommendation: Choose threshold with ~30-50% coverage")
    print("              that captures sarcomere regions without background")
    
    napari.run()

if __name__ == "__main__":
    main()
