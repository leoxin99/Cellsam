"""
Visualize morphological cleaning before/after comparison in Napari.
Shows the effect of each processing step on DAPI nucleus detection.
"""
import numpy as np
import tifffile
import napari
from pathlib import Path
from skimage import filters, morphology, measure
from scipy import ndimage

RAW_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
CH_DAPI = 4

def normalize_channel(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)

def main():
    tiff_files = sorted(RAW_DIR.glob("*.tiff"))
    if not tiff_files:
        print("No TIFF files found")
        return
    
    sample = tiff_files[0]
    print(f"Loading: {sample.name}")
    
    data = tifffile.imread(sample)
    dapi = data[CH_DAPI]
    
    # Normalize
    dapi_norm = normalize_channel(dapi)
    
    # Step 1: Otsu thresholding (raw binary)
    try:
        thresh = filters.threshold_otsu(dapi_norm)
    except:
        thresh = 0.3
    step1_binary = dapi_norm > thresh
    
    # Step 2: Morphological opening (remove noise)
    step2_opened = morphology.binary_opening(step1_binary, morphology.disk(3))
    
    # Step 3: Remove small objects
    step3_size_filter = morphology.remove_small_objects(step2_opened, min_size=500)
    
    # Step 4: Fill holes
    step4_filled = ndimage.binary_fill_holes(step3_size_filter)
    
    # Statistics
    print("\n=== Processing Statistics ===")
    print(f"Step 1 (Otsu):         {step1_binary.sum()} pixels foreground")
    print(f"Step 2 (Opening):      {step2_opened.sum()} pixels foreground ({step2_opened.sum() - step1_binary.sum():+d})")
    print(f"Step 3 (Size filter):  {step3_size_filter.sum()} pixels foreground ({step3_size_filter.sum() - step2_opened.sum():+d})")
    print(f"Step 4 (Fill holes):   {step4_filled.sum()} pixels foreground ({step4_filled.sum() - step3_size_filter.sum():+d})")
    
    # Count connected components
    labels_before = measure.label(step1_binary)
    labels_after = measure.label(step4_filled)
    print(f"\nNuclei count: {labels_before.max()} -> {labels_after.max()} (removed {labels_before.max() - labels_after.max()} noise regions)")
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Add layers
    viewer.add_image(dapi_norm, name='1. DAPI (normalized)', colormap='gray')
    viewer.add_image(step1_binary.astype(float), name='2. Otsu threshold', colormap='green', 
                     blending='additive', visible=False, opacity=0.7)
    viewer.add_image(step2_opened.astype(float), name='3. After opening (noise removed)', colormap='yellow', 
                     blending='additive', visible=False, opacity=0.7)
    viewer.add_image(step3_size_filter.astype(float), name='4. After size filter', colormap='magenta', 
                     blending='additive', visible=False, opacity=0.7)
    viewer.add_image(step4_filled.astype(float), name='5. Final (holes filled)', colormap='cyan', 
                     blending='additive', visible=True, opacity=0.7)
    
    # Difference layers
    noise_removed = (step1_binary.astype(float) - step4_filled.astype(float))
    noise_removed = np.clip(noise_removed, 0, 1)
    viewer.add_image(noise_removed, name='NOISE REMOVED (red)', colormap='red', 
                     blending='additive', visible=True, opacity=0.9)
    
    print("\n=== Napari Layers ===")
    print("  Green:  Raw Otsu threshold (with noise)")
    print("  Cyan:   Final cleaned result")
    print("  Red:    Noise that was removed")
    print("\nToggle layers 2-5 to see each step's effect")
    
    napari.run()

if __name__ == "__main__":
    main()
