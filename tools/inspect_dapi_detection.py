"""
Napari visualization for manual inspection of DAPI detection results.
Displays all channels: BF, DAPI, Actn2, GT mask, Pred mask, and detected boxes.
"""
import numpy as np
import tifffile
import napari
from pathlib import Path
from skimage import measure, filters, morphology
from scipy import ndimage

RAW_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9

# Test sample IDs
TEST_SAMPLES = [
    "cf4fb0e8_5500000013_63X_20190807_S1_P28_C1",
    "ebfc8c4d_5500000013_63X_20190807_S1_P30_B4",  # +6 detections
    "ec4c125c_5500000013_63X_20190807_S1_P27_C4",
]

def normalize_channel(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)

def detect_nuclei_dapi(dapi_channel, min_nucleus_area=500, max_nucleus_area=30000):
    """New params: 500-30000, with opening."""
    img_norm = normalize_channel(dapi_channel)
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    labels = measure.label(binary)
    valid_regions = []
    
    for region in measure.regionprops(labels):
        if min_nucleus_area <= region.area <= max_nucleus_area:
            valid_regions.append(region)
    
    return valid_regions, labels

def main():
    # Find samples
    tiff_files = sorted(RAW_DIR.glob("*.tiff"))
    
    # Load first sample (or specific high-detection sample)
    target_id = TEST_SAMPLES[1]  # ebfc8c4d - has +6 extra detections
    
    sample = None
    for tf in tiff_files:
        if target_id.split("_")[0] in tf.name:
            sample = tf
            break
    
    if sample is None:
        sample = tiff_files[0]
    
    print(f"Loading: {sample.name}")
    
    data = tifffile.imread(sample)
    bf = data[CH_BRIGHTFIELD]
    dapi = data[CH_DAPI]
    actn2 = data[CH_ACTN2]
    gt_mask = data[CH_MASK]
    
    # Normalize
    bf_norm = normalize_channel(bf)
    dapi_norm = normalize_channel(dapi)
    actn2_norm = normalize_channel(actn2)
    
    # Detect nuclei
    regions, nucleus_labels = detect_nuclei_dapi(dapi)
    print(f"Detected {len(regions)} nuclei")
    print(f"GT cells: {len(np.unique(gt_mask)) - 1}")
    
    # Create detection mask
    detection_mask = np.zeros_like(dapi, dtype=np.int32)
    for i, r in enumerate(regions):
        coords = r.coords
        detection_mask[coords[:, 0], coords[:, 1]] = i + 1
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Add image layers
    viewer.add_image(bf_norm, name='0_Brightfield', colormap='gray')
    viewer.add_image(dapi_norm, name='1_DAPI (nuclei)', colormap='blue', 
                     blending='additive', visible=True)
    viewer.add_image(actn2_norm, name='2_Actn2 (sarcomere)', colormap='green', 
                     blending='additive', visible=False)
    
    # Add mask layers
    viewer.add_labels(gt_mask.astype(np.int32), name='3_GT_Mask (yellow)', 
                      visible=True, opacity=0.5)
    viewer.add_labels(detection_mask, name='4_Detected_Nuclei (cyan)', 
                      visible=True, opacity=0.7)
    
    # Add boxes as shapes
    boxes = []
    for r in regions:
        y1, x1, y2, x2 = r.bbox
        # Rectangle: [[y1, x1], [y1, x2], [y2, x2], [y2, x1]]
        box = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
        boxes.append(box)
    
    if boxes:
        viewer.add_shapes(boxes, shape_type='polygon', 
                          edge_color='magenta', face_color='transparent',
                          edge_width=2, name='5_Detected_Boxes')
    
    print("\n=== Manual Inspection Guide ===")
    print("Layer 0: Brightfield (cell structure)")
    print("Layer 1: DAPI (nuclei - blue)")
    print("Layer 2: Actn2 (sarcomere - green, toggle for boundary reference)")
    print("Layer 3: GT Mask (yellow)")
    print("Layer 4: Detected Nuclei")
    print("Layer 5: Detected Boxes (magenta)")
    print("\nCheck: Are extra detections real cells or noise?")
    
    napari.run()

if __name__ == "__main__":
    main()
