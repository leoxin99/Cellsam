"""
Napari visualization: Compare nucleus bounding boxes vs GT cell boxes.
Shows 10 samples with both box types overlaid.
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

def normalize_channel(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)

def detect_nuclei(dapi_channel, min_area=500):
    """Simple nucleus detection."""
    img_norm = normalize_channel(dapi_channel)
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = ndimage.binary_fill_holes(binary)
    
    return measure.label(binary)

def bbox_to_rect(bbox, color='red'):
    """Convert bbox [y1, x1, y2, x2] to rectangle coordinates."""
    y1, x1, y2, x2 = bbox
    # Return corners: top-left, top-right, bottom-right, bottom-left, back to top-left
    return np.array([
        [y1, x1], [y1, x2], [y2, x2], [y2, x1], [y1, x1]
    ])

def main():
    tiff_files = sorted(RAW_DIR.glob("*.tiff"))[:10]  # 10 samples
    print(f"Processing {len(tiff_files)} samples")
    
    viewer = napari.Viewer()
    
    for idx, tiff_path in enumerate(tiff_files):
        print(f"Loading {idx+1}/10: {tiff_path.name[:30]}...")
        
        data = tifffile.imread(tiff_path)
        dapi = data[CH_DAPI]
        mask = data[CH_MASK]
        bf = data[CH_BRIGHTFIELD]
        
        bf_norm = normalize_channel(bf)
        
        if idx == 0:
            # Add base image for first sample only
            viewer.add_image(bf_norm, name=f'Brightfield', colormap='gray')
            viewer.add_labels(mask.astype(np.int32), name=f'GT Mask', visible=False)
        
        # Detect nuclei
        nucleus_labels = detect_nuclei(dapi)
        
        # Collect boxes
        nucleus_rects = []
        gt_cell_rects = []
        expanded_rects = []  # 4x expansion
        
        for cell_region in measure.regionprops(mask.astype(np.int32)):
            gt_bbox = cell_region.bbox
            gt_cell_rects.append(bbox_to_rect(gt_bbox))
            
            # Find nucleus within this cell
            cell_mask = (mask == cell_region.label)
            overlapping_nuclei = nucleus_labels * cell_mask
            
            if overlapping_nuclei.max() > 0:
                nucleus_regions = measure.regionprops(overlapping_nuclei.astype(np.int32))
                if nucleus_regions:
                    largest_nucleus = max(nucleus_regions, key=lambda r: r.area)
                    nuc_bbox = largest_nucleus.bbox
                    nucleus_rects.append(bbox_to_rect(nuc_bbox))
                    
                    # Compute 4x expanded box from nucleus
                    y1, x1, y2, x2 = nuc_bbox
                    h, w = y2 - y1, x2 - x1
                    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                    new_h, new_w = h * 4, w * 4
                    exp_bbox = [
                        int(max(0, cy - new_h/2)),
                        int(max(0, cx - new_w/2)),
                        int(min(mask.shape[0], cy + new_h/2)),
                        int(min(mask.shape[1], cx + new_w/2))
                    ]
                    expanded_rects.append(bbox_to_rect(exp_bbox))
        
        # Add shapes for first sample
        if idx == 0:
            if nucleus_rects:
                viewer.add_shapes(
                    nucleus_rects,
                    shape_type='path',
                    edge_color='cyan',
                    edge_width=2,
                    name='Nucleus Box (detected)',
                    face_color='transparent'
                )
            if gt_cell_rects:
                viewer.add_shapes(
                    gt_cell_rects,
                    shape_type='path',
                    edge_color='yellow',
                    edge_width=2,
                    name='GT Cell Box',
                    face_color='transparent'
                )
            if expanded_rects:
                viewer.add_shapes(
                    expanded_rects,
                    shape_type='path',
                    edge_color='magenta',
                    edge_width=2,
                    name='Nucleus x4 Expanded',
                    face_color='transparent'
                )
        
        # Print stats for this sample
        print(f"  GT cells: {len(gt_cell_rects)}, Matched nuclei: {len(nucleus_rects)}")
    
    print("\n=== Legend ===")
    print("  Cyan:    Detected nucleus box")
    print("  Yellow:  GT cell box (ground truth)")
    print("  Magenta: Nucleus x4 expanded (recommended)")
    print("\nCompare yellow (GT) with magenta (4x) to verify expansion factor")
    
    napari.run()

if __name__ == "__main__":
    main()
