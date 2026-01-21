"""View min and max area cells in Napari for verification."""
import numpy as np
import tifffile
import napari
from pathlib import Path

raw_dir = Path('d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full')
tiff_files = sorted(raw_dir.glob('*.tiff'))

print('Searching for Min/Max cells...')

min_area = float('inf')
max_area = 0
min_info = None
max_info = None

for i, f in enumerate(tiff_files):
    data = tifffile.imread(f)
    gt = data[9]
    for cell_id in [x for x in np.unique(gt) if x > 0]:
        area = (gt == cell_id).sum()
        if area < min_area:
            min_area = area
            min_info = {'file': f, 'cell_id': cell_id, 'area': area}
        if area > max_area:
            max_area = area
            max_info = {'file': f, 'cell_id': cell_id, 'area': area}

print(f"Min cell: {min_info['file'].stem}, cell_id={min_info['cell_id']}, area={min_info['area']}")
print(f"Max cell: {max_info['file'].stem}, cell_id={max_info['cell_id']}, area={max_info['area']}")

# Load data
min_data = tifffile.imread(min_info['file'])
max_data = tifffile.imread(max_info['file'])

# Open in Napari
v = napari.Viewer()

# Min cell image
min_gt = min_data[9]
min_cell_mask = (min_gt == min_info['cell_id']).astype(np.int32) * 255
v.add_image(min_data[0], name=f"MIN_BF_{min_info['file'].stem}", colormap='gray')
v.add_image(min_data[4], name='MIN_DAPI', colormap='blue', blending='additive', visible=False)
v.add_labels(min_gt.astype(np.int32), name='MIN_GT_all')
v.add_labels(min_cell_mask, name=f"MIN_cell_area{min_info['area']}")

# Max cell image  
max_gt = max_data[9]
max_cell_mask = (max_gt == max_info['cell_id']).astype(np.int32) * 255
v.add_image(max_data[0], name=f"MAX_BF_{max_info['file'].stem}", colormap='gray')
v.add_image(max_data[4], name='MAX_DAPI', colormap='blue', blending='additive', visible=False)
v.add_labels(max_gt.astype(np.int32), name='MAX_GT_all')
v.add_labels(max_cell_mask, name=f"MAX_cell_area{max_info['area']}")

print('Napari launched with Min and Max cell images')
napari.run()
