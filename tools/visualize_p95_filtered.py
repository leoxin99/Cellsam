"""
visualize_p95_filtered.py

功能: 找到并可视化 P95 双核距离 (~373px) 的案例 (确认过滤 min_area=500)
"""
import numpy as np
import tifffile
import napari
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage
from collections import defaultdict

sys.path.insert(0, 'src')

def detect_nuclei(dapi, min_area=500, max_area=30000):
    p2, p98 = np.percentile(dapi, [2, 98])
    if p98 > p2:
        dapi_norm = np.clip((dapi - p2) / (p98 - p2), 0, 1)
    else:
        return []
    
    try:
        thresh = filters.threshold_otsu(dapi_norm)
    except:
        thresh = 0.3
    
    binary = dapi_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    labels = measure.label(binary)
    nuclei = []
    for r in measure.regionprops(labels):
        if min_area <= r.area <= max_area:
            nuclei.append(r)
    return nuclei

# 373px 案例查找
data_dir = 'data/raw/allen_segmented_fields_full'
files = sorted(os.listdir(data_dir))[:50]

all_cases = []
for fname in files:
    img = tifffile.imread(os.path.join(data_dir, fname))
    dapi = img[4]
    gt = img[9]
    
    nuclei = detect_nuclei(dapi, min_area=500, max_area=30000)
    
    cell_nuclei = defaultdict(list)
    for nuc in nuclei:
        y, x = int(nuc.centroid[0]), int(nuc.centroid[1])
        if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
            cell_id = gt[y, x]
            if cell_id > 0:
                cell_nuclei[cell_id].append(nuc)
    
    for cell_id, nucs in cell_nuclei.items():
        if len(nucs) >= 2:
            for i in range(len(nucs)):
                for j in range(i+1, len(nucs)):
                    d = np.sqrt(
                        (nucs[i].centroid[0] - nucs[j].centroid[0])**2 +
                        (nucs[i].centroid[1] - nucs[j].centroid[1])**2
                    )
                    all_cases.append({
                        'file': fname,
                        'cell_id': cell_id,
                        'distance': d,
                        'nuc1': nucs[i],
                        'nuc2': nucs[j]
                    })

target_dist = 373
closest = min(all_cases, key=lambda x: abs(x['distance'] - target_dist))

print(f"找到 P95 (~373px) 双核案例 (Filter: min_area=500):")
print(f"  文件: {closest['file']}")
print(f"  GT Cell ID: {closest['cell_id']}")
print(f"  核间距: {closest['distance']:.1f} px")
print(f"  核1: 面积={closest['nuc1'].area} px (>500)")
print(f"  核2: 面积={closest['nuc2'].area} px (>500)")

# Visualizing
print("打开 Napari...")
img = tifffile.imread(os.path.join(data_dir, closest['file']))
bf = img[0]
dapi = img[4]
gt = img[9]

viewer = napari.Viewer(title=f"P95 Binuclear Filtered: dist={closest['distance']:.0f}px")
viewer.add_image(bf, name='Brightfield')
viewer.add_image(dapi, name='DAPI', colormap='blue', visible=False)
viewer.add_labels(gt, name='GT_Mask')

target_mask = (gt == closest['cell_id']).astype(int)
viewer.add_labels(target_mask * 2, name=f"Target_Cell_{closest['cell_id']}", opacity=0.5)

nuc_points = np.array([
    [closest['nuc1'].centroid[0], closest['nuc1'].centroid[1]],
    [closest['nuc2'].centroid[0], closest['nuc2'].centroid[1]]
])
viewer.add_points(nuc_points, size=20, face_color='yellow', name='Nucleus_Centers')

line = np.array([[
    [closest['nuc1'].centroid[0], closest['nuc1'].centroid[1]],
    [closest['nuc2'].centroid[0], closest['nuc2'].centroid[1]]
]])
viewer.add_shapes(line, shape_type='line', edge_color='yellow', edge_width=3, name=f"Distance_{closest['distance']:.0f}px")

napari.run()
