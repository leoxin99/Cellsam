"""
visualize_3000px_nuclei.py

功能: 找到面积接近 3000 px 的 GT 核并可视化，确认其是否为有效核
"""
import numpy as np
import tifffile
import napari
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage

sys.path.insert(0, 'src')

def detect_nuclei(dapi):
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
    label_img = measure.label(binary)
    return measure.regionprops(label_img)

data_dir = 'data/raw/allen_segmented_fields_full'
files = sorted(os.listdir(data_dir))[:50]

candidates = []
target_area = 3000
tolerance = 500 # 2500-3500

print(f"正在寻找面积接近 {target_area} px 的 GT 核...")
for fname in files:
    img = tifffile.imread(os.path.join(data_dir, fname))
    dapi = img[4]
    gt = img[9]
    
    nuclei = detect_nuclei(dapi)
    
    for r in nuclei:
        # 必须在 GT 内
        y, x = int(r.centroid[0]), int(r.centroid[1])
        if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
            if gt[y, x] > 0:
                if abs(r.area - target_area) < tolerance:
                    candidates.append({
                        'file': fname,
                        'area': r.area,
                        'centroid': r.centroid,
                        'diff': abs(r.area - target_area)
                    })

candidates.sort(key=lambda x: x['diff'])

if not candidates:
    print("未找到匹配的核")
else:
    best = candidates[0]
    print(f"\n找到最接近的核:")
    print(f"  文件: {best['file']}")
    print(f"  面积: {best['area']} px (Diff: {best['diff']})")
    
    print("\n打开 Napari...")
    img = tifffile.imread(os.path.join(data_dir, best['file']))
    bf = img[0]
    dapi = img[4]
    gt = img[9]
    
    viewer = napari.Viewer(title=f"Validator: Area={best['area']} px")
    viewer.add_image(bf, name='Brightfield')
    viewer.add_image(dapi, name='DAPI', colormap='blue', visible=False)
    viewer.add_labels(gt, name='GT_Mask')
    
    y, x = best['centroid']
    viewer.add_points([[y, x]], size=30, face_color='green', name='Target_Nucleus')
    
    napari.run()
