"""
view_minmax_gt_nuclei.py

功能: 可视化 GT 中 >500px 的核，检查是否存在极小核/碎片
创建日期: 2026-01-26

逻辑:
1. 找出所有 GT 覆盖的核
2. 按面积排序
3. 可视化：
   - 最小的 5 个核 (>= 500 px) -> 检查是否是碎片
   - 最大的 3 个核 -> 检查是否过度合并
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

all_gt_nuclei = []

print("正在收集 GT 核信息...")
for fname in files:
    img = tifffile.imread(os.path.join(data_dir, fname))
    dapi = img[4]
    gt = img[9]
    
    nuclei = detect_nuclei(dapi)
    
    for r in nuclei:
        # 必须是 GT 核
        y, x = int(r.centroid[0]), int(r.centroid[1])
        if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
            if gt[y, x] > 0:
                # 只有 >= 500px 的才算"有效"
                if r.area >= 500:
                    all_gt_nuclei.append({
                        'file': fname,
                        'area': r.area,
                        'centroid': r.centroid,
                        'bbox': r.bbox,
                        'label': r.label
                    })

# 排序
all_gt_nuclei.sort(key=lambda x: x['area'])

print(f"\n找到 {len(all_gt_nuclei)} 个有效 GT 核 (>=500px)")
print("最小的 10 个:")
for i in range(10):
    n = all_gt_nuclei[i]
    print(f"  {i+1}. 面积={n['area']} px, 文件={n['file']}")

print("\n最大的 5 个:")
for i in range(1, 6):
    n = all_gt_nuclei[-i]
    print(f"  {i}. 面积={n['area']} px, 文件={n['file']}")

# 可视化最小的几个
smallest = all_gt_nuclei[0]
print(f"\n正在打开 Napari 查看最小核 (Area={smallest['area']} px)...")

img = tifffile.imread(os.path.join(data_dir, smallest['file']))
bf = img[0]
dapi = img[4]
gt = img[9]

viewer = napari.Viewer(title=f"Min GT Nucleus: Area={smallest['area']} px")
viewer.add_image(bf, name='Brightfield')
viewer.add_image(dapi, name='DAPI', colormap='blue', visible=False)
viewer.add_labels(gt, name='GT_Mask')

# 标记
y, x = smallest['centroid']
viewer.add_points([[y, x]], size=20, face_color='red', name='Smallest_Valid_Nucleus')

napari.run()
