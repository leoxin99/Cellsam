"""
visualize_excluded_nuclei.py

功能: 可视化被 100px 边缘规则排除的有效 GT 核 (>=5000px)
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
current_min_area = 5000
edge_thresh = 100

excluded_cells = []
sample_5000_cells = []

print("正在寻找被排除的核...")
for fname in files:
    img = tifffile.imread(os.path.join(data_dir, fname))
    dapi = img[4]
    gt = img[9]
    image_shape = dapi.shape
    h, w = image_shape
    
    nuclei = detect_nuclei(dapi)
    
    for r in nuclei:
        # 必须是 "有效大小" 的 GT 核
        if r.area < current_min_area: continue
        
        y, x = int(r.centroid[0]), int(r.centroid[1])
        if 0 <= y < h and 0 <= x < w:
            if gt[y, x] > 0:
                # 记录一个典型的 5000px 核作为参考
                if abs(r.area - 5000) < 500 and len(sample_5000_cells) < 1:
                    sample_5000_cells.append({
                        'file': fname, 'centroid': r.centroid, 'area': r.area, 'type': 'reference'
                    })
                
                # 检查是否被边缘过滤排除
                dist = min(y, h-y, x, w-x)
                if dist < edge_thresh:
                    excluded_cells.append({
                        'file': fname,
                        'centroid': r.centroid,
                        'area': r.area,
                        'dist': dist,
                        'type': 'excluded'
                    })

print(f"找到 {len(excluded_cells)} 个被排除的有效核")
if not excluded_cells:
    print("没有核被排除")
    sys.exit()

# 选择展示
# 优先展示面积接近 5000px 的被排除核
target_area = 5000
excluded_cells.sort(key=lambda x: abs(x['area'] - target_area))

target = excluded_cells[0]
ref_cell = sample_5000_cells[0] if sample_5000_cells else None

print(f"展示被排除的核心 (Area={target['area']} px, ~5000px):")
print(f"  文件: {target['file']}")
print(f"  距离边缘: {target['dist']:.1f} px")
print(f"  位置: {target['centroid']}")

if ref_cell:
    print(f"\n参考核 (Area={ref_cell['area']} px):")
    print(f"  文件: {ref_cell['file']}")

print("\n打开 Napari...")

# 加载被排除核的图像
img1 = tifffile.imread(os.path.join(data_dir, target['file']))
bf1 = img1[0]
dapi1 = img1[4]
gt1 = img1[9]

viewer = napari.Viewer(title=f"Excluded Nucleus: Dist={target['dist']:.1f}px (Edge < {edge_thresh}px)")
viewer.add_image(bf1, name='Brightfield')
viewer.add_image(dapi1, name='DAPI', colormap='blue', visible=False)
viewer.add_labels(gt1, name='GT_Mask')

# 标记被排除的核
y, x = target['centroid']
viewer.add_points([[y, x]], size=30, face_color='red', name='Excluded_Nucleus', symbol='x')

# 如果参考核在另一张图，需要另外加载或者提示
if ref_cell and ref_cell['file'] != target['file']:
    print(f"\n注意: 参考核在另一张图 ({ref_cell['file']})，本视图只显示被排除的核。")
    # 为了对比，我们可以尝试在同一个viewer里加一个layer，但这会很乱。
    # 我们只显示被排除的核，用户已经看过 5000px 的核了。
elif ref_cell:
     # 如果在同一张图
     ry, rx = ref_cell['centroid']
     viewer.add_points([[ry, rx]], size=30, face_color='green', name='Reference_5000px', symbol='o')

# 画出 100px 边缘线
h, w = dapi1.shape
border_shapes = [
    [[edge_thresh, edge_thresh], [edge_thresh, w-edge_thresh]],
    [[edge_thresh, w-edge_thresh], [h-edge_thresh, w-edge_thresh]],
    [[h-edge_thresh, w-edge_thresh], [h-edge_thresh, edge_thresh]],
    [[h-edge_thresh, edge_thresh], [edge_thresh, edge_thresh]]
]
viewer.add_shapes(border_shapes, shape_type='line', edge_color='yellow', edge_width=2, name='100px_Margin')

napari.run()
