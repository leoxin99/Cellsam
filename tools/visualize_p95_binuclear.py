"""
visualize_p95_binuclear.py

功能: 找到并可视化 P95 双核距离 (~373px) 的具体案例
所属实验: 参数验证
创建日期: 2026-01-26
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
    """检测所有核"""
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


def analyze_binuclear_cells(nuclei, gt_mask):
    """找出双核细胞并计算核间距"""
    cell_nuclei = defaultdict(list)
    
    for nuc in nuclei:
        y, x = int(nuc.centroid[0]), int(nuc.centroid[1])
        if 0 <= y < gt_mask.shape[0] and 0 <= x < gt_mask.shape[1]:
            cell_id = gt_mask[y, x]
            if cell_id > 0:
                cell_nuclei[cell_id].append(nuc)
    
    binuc_cases = []
    for cell_id, nucs in cell_nuclei.items():
        if len(nucs) >= 2:
            for i in range(len(nucs)):
                for j in range(i+1, len(nucs)):
                    d = np.sqrt(
                        (nucs[i].centroid[0] - nucs[j].centroid[0])**2 +
                        (nucs[i].centroid[1] - nucs[j].centroid[1])**2
                    )
                    binuc_cases.append({
                        'cell_id': cell_id,
                        'distance': d,
                        'nuc1_centroid': nucs[i].centroid,
                        'nuc2_centroid': nucs[j].centroid
                    })
    
    return binuc_cases


# 分析数据找到 P95 案例
data_dir = 'data/raw/allen_segmented_fields_full'
files = sorted(os.listdir(data_dir))[:50]

all_cases = []
for fname in files:
    img = tifffile.imread(os.path.join(data_dir, fname))
    dapi = img[4]
    gt = img[9]
    
    nuclei = detect_nuclei(dapi)
    cases = analyze_binuclear_cells(nuclei, gt)
    
    for c in cases:
        c['file'] = fname
        all_cases.append(c)

# 找到接近 P95 (~373px) 的案例
all_distances = [c['distance'] for c in all_cases]
p95 = np.percentile(all_distances, 95)
print(f"所有双核距离 P95: {p95:.1f} px")

# 找最接近 373px 的案例
target_dist = 373
closest_case = min(all_cases, key=lambda x: abs(x['distance'] - target_dist))
print(f"\n找到最接近 373px 的案例:")
print(f"  文件: {closest_case['file']}")
print(f"  GT Cell ID: {closest_case['cell_id']}")
print(f"  核间距: {closest_case['distance']:.1f} px")
print(f"  核1 中心: ({closest_case['nuc1_centroid'][1]:.0f}, {closest_case['nuc1_centroid'][0]:.0f})")
print(f"  核2 中心: ({closest_case['nuc2_centroid'][1]:.0f}, {closest_case['nuc2_centroid'][0]:.0f})")

# 加载并可视化
print(f"\n正在打开 Napari...")
img = tifffile.imread(os.path.join(data_dir, closest_case['file']))
bf = img[0]
dapi = img[4]
gt = img[9]

viewer = napari.Viewer(title=f"P95 Binuclear Case: dist={closest_case['distance']:.0f}px")

viewer.add_image(bf, name='Brightfield')
viewer.add_image(dapi, name='DAPI', colormap='blue', visible=False)
viewer.add_labels(gt, name='GT_Mask')

# 标记目标 GT 细胞
target_mask = (gt == closest_case['cell_id']).astype(int)
viewer.add_labels(target_mask * 2, name=f'Target_Cell_{closest_case["cell_id"]}', opacity=0.5)

# 标记两个核中心
nuc_points = np.array([
    [closest_case['nuc1_centroid'][0], closest_case['nuc1_centroid'][1]],
    [closest_case['nuc2_centroid'][0], closest_case['nuc2_centroid'][1]]
])
viewer.add_points(nuc_points, size=20, face_color='yellow', name='Nucleus_Centers')

# 画连线
line = np.array([
    [[closest_case['nuc1_centroid'][0], closest_case['nuc1_centroid'][1]],
     [closest_case['nuc2_centroid'][0], closest_case['nuc2_centroid'][1]]]
])
viewer.add_shapes(line, shape_type='line', edge_color='yellow', edge_width=3, name=f'Distance_{closest_case["distance"]:.0f}px')

print("\n图例:")
print("  🟡 黄色点 = 两个核中心")
print("  🟡 黄色线 = 核间距 (~373px)")
print("  🔵 紫色 = 目标 GT 细胞 (双核)")

napari.run()
