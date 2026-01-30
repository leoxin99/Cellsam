"""
visualize_detection_comparison.py

功能: Napari 可视化对比不同检测方法
所属实验: E18 - SarcGraph 检测对比
创建日期: 2026-01-23
最后修改: 2026-01-23
版本: v2

对比方法:
1. DAPI only (固定扩展)
2. Adaptive (Z-线引导自适应框)
3. Z-线点云 (cyan)

依赖函数:
- detection.dapi.detect_with_adaptive_box
- detection.dapi.detect_nuclei, merge_close_nuclei, create_bounding_boxes

更新日志:
- 2026-01-23 v2: 添加 Adaptive 方法和 Z-线点可视化
- 2026-01-23 v1: 初始版本 (DAPI vs SarcGraph)

Usage:
    conda activate cellsam
    python tools/visualize_detection_comparison.py
"""
import napari
import tifffile
import numpy as np
import os
import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'src/comparison/sarcgraph_pipeline')

from detection.dapi import (
    detect_nuclei, 
    merge_close_nuclei, 
    create_bounding_boxes,
    detect_cardiomyocytes,
    detect_with_adaptive_box  # NEW adaptive method
)
from prompt_generator import SarcGraphPromptGenerator

# Load 3 samples
data_dir = 'data/raw/allen_segmented_fields_full'
files = sorted(os.listdir(data_dir))[:3]

# SarcGraph parameters
sarcgraph_gen = SarcGraphPromptGenerator(
    pixel_size_um=0.108,
    sarcomere_length_um=2.0,
    eps_factor=2.5,
    min_samples=10,
    padding_pixels=50,
    padding_ratio=0.2
)
sarcgraph_gen.zline_detector.threshold = 0.03

viewer = napari.Viewer()

for idx, fname in enumerate(files):
    print(f'Loading {fname}...')
    img = tifffile.imread(os.path.join(data_dir, fname))
    
    bf = img[0]      # Brightfield
    actn2 = img[1]   # Actn2
    dapi = img[4]    # DAPI
    gt = img[9]      # GT mask
    
    # Method 1: DAPI only
    nuclei_regions = detect_nuclei(dapi, min_area=3000)
    cell_groups = merge_close_nuclei(nuclei_regions)
    dapi_boxes = create_bounding_boxes(cell_groups, dapi.shape)
    
    # Method 2: DAPI + Actn2 Adaptive Box (NEW)
    adaptive_boxes, _, debug_info = detect_with_adaptive_box(
        dapi, actn2,
        min_nucleus_area=3000,
        search_radius=400,
        min_zlines=15,
        zline_threshold=0.03
    )
    
    # Count adaptive vs fallback
    num_adaptive = sum(1 for d in debug_info if d['adaptive'])
    num_fallback = len(debug_info) - num_adaptive
    
    # Collect all Z-lines for visualization
    all_zlines = []
    for d in debug_info:
        if len(d['zlines']) > 0:
            all_zlines.append(d['zlines'])
    
    print(f'  Sample {idx+1}:')
    print(f'    DAPI only: {len(dapi_boxes)} boxes')
    print(f'    Adaptive: {len(adaptive_boxes)} boxes ({num_adaptive} adaptive, {num_fallback} fallback)')
    
    # Add to viewer
    viewer.add_image(bf, name=f'S{idx+1}_BF', visible=(idx==0))
    viewer.add_image(actn2, name=f'S{idx+1}_Actn2', visible=(idx==0), colormap='hot')
    viewer.add_image(dapi, name=f'S{idx+1}_DAPI_img', visible=False, colormap='blue')
    viewer.add_labels(gt, name=f'S{idx+1}_GT', visible=(idx==0))
    
    # DAPI boxes (Red)
    if dapi_boxes:
        dapi_rects = [[[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]] for b in dapi_boxes]
        viewer.add_shapes(dapi_rects, shape_type='polygon', edge_color='red',
                         face_color='transparent', edge_width=2, name=f'S{idx+1}_DAPI_only')
    
    # Adaptive boxes (Yellow) - NEW
    if adaptive_boxes:
        adaptive_rects = [[[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]] for b in adaptive_boxes]
        viewer.add_shapes(adaptive_rects, shape_type='polygon', edge_color='yellow',
                         face_color='transparent', edge_width=3, name=f'S{idx+1}_Adaptive')
    
    # Z-lines as points (Cyan)
    if all_zlines:
        zlines_concat = np.concatenate(all_zlines, axis=0)
        viewer.add_points(zlines_concat, size=3, face_color='cyan', 
                         name=f'S{idx+1}_ZLines', visible=(idx==0))

print('\n' + '='*50)
print('Detection Methods:')
print('  🔴 Red = DAPI only (fixed expansion)')
print('  🟡 Yellow = Adaptive (Z-line guided box size)')
print('  🔵 Cyan points = Z-lines detected')
print('='*50)
napari.run()

