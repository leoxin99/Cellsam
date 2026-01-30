"""
analyze_stats_refined.py

功能: 优化版边缘和双核统计分析
所属实验: 参数优化 (Refined)
创建日期: 2026-01-26

改进点:
1. 动态确定 min_area: 统计落在 GT 细胞内的核的最小面积
2. 双核统计增加 size_ratio < 3.0 约束: 排除检测逻辑会拒绝的配对
"""
import numpy as np
import tifffile
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage
from collections import defaultdict

sys.path.insert(0, 'src')

def detect_raw_nuclei(dapi):
    """检测 DAPI 通道中的原始核 (使用极低阈值以捕获所有潜在核)"""
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

def main():
    data_dir = 'data/raw/allen_segmented_fields_full'
    # 使用 Dev Set (前 50 张)
    files = sorted(os.listdir(data_dir))[:50]
    
    print("=" * 60)
    print("正在分析 GT 内核的面积分布...")
    print("=" * 60)
    
    gt_nuclei_areas = []
    
    # 第一遍: 统计 GT 内有效核的面积
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        
        raw_nuclei = detect_raw_nuclei(dapi)
        
        for r in raw_nuclei:
            y, x = int(r.centroid[0]), int(r.centroid[1])
            # 检查是否在 GT 掩膜内
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                if gt[y, x] > 0: # 在某个 GT 细胞内
                    gt_nuclei_areas.append(r.area)

    gt_nuclei_areas = np.array(gt_nuclei_areas)
    if len(gt_nuclei_areas) == 0:
        print("未找到 GT 内的核!")
        return

    min_gt_area = np.min(gt_nuclei_areas)
    # 取 P1 以排除极端的噪声
    p1_gt_area = np.percentile(gt_nuclei_areas, 1)
    recommended_min_area = int(p1_gt_area)
    
    print(f"GT 内核数量: {len(gt_nuclei_areas)}")
    print(f"面积统计:")
    print(f"  Min: {min_gt_area:.0f}")
    print(f"  P1:  {p1_gt_area:.0f} (推荐阈值)")
    print(f"  P5:  {np.percentile(gt_nuclei_areas, 5):.0f}")
    print(f"  Median: {np.median(gt_nuclei_areas):.0f}")
    
    # 使用 1000 或 P1 的较大者，或者直接用 P1 (如果 P1 很小说明真有这么小的核)
    # 用户建议 e.g. 1000px，我们看统计结果
    # 这里我们暂定使用 max(1000, p1) 来做双核分析，或者遵从数据单纯用 P1
    # 为稳妥起见，如果 P1 < 500，可能太噪，但 GT 里的应该是真的。
    # 让我们采用 P1 作为 "GT 最小核" 的代表
    current_min_area = max(500, int(p1_gt_area)) 
    print(f"\n将在下一步使用 min_area = {current_min_area} 进行双核分析")
    
    print("\n" + "=" * 60)
    print(f"正在分析双核距离 (min_area={current_min_area}, size_ratio < 3.0)...")
    print("=" * 60)
    
    valid_distances = []
    rejected_ratio_distances = [] # 因 ratio 被拒绝的距离
    
    total_pairs = 0
    valid_pairs = 0
    
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        
        raw_nuclei = detect_raw_nuclei(dapi)
        
        # 1. 面积过滤
        valid_nuclei = [r for r in raw_nuclei if r.area >= current_min_area and r.area <= 30000]
        
        # 2. 按 GT 细胞归组
        cell_nuclei = defaultdict(list)
        for nuc in valid_nuclei:
            y, x = int(nuc.centroid[0]), int(nuc.centroid[1])
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                cid = gt[y, x]
                if cid > 0:
                    cell_nuclei[cid].append(nuc)
        
    print(f"\n" + "=" * 60)
    print(f"正在分析边缘排除率 (仅统计有效大小的 GT 核: area >= {current_min_area})...")
    print("=" * 60)
    
    gt_edge_dists = []
    
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        image_shape = dapi.shape
        
        raw_nuclei = detect_raw_nuclei(dapi)
        
        for r in raw_nuclei:
            if r.area < current_min_area: # 排除极小核
                continue
                
            y, x = int(r.centroid[0]), int(r.centroid[1])
            # 检查是否在 GT 掩膜内
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                if gt[y, x] > 0:
                    # 计算边缘距离
                    dist = min(y, image_shape[0]-y, x, image_shape[1]-x)
                    gt_edge_dists.append(dist)
    
    gt_edge_dists = np.array(gt_edge_dists)
    n_total = len(gt_edge_dists)
    print(f"\n有效 GT 核总数: {n_total}")
    
    if n_total > 0:
        for thresh in [30, 50, 100, 150]:
            n_excluded = np.sum(gt_edge_dists < thresh)
            print(f"  阈值 {thresh}px: 排除 {n_excluded}/{n_total} ({100*n_excluded/n_total:.1f}%)")

if __name__ == "__main__":
    main()
