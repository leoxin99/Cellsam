"""
analyze_stats_final.py

功能: 最终版边缘和双核统计分析
所属实验: 参数优化 (Final)
创建日期: 2026-01-26

逻辑:
1. min_area = 3000 (用户指定的有效核阈值)
2. size_ratio < 3.0 (排除碎片配对)
3. 统计:
   - 有效 GT 核的边缘排除率 (30px vs 50px vs 100px)
   - 双核间距 (P95)
"""
import numpy as np
import tifffile
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage
from collections import defaultdict

sys.path.insert(0, 'src')

def detect_nuclei(dapi, min_area):
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
    files = sorted(os.listdir(data_dir))[:50]
    
    current_min_area = 3000
    
    print("=" * 60)
    print(f"最终参数统计 (min_area={current_min_area}, size_ratio < 3.0)")
    print("=" * 60)
    
    # 1. 边缘排除率
    print("\n[1] 边缘排除率 (仅统计有效 GT 核):")
    gt_edge_dists = []
    
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        image_shape = dapi.shape
        
        nuclei = detect_nuclei(dapi, current_min_area)
        
        for r in nuclei:
            if r.area < current_min_area: continue
            
            y, x = int(r.centroid[0]), int(r.centroid[1])
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                if gt[y, x] > 0:
                    dist = min(y, image_shape[0]-y, x, image_shape[1]-x)
                    gt_edge_dists.append(dist)
    
    gt_edge_dists = np.array(gt_edge_dists)
    n_total = len(gt_edge_dists)
    print(f"  有效 GT 核总数: {n_total}")
    
    if n_total > 0:
        for thresh in [30, 50, 100, 150]:
            n_excluded = np.sum(gt_edge_dists < thresh)
            print(f"  阈值 {thresh}px: 排除 {n_excluded}/{n_total} ({100*n_excluded/n_total:.1f}%)")
    
    # 2. 双核统计
    print("\n[2] 双核距离统计 (有效核配对):")
    valid_distances = []
    
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        
        nuclei = detect_nuclei(dapi, current_min_area)
        valid_nuclei = [r for r in nuclei if r.area >= current_min_area]
        
        cell_nuclei = defaultdict(list)
        for nuc in valid_nuclei:
            y, x = int(nuc.centroid[0]), int(nuc.centroid[1])
            if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]:
                cid = gt[y, x]
                if cid > 0:
                    cell_nuclei[cid].append(nuc)
        
        for cid, nucs in cell_nuclei.items():
            if len(nucs) >= 2:
                for i in range(len(nucs)):
                    for j in range(i+1, len(nucs)):
                        n1 = nucs[i]
                        n2 = nucs[j]
                        ratio = max(n1.area, n2.area) / (min(n1.area, n2.area) + 1e-6)
                        
                        if ratio < 3.0:
                            dist = np.sqrt((n1.centroid[0]-n2.centroid[0])**2 + (n1.centroid[1]-n2.centroid[1])**2)
                            valid_distances.append(dist)
    
    valid_distances = np.array(valid_distances)
    if len(valid_distances) > 0:
        print(f"  有效配对数: {len(valid_distances)}")
        print(f"  Min: {valid_distances.min():.1f} px")
        print(f"  P25: {np.percentile(valid_distances, 25):.1f} px")
        print(f"  Median: {np.median(valid_distances):.1f} px")
        print(f"  Mean: {valid_distances.mean():.1f} px")
        print(f"  P75: {np.percentile(valid_distances, 75):.1f} px")
        print(f"  P90: {np.percentile(valid_distances, 90):.1f} px")
        print(f"  P95: {np.percentile(valid_distances, 95):.1f} px")
        print(f"  Max: {valid_distances.max():.1f} px")
    else:
        print("  未找到有效配对")

if __name__ == "__main__":
    main()
