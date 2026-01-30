"""
analyze_edge_and_binuclear_stats.py

功能: 统计数据集中的边缘距离和双核距离分布
所属实验: E18 扩展 - 参数优化
创建日期: 2026-01-25
版本: v1

分析内容:
1. 核位置距离边缘的阈值 → 用于确定边缘过滤参数
2. 双核细胞中两核的距离 → 用于确定双核合并阈值

方法:
- 从 DAPI (Ch4) 检测所有核
- 从 GT (Ch9) 获取细胞标注
- 分析哪些核在 GT 中有对应细胞
"""
import numpy as np
import tifffile
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict

sys.path.insert(0, 'src')

def detect_all_nuclei(dapi, min_area=500, max_area=30000):
    """检测 DAPI 通道中的所有核"""
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
            nuclei.append({
                'centroid': r.centroid,  # (y, x)
                'area': r.area,
                'bbox': r.bbox  # (y1, x1, y2, x2)
            })
    return nuclei


def get_edge_distance(centroid, image_shape):
    """计算核心到最近图像边缘的距离"""
    y, x = centroid
    h, w = image_shape
    
    dist_top = y
    dist_bottom = h - y
    dist_left = x
    dist_right = w - x
    
    return min(dist_top, dist_bottom, dist_left, dist_right)


def match_nucleus_to_gt(nucleus, gt_mask):
    """检查核是否在 GT 标注的细胞内"""
    y, x = int(nucleus['centroid'][0]), int(nucleus['centroid'][1])
    
    if 0 <= y < gt_mask.shape[0] and 0 <= x < gt_mask.shape[1]:
        cell_id = gt_mask[y, x]
        return cell_id > 0, cell_id
    return False, 0


def analyze_binuclear_cells(nuclei, gt_mask):
    """
    分析双核细胞:
    - 找出在同一个 GT 细胞内的多个核
    - 计算它们之间的距离
    """
    # 按 GT cell ID 分组核
    cell_nuclei = defaultdict(list)
    
    for nuc in nuclei:
        in_gt, cell_id = match_nucleus_to_gt(nuc, gt_mask)
        if in_gt and cell_id > 0:
            cell_nuclei[cell_id].append(nuc)
    
    results = {
        'single_nucleus': 0,
        'dual_nucleus': 0,
        'multi_nucleus': 0,
        'inter_nucleus_distances': []
    }
    
    for cell_id, nucs in cell_nuclei.items():
        n = len(nucs)
        if n == 1:
            results['single_nucleus'] += 1
        elif n == 2:
            results['dual_nucleus'] += 1
            # 计算两核距离
            d = np.sqrt(
                (nucs[0]['centroid'][0] - nucs[1]['centroid'][0])**2 +
                (nucs[0]['centroid'][1] - nucs[1]['centroid'][1])**2
            )
            results['inter_nucleus_distances'].append(d)
        else:
            results['multi_nucleus'] += 1
            # 计算所有配对距离
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(
                        (nucs[i]['centroid'][0] - nucs[j]['centroid'][0])**2 +
                        (nucs[i]['centroid'][1] - nucs[j]['centroid'][1])**2
                    )
                    results['inter_nucleus_distances'].append(d)
    
    return results


def main():
    data_dir = 'data/raw/allen_segmented_fields_full'
    files = sorted(os.listdir(data_dir))[:50]  # 分析 50 张
    
    print("=" * 70)
    print("边缘距离和双核距离统计分析")
    print("=" * 70)
    
    # 收集统计数据
    edge_in_gt = []      # 在 GT 中的核的边缘距离
    edge_not_in_gt = []  # 不在 GT 中的核的边缘距离
    all_binuc_dists = [] # 所有双核间距离
    
    total_single = 0
    total_dual = 0
    total_multi = 0
    
    for i, fname in enumerate(files):
        if i % 10 == 0:
            print(f"\n处理中: {i+1}/{len(files)}...")
        
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        image_shape = dapi.shape
        
        # 检测核
        nuclei = detect_all_nuclei(dapi)
        
        # 分析边缘距离
        for nuc in nuclei:
            edge_dist = get_edge_distance(nuc['centroid'], image_shape)
            in_gt, _ = match_nucleus_to_gt(nuc, gt)
            
            if in_gt:
                edge_in_gt.append(edge_dist)
            else:
                edge_not_in_gt.append(edge_dist)
        
        # 分析双核
        binuc = analyze_binuclear_cells(nuclei, gt)
        total_single += binuc['single_nucleus']
        total_dual += binuc['dual_nucleus']
        total_multi += binuc['multi_nucleus']
        all_binuc_dists.extend(binuc['inter_nucleus_distances'])
    
    # 输出统计结果
    print("\n" + "=" * 70)
    print("1. 边缘距离分析 (核心距图像边缘)")
    print("=" * 70)
    
    edge_in_gt = np.array(edge_in_gt)
    edge_not_in_gt = np.array(edge_not_in_gt)
    
    print(f"\n在 GT 中标注的核 (n={len(edge_in_gt)}):")
    print(f"  Min: {edge_in_gt.min():.0f} px")
    print(f"  P5:  {np.percentile(edge_in_gt, 5):.0f} px")
    print(f"  P10: {np.percentile(edge_in_gt, 10):.0f} px")
    print(f"  Median: {np.median(edge_in_gt):.0f} px")
    
    print(f"\n不在 GT 中的核 (n={len(edge_not_in_gt)}):")
    if len(edge_not_in_gt) > 0:
        print(f"  Min: {edge_not_in_gt.min():.0f} px")
        print(f"  P50: {np.median(edge_not_in_gt):.0f} px")
        print(f"  P95: {np.percentile(edge_not_in_gt, 95):.0f} px")
        print(f"  Max: {edge_not_in_gt.max():.0f} px")
    
    # 找到排除阈值
    print("\n推荐边缘排除阈值:")
    for thresh in [50, 100, 150, 200, 250, 300]:
        excluded_gt = np.sum(edge_in_gt < thresh)
        excluded_non_gt = np.sum(edge_not_in_gt < thresh) if len(edge_not_in_gt) > 0 else 0
        print(f"  阈值={thresh}px: 排除 GT核 {excluded_gt}/{len(edge_in_gt)} ({100*excluded_gt/len(edge_in_gt):.1f}%), "
              f"排除非GT核 {excluded_non_gt}/{len(edge_not_in_gt)} ({100*excluded_non_gt/len(edge_not_in_gt) if len(edge_not_in_gt) > 0 else 0:.1f}%)")
    
    print("\n" + "=" * 70)
    print("2. 双核细胞分析")
    print("=" * 70)
    
    print(f"\n细胞核数量分布:")
    print(f"  单核细胞: {total_single}")
    print(f"  双核细胞: {total_dual}")
    print(f"  多核细胞 (>2): {total_multi}")
    
    if all_binuc_dists:
        dists = np.array(all_binuc_dists)
        print(f"\n双核/多核细胞中核间距离 (n={len(dists)}):")
        print(f"  Min: {dists.min():.0f} px")
        print(f"  P25: {np.percentile(dists, 25):.0f} px")
        print(f"  Median: {np.median(dists):.0f} px")
        print(f"  Mean: {dists.mean():.0f} px")
        print(f"  P75: {np.percentile(dists, 75):.0f} px")
        print(f"  P95: {np.percentile(dists, 95):.0f} px")
        print(f"  Max: {dists.max():.0f} px")
        
        print("\n推荐双核合并阈值 (应覆盖 P95):")
        print(f"  建议: {np.percentile(dists, 95):.0f} px (覆盖 95% 双核)")


if __name__ == "__main__":
    main()
