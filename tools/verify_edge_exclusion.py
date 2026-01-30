"""
verify_edge_exclusion.py

功能: 验证边缘排除率是否包含极小核
创建日期: 2026-01-26

逻辑:
1. 检测所有核
2. 过滤掉 min_area < 500 的核 (这是我们检测流程的标准的第一步)
3. 检查剩余的"有效核"中，有多少在 GT 内且距离边缘 < 100px
4. 排除率 = (有效且边缘GT核) / (有效且GT核总数)
"""
import numpy as np
import tifffile
import os
import sys
from skimage import measure, morphology, filters
from scipy import ndimage

sys.path.insert(0, 'src')

def detect_raw_nuclei(dapi):
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
    
    print("=" * 60)
    print("边缘排除率验证 (Filter: min_area >= 500)")
    print("=" * 60)
    
    gt_nuclei_total = 0
    gt_nuclei_excluded = 0
    
    nuclei_small_excluded = 0 # 因面积小被排除的 GT 核
    
    for fname in files:
        img = tifffile.imread(os.path.join(data_dir, fname))
        dapi = img[4]
        gt = img[9]
        h, w = dapi.shape
        
        raw_nuclei = detect_raw_nuclei(dapi)
        
        for r in raw_nuclei:
            y, x = int(r.centroid[0]), int(r.centroid[1])
            
            # 检查是否在 GT 内
            in_gt = False
            if 0 <= y < h and 0 <= x < w:
                if gt[y, x] > 0:
                    in_gt = True
            
            if not in_gt:
                continue
                
            # 这是一个 GT 核
            if r.area < 500:
                nuclei_small_excluded += 1
                continue
            
            # 这是一个"有效大小"的 GT 核
            gt_nuclei_total += 1
            
            # 检查边缘距离
            edge_dist = min(y, h-y, x, w-x)
            if edge_dist < 100:
                gt_nuclei_excluded += 1

    print(f"\nGT 内核统计:")
    print(f"  极小核 (<500px): {nuclei_small_excluded} (已排除，不计入分母)")
    print(f"  有效核 (>=500px): {gt_nuclei_total} (作为分母)")
    print(f"  因边缘 < 100px 被排除的有效核: {gt_nuclei_excluded}")
    
    if gt_nuclei_total > 0:
        ratio = 100 * gt_nuclei_excluded / gt_nuclei_total
        print(f"\n最终排除率: {ratio:.1f}%")
        print(f"这意味着: 在 {gt_nuclei_total} 个正常的 GT 细胞中，有 {gt_nuclei_excluded} 个因为靠边被排除了")
    else:
        print("未找到有效 GT 核")

if __name__ == "__main__":
    main()
