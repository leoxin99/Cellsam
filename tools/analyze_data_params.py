"""
Data Analysis Script: Actn2 threshold and GT boundary width analysis
For determining data-driven parameters.
"""
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt

# Paths
PROCESSED_DIR = Path("d:/AI/paper/CellSam/data/processed")
SPLITS_DIR = Path("d:/AI/paper/CellSam/data/splits")

def load_split_ids(split="val"):
    """Load sample IDs from split file."""
    with open(SPLITS_DIR / f"{split}_ids.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

def analyze_actn2_threshold():
    """Analyze Actn2 channel intensity distribution to determine optimal threshold."""
    print("=" * 60)
    print("Actn2 阈值分析")
    print("=" * 60)
    
    val_ids = load_split_ids("val")
    
    actn2_percentiles = []
    
    for sample_id in val_ids[:20]:  # Sample 20 images
        image_path = PROCESSED_DIR / "images" / f"{sample_id}.npy"
        if not image_path.exists():
            continue
        
        image = np.load(image_path)
        # image shape: (3, H, W) - [BF, DAPI, Actn2]
        actn2 = image[2] if image.shape[0] == 3 else image[0]
        
        # Normalize to 0-1
        actn2_norm = (actn2 - actn2.min()) / (actn2.max() - actn2.min() + 1e-8)
        
        # Collect percentiles
        for p in [5, 10, 20, 30, 50, 70, 80, 90, 95]:
            actn2_percentiles.append({
                'sample': sample_id[:20],
                'percentile': p,
                'value': np.percentile(actn2_norm, p)
            })
    
    # Summarize
    import pandas as pd
    df = pd.DataFrame(actn2_percentiles)
    summary = df.groupby('percentile')['value'].agg(['mean', 'std', 'min', 'max'])
    
    print("\nActn2 归一化强度百分位统计:")
    print(summary)
    
    # Recommendation
    p20_mean = summary.loc[20, 'mean']
    p30_mean = summary.loc[30, 'mean']
    
    print(f"\n推荐阈值范围: {p20_mean:.3f} ~ {p30_mean:.3f}")
    print(f"建议使用: {(p20_mean + p30_mean) / 2:.3f} (P25 均值)")
    
    return summary

def analyze_boundary_width():
    """Analyze GT mask boundary width."""
    print("\n" + "=" * 60)
    print("GT 边界宽度分析")
    print("=" * 60)
    
    val_ids = load_split_ids("val")
    
    boundary_widths = []
    
    for sample_id in val_ids[:20]:
        mask_path = PROCESSED_DIR / "masks" / f"{sample_id}.npy"
        if not mask_path.exists():
            continue
        
        mask = np.load(mask_path)
        
        # For each cell
        for region in measure.regionprops(mask.astype(np.int32)):
            # Create binary mask for this cell
            cell_mask = (mask == region.label).astype(np.float32)
            
            # Compute boundary via erosion
            for erosion_iter in range(1, 10):
                eroded = ndimage.binary_erosion(cell_mask, iterations=erosion_iter)
                boundary = cell_mask - eroded.astype(np.float32)
                boundary_pixels = boundary.sum()
                
                if boundary_pixels == 0:
                    break
                    
                boundary_widths.append({
                    'cell_id': region.label,
                    'erosion_iter': erosion_iter,
                    'boundary_pixels': boundary_pixels,
                    'cell_area': region.area
                })
    
    import pandas as pd
    df = pd.DataFrame(boundary_widths)
    
    # Analyze how many erosion iterations until boundary disappears
    last_iter = df.groupby('cell_id')['erosion_iter'].max()
    
    print(f"\n单次腐蚀边界宽度统计 (erosion=1):")
    first_erosion = df[df['erosion_iter'] == 1]
    print(f"  边界平均像素: {first_erosion['boundary_pixels'].mean():.1f}")
    print(f"  边界/面积比: {(first_erosion['boundary_pixels'] / first_erosion['cell_area']).mean():.4f}")
    
    print(f"\n细胞消失所需腐蚀次数:")
    print(f"  Mean: {last_iter.mean():.1f}")
    print(f"  Median: {last_iter.median():.1f}")
    print(f"  P25-P75: {last_iter.quantile(0.25):.1f} - {last_iter.quantile(0.75):.1f}")
    
    print(f"\n推荐边界宽度: 2-3 像素 (基于中位腐蚀次数)")
    
    return df

def analyze_cell_areas():
    """Re-verify cell area thresholds."""
    print("\n" + "=" * 60)
    print("细胞面积阈值验证")
    print("=" * 60)
    
    val_ids = load_split_ids("val")
    
    areas = []
    
    for sample_id in val_ids:
        mask_path = PROCESSED_DIR / "masks" / f"{sample_id}.npy"
        if not mask_path.exists():
            continue
        
        mask = np.load(mask_path)
        
        for region in measure.regionprops(mask.astype(np.int32)):
            areas.append(region.area)
    
    areas = np.array(areas)
    
    print(f"\n细胞面积统计 (n={len(areas)}):")
    print(f"  Min: {areas.min():,}")
    print(f"  P1: {np.percentile(areas, 1):,.0f}")
    print(f"  P5: {np.percentile(areas, 5):,.0f}")
    print(f"  P25: {np.percentile(areas, 25):,.0f}")
    print(f"  Median: {np.median(areas):,.0f}")
    print(f"  P75: {np.percentile(areas, 75):,.0f}")
    print(f"  P95: {np.percentile(areas, 95):,.0f}")
    print(f"  P99: {np.percentile(areas, 99):,.0f}")
    print(f"  Max: {areas.max():,}")
    
    print(f"\n当前代码使用阈值:")
    print(f"  min_area = 40,836 (P1)")
    print(f"  max_area = 513,928 (P99+)")
    
    p1 = np.percentile(areas, 1)
    p99 = np.percentile(areas, 99)
    
    if abs(p1 - 40836) / 40836 > 0.1:
        print(f"  ⚠️ 警告: P1={p1:.0f} 与代码值 40836 差异 > 10%")
    else:
        print(f"  ✅ P1 验证通过")
    
    return areas

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CellSAM 数据驱动参数分析")
    print("=" * 60)
    
    # 1. Actn2 threshold
    actn2_summary = analyze_actn2_threshold()
    
    # 2. Boundary width
    boundary_df = analyze_boundary_width()
    
    # 3. Cell area verification
    areas = analyze_cell_areas()
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
