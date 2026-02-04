"""
Visualize P1 and P99 area cells in napari to verify thresholds.
Based on E17 analysis: P1=40836, P99=513928 from full dataset.
"""
import napari
import numpy as np
from pathlib import Path
from skimage import measure

# Paths
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

def load_split_ids(split="train"):
    with open(SPLITS_DIR / f"{split}_ids.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

def find_cells_near_percentile(target_area, tolerance=0.1):
    """Find cells with area close to target."""
    all_ids = load_split_ids("train") + load_split_ids("val") + load_split_ids("test")
    
    results = []
    
    for sample_id in all_ids:
        mask_path = PROCESSED_DIR / "masks" / f"{sample_id}.npy"
        if not mask_path.exists():
            continue
        
        mask = np.load(mask_path)
        
        for region in measure.regionprops(mask.astype(np.int32)):
            area_diff = abs(region.area - target_area) / target_area
            if area_diff < tolerance:
                results.append({
                    'sample_id': sample_id,
                    'label': region.label,
                    'area': region.area,
                    'centroid': region.centroid,
                    'diff': area_diff
                })
    
    # Sort by closest match
    results.sort(key=lambda x: x['diff'])
    return results[:10]  # Top 10 matches

def visualize_cells(target_area, title="Cells"):
    """Open napari with cells near target area."""
    cells = find_cells_near_percentile(target_area)
    
    if not cells:
        print(f"No cells found near {target_area}")
        return
    
    print(f"\n找到 {len(cells)} 个接近 {target_area} 面积的细胞:")
    for c in cells:
        print(f"  {c['sample_id']}: area={c['area']:,} (diff={c['diff']:.2%})")
    
    # Load first match
    cell = cells[0]
    sample_id = cell['sample_id']
    
    image = np.load(PROCESSED_DIR / "images" / f"{sample_id}.npy")
    mask = np.load(PROCESSED_DIR / "masks" / f"{sample_id}.npy")
    
    # Highlight target cell
    highlight = (mask == cell['label']).astype(np.uint8)
    
    viewer = napari.Viewer(title=f"{title}: Area={cell['area']:,}")
    
    # Add channels
    viewer.add_image(image[0], name="BF", colormap="gray")
    if image.shape[0] > 1:
        viewer.add_image(image[1], name="DAPI", colormap="blue", blending="additive", visible=False)
    if image.shape[0] > 2:
        viewer.add_image(image[2], name="Actn2", colormap="green", blending="additive", visible=False)
    
    # Add masks
    viewer.add_labels(mask, name="All Cells")
    viewer.add_image(highlight, name=f"Target Cell (area={cell['area']:,})", colormap="red", blending="additive", opacity=0.5)
    
    # Zoom to cell
    y, x = cell['centroid']
    viewer.camera.center = (y, x)
    viewer.camera.zoom = 2
    
    print(f"\n已打开 napari，显示 {title}")
    
    return viewer

if __name__ == "__main__":
    import sys
    
    # E17 thresholds
    P1_AREA = 40836
    P99_AREA = 513928
    
    print("E17 细胞面积统计 (全数据集 478张, 5173细胞):")
    print(f"  P1  = {P1_AREA:,} px (~202×202)")
    print(f"  P99 = {P99_AREA:,} px (~717×717)")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "p1":
            visualize_cells(P1_AREA, "P1 Cell (Smallest 1%)")
        elif sys.argv[1] == "p99":
            visualize_cells(P99_AREA, "P99 Cell (Largest 1%)")
        else:
            visualize_cells(int(sys.argv[1]), f"Custom area {sys.argv[1]}")
        napari.run()
    else:
        print("\n使用方法:")
        print("  python tools/view_percentile_cells.py p1   # 查看P1面积细胞")
        print("  python tools/view_percentile_cells.py p99  # 查看P99面积细胞")
        print("  python tools/view_percentile_cells.py 100000  # 自定义面积")
        
        # 自动显示两个
        print("\n自动打开 P1 和 P99 细胞...")
        v1 = visualize_cells(P1_AREA, "P1 Cell (Smallest 1%)")
        v2 = visualize_cells(P99_AREA, "P99 Cell (Largest 1%)")
        napari.run()
