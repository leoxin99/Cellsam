"""
View Min/Max Size GT Cells in Napari
Purpose: User visual review of size thresholds used in SizeLoss
"""

import sys
import numpy as np
import napari
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.augmented_dataset import load_split_ids
from skimage.measure import regionprops

# Size thresholds from SizeLoss (1024px scaled)
MIN_SIZE = 13884  # P1 scaled
MAX_SIZE = 174735  # P99 scaled

def find_cells_by_size():
    """Find cells closest to min and max size thresholds."""
    
    data_dir = Path("data/processed")
    all_ids = load_split_ids('train') + load_split_ids('val')
    
    all_cells = []
    
    print(f"Scanning {len(all_ids)} samples for cell sizes...")
    
    for sample_id in all_ids:
        mask_path = data_dir / "masks" / f"{sample_id}.npy"
        img_path = data_dir / "images" / f"{sample_id}.npy"
        
        if not mask_path.exists() or not img_path.exists():
            continue
        
        mask = np.load(mask_path)
        regions = regionprops(mask)
        
        for region in regions:
            all_cells.append({
                'sample_id': sample_id,
                'label': region.label,
                'area': region.area,
                'bbox': region.bbox
            })
    
    print(f"Found {len(all_cells)} cells total")
    
    # Find cells closest to min and max thresholds
    areas = np.array([c['area'] for c in all_cells])
    
    # Cells near MIN_SIZE (P1 threshold)
    min_diff = np.abs(areas - MIN_SIZE)
    min_indices = np.argsort(min_diff)[:5]  # 5 closest to min
    
    # Cells near MAX_SIZE (P99 threshold)
    max_diff = np.abs(areas - MAX_SIZE)
    max_indices = np.argsort(max_diff)[:5]  # 5 closest to max
    
    # Also find actual smallest and largest
    smallest_indices = np.argsort(areas)[:3]
    largest_indices = np.argsort(areas)[-3:]
    
    return all_cells, {
        'near_min_threshold': [all_cells[i] for i in min_indices],
        'near_max_threshold': [all_cells[i] for i in max_indices],
        'smallest_actual': [all_cells[i] for i in smallest_indices],
        'largest_actual': [all_cells[i] for i in largest_indices]
    }


def visualize_cells(cells_info, category='near_min_threshold'):
    """Show cells in napari."""
    
    data_dir = Path("data/processed")
    cells = cells_info[category]
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {category}")
    print(f"{'='*60}")
    
    viewer = napari.Viewer(title=f"CellSAM - {category}")
    
    for i, cell in enumerate(cells):
        sample_id = cell['sample_id']
        label = cell['label']
        area = cell['area']
        
        img = np.load(data_dir / "images" / f"{sample_id}.npy")
        mask = np.load(data_dir / "masks" / f"{sample_id}.npy")
        
        # Highlight just this cell
        cell_mask = (mask == label).astype(np.uint8)
        
        # BF channel for visualization
        bf = img[0] if img.ndim == 3 else img[:, :, 0]
        
        print(f"  Cell {i+1}: area={area:,} px, sample={sample_id}, label={label}")
        
        layer_name = f"Cell_{i+1}_area_{area}"
        viewer.add_image(bf, name=f"{layer_name}_BF", visible=(i==0))
        viewer.add_labels(cell_mask * (i+1), name=f"{layer_name}_mask", 
                         visible=(i==0), opacity=0.5)
    
    print(f"\nThreshold values:")
    print(f"  MIN_SIZE (P1): {MIN_SIZE:,} px")
    print(f"  MAX_SIZE (P99): {MAX_SIZE:,} px")
    print(f"\nToggle layers to compare different cells")
    
    napari.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='near_min_threshold',
                       choices=['near_min_threshold', 'near_max_threshold', 
                               'smallest_actual', 'largest_actual'])
    args = parser.parse_args()
    
    all_cells, cells_info = find_cells_by_size()
    
    # Print summary
    areas = [c['area'] for c in all_cells]
    print(f"\n{'='*60}")
    print(f"Dataset Cell Size Statistics")
    print(f"{'='*60}")
    print(f"  Total cells: {len(all_cells)}")
    print(f"  Min area: {min(areas):,} px")
    print(f"  Max area: {max(areas):,} px")
    print(f"  Mean: {np.mean(areas):,.0f} px")
    print(f"  Median: {np.median(areas):,.0f} px")
    print(f"  P1: {np.percentile(areas, 1):,.0f} px")
    print(f"  P99: {np.percentile(areas, 99):,.0f} px")
    print(f"\nSize thresholds in SizeLoss:")
    print(f"  MIN_SIZE: {MIN_SIZE:,} px")
    print(f"  MAX_SIZE: {MAX_SIZE:,} px")
    
    visualize_cells(cells_info, args.category)
