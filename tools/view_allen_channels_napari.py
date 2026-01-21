"""
View Allen Cell dataset channels with napari.
Opens a TIFF file and displays all 10 channels for visual inspection.
"""

import napari
import tifffile
from pathlib import Path
import numpy as np

# --- Channel definitions from allen_channel_defs.json ---
CHANNEL_NAMES = {
    0: "bf (Brightfield)",
    1: "488 (Green fluorescence)",
    2: "561 (Orange/Red fluorescence)", 
    3: "638 (Far-red fluorescence)",
    4: "nuc (Nuclear)",
    5: "seg488 (GT: 488 segmentation)",
    6: "seg561 (GT: 561 segmentation)",
    7: "seg638 (GT: 638 segmentation)",
    8: "backmask (Background mask)",
    9: "cell (GT: Cell segmentation)",
}

# --- Color mappings for visualization ---
CHANNEL_COLORS = {
    0: "gray",      # Brightfield
    1: "green",     # 488
    2: "yellow",    # 561
    3: "red",       # 638
    4: "cyan",      # Nuclear
    5: "green",     # seg488
    6: "yellow",    # seg561
    7: "red",       # seg638
    8: "magenta",   # backmask
    9: "blue",      # cell
}


def view_allen_tiff(tiff_path: str):
    """
    Open an Allen Cell TIFF file in napari with all channels labeled.
    
    Args:
        tiff_path: Path to the TIFF file
    """
    tiff_path = Path(tiff_path)
    print(f"Loading: {tiff_path.name}")
    
    # Load the TIFF file
    data = tifffile.imread(tiff_path)
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")
    
    # Verify we have the expected number of channels
    if len(data.shape) == 3 and data.shape[0] == 10:
        print("✓ Found 10 channels as expected")
    else:
        print(f"⚠ Unexpected shape: {data.shape}")
    
    # Create napari viewer
    viewer = napari.Viewer(title=f"Allen Cell Dataset - {tiff_path.name}")
    
    # Add each channel as a separate layer
    for ch_idx in range(min(data.shape[0], 10)):
        channel_data = data[ch_idx]
        channel_name = CHANNEL_NAMES.get(ch_idx, f"Channel {ch_idx}")
        color = CHANNEL_COLORS.get(ch_idx, "gray")
        
        # Check if this is a segmentation mask (discrete values)
        is_segmentation = ch_idx in [5, 6, 7, 8, 9]
        
        if is_segmentation:
            # Add as labels layer for segmentation masks
            unique_vals = np.unique(channel_data)
            print(f"  [{ch_idx}] {channel_name}: {len(unique_vals)} unique values")
            
            # For mask/label visualization
            viewer.add_labels(
                channel_data.astype(np.int32),
                name=channel_name,
                visible=(ch_idx == 9),  # Only show cell seg by default
            )
        else:
            # Add as image layer for intensity images
            print(f"  [{ch_idx}] {channel_name}: range [{channel_data.min()}, {channel_data.max()}]")
            
            viewer.add_image(
                channel_data,
                name=channel_name,
                colormap=color,
                visible=(ch_idx == 0),  # Only show brightfield by default
                blending="additive" if ch_idx > 0 else "translucent",
            )
    
    print("\n" + "="*60)
    print("Napari viewer opened!")
    print("Tips:")
    print("  - Toggle visibility with the eye icon ✓")
    print("  - Use mouse scroll to zoom")
    print("  - Hold Shift and drag to pan")
    print("  - Click layer name to select and adjust settings")
    print("="*60)
    
    napari.run()


if __name__ == "__main__":
    import sys
    
    # Default data directory
    DATA_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
    
    if len(sys.argv) > 1:
        # Use provided file path
        tiff_file = Path(sys.argv[1])
    else:
        # Pick the first TIFF file in the directory
        tiff_files = list(DATA_DIR.glob("*.tiff"))
        if not tiff_files:
            print(f"No TIFF files found in {DATA_DIR}")
            sys.exit(1)
        tiff_file = tiff_files[0]
        print(f"Using first available file: {tiff_file.name}")
    
    if not tiff_file.exists():
        print(f"File not found: {tiff_file}")
        sys.exit(1)
    
    view_allen_tiff(str(tiff_file))
