"""
View Raw 10-Channel TIFF in Napari
Purpose: Let user verify actual channel contents of Allen dataset
"""

import napari
import tifffile
from pathlib import Path

# Sample raw TIFF file
RAW_DATA_DIR = Path("data/raw/allen_segmented_fields_full")

def view_raw_tiff_channels():
    """Open a raw 10-channel TIFF for user to verify channel contents."""
    
    # Find first available TIFF
    tiff_files = list(RAW_DATA_DIR.glob("*.tiff"))[:1]
    if not tiff_files:
        print("No TIFF files found!")
        return
    
    tiff_path = tiff_files[0]
    print(f"Loading: {tiff_path.name}")
    
    img = tifffile.imread(tiff_path)
    print(f"Shape: {img.shape}")  # Expected: (10, 1736, 1776)
    
    viewer = napari.Viewer(title=f"Raw TIFF Channels: {tiff_path.name}")
    
    # Add all 10 channels for user inspection
    channel_names = [
        "Ch0_BF",           # Brightfield (confirmed)
        "Ch1_?",            # extract_expanded_pairs.py says Actn2
        "Ch2_?",  
        "Ch3_?",  
        "Ch4_?",            # extract_expanded_pairs.py says DAPI
        "Ch5_?",            # technical_details.md says Actn2
        "Ch6_?",            # compare_segmentation.py used this
        "Ch7_?",            # compare_segmentation.py used this
        "Ch8_?",  
        "Ch9_Mask",         # GT mask (confirmed)
    ]
    
    for i in range(min(10, img.shape[0])):
        name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
        viewer.add_image(
            img[i], 
            name=name, 
            visible=(i in [0, 1, 4, 5, 9]),  # Show key channels
            colormap='gray' if i != 9 else 'turbo'
        )
    
    print("\n" + "="*60)
    print("Channel Verification Guide")
    print("="*60)
    print("Please check each channel:")
    print("  - Ch0: Should be Brightfield (grayscale cell image)")
    print("  - Ch1: extract script says Actn2 - is it?")
    print("  - Ch4: extract script says DAPI - is it blue nuclei?")
    print("  - Ch5: technical_details says Actn2 - check!")
    print("  - Ch9: Should be GT mask (colored regions)")
    print("="*60)
    print("\nCurrent extract_expanded_pairs.py settings:")
    print("  CH_BRIGHTFIELD = 0")
    print("  CH_ACTN2 = 1")
    print("  CH_DAPI = 4")
    print("  CH_MASK = 9")
    print("="*60)
    
    napari.run()


if __name__ == "__main__":
    view_raw_tiff_channels()
