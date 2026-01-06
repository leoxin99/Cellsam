"""
Extract training pairs from expanded Allen Cell annotation dataset.
Processes 50+ files and extracts ~500+ cells for training.
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from skimage import io


# Configuration
INPUT_DIR = "d:/AI/paper/CellSam/allen_data_expanded"
OUTPUT_DIR = "d:/AI/paper/CellSam/training_pairs_expanded"

IMAGE_CHANNEL = 0      # Brightfield
MASK_CHANNEL = 9       # Instance segmentation


def normalize_image(image):
    """Normalize image to 0-255 uint8 range."""
    img_min = image.min()
    img_max = image.max()
    
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    return normalized


def extract_pairs():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "masks").mkdir(exist_ok=True)
    
    tiff_files = list(input_path.glob("*.ome.tiff")) + list(input_path.glob("*.tif"))
    
    if not tiff_files:
        print(f"No TIFF files found in {INPUT_DIR}")
        return
    
    print("=" * 60)
    print(f"Processing {len(tiff_files)} annotation files")
    print("=" * 60)
    
    extracted = 0
    total_cells = 0
    
    for tiff_file in tqdm(tiff_files, desc="Extracting"):
        try:
            data = tifffile.imread(tiff_file)
            
            if len(data.shape) != 3:
                continue
            
            n_channels = data.shape[0]
            if n_channels <= max(IMAGE_CHANNEL, MASK_CHANNEL):
                continue
            
            image = data[IMAGE_CHANNEL]
            mask = data[MASK_CHANNEL]
            
            # Count cells
            n_cells = len(np.unique(mask)) - 1
            total_cells += n_cells
            
            # Normalize image
            image_norm = normalize_image(image)
            
            # Generate filename
            base_name = tiff_file.stem.replace("_annotations_corrected_rescaled.ome", "")
            
            # Save as .npy
            np.save(output_path / "images" / f"{base_name}.npy", image_norm)
            np.save(output_path / "masks" / f"{base_name}.npy", mask)
            
            extracted += 1
            
        except Exception as e:
            print(f"Error processing {tiff_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"  Files processed: {len(tiff_files)}")
    print(f"  Successfully extracted: {extracted}")
    print(f"  Total cells: {total_cells}")
    print(f"  Average cells per image: {total_cells / max(extracted, 1):.1f}")
    print(f"  Output directory: {output_path}")
    print("=" * 60)
    
    return extracted, total_cells


if __name__ == "__main__":
    extract_pairs()
