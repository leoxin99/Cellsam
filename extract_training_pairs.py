"""
Extract training pairs (Image, GT Mask) from Allen Cell annotation TIFF files.
Each annotation file contains multiple channels, we extract:
- Channel 0: Brightfield image (or other channels as specified)
- Channel 9: Instance segmentation mask (Ground Truth)

Output: Paired .npy files for training
"""
import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = "d:/AI/paper/CellSam/allen_data"
OUTPUT_DIR = "d:/AI/paper/CellSam/training_pairs"

# Channel mapping (based on Allen Cell data structure)
IMAGE_CHANNEL = 0      # Brightfield
MASK_CHANNEL = 9       # Instance segmentation

def extract_pairs():
    """Extract image-mask pairs from annotation TIFF files."""
    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "masks").mkdir(exist_ok=True)
    
    # Find all TIFF files
    tiff_files = list(input_path.glob("*.ome.tiff")) + list(input_path.glob("*.tif"))
    
    if not tiff_files:
        print(f"No TIFF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(tiff_files)} annotation files")
    print(f"Extracting Channel {IMAGE_CHANNEL} (image) and Channel {MASK_CHANNEL} (mask)")
    print()
    
    extracted = 0
    
    for tiff_file in tqdm(tiff_files, desc="Processing"):
        try:
            # Read the multi-channel TIFF
            data = tifffile.imread(tiff_file)
            
            # Validate shape
            if len(data.shape) != 3:
                print(f"  [SKIP] {tiff_file.name}: unexpected shape {data.shape}")
                continue
            
            n_channels = data.shape[0]
            
            if n_channels <= max(IMAGE_CHANNEL, MASK_CHANNEL):
                print(f"  [SKIP] {tiff_file.name}: only {n_channels} channels")
                continue
            
            # Extract image and mask
            image = data[IMAGE_CHANNEL]  # 2D image
            mask = data[MASK_CHANNEL]    # 2D mask
            
            # Normalize image to 0-255 uint8
            image_norm = normalize_image(image)
            
            # Generate output filename
            base_name = tiff_file.stem.replace("_annotations_corrected_rescaled.ome", "")
            
            # Save as .npy for fast loading
            np.save(output_path / "images" / f"{base_name}.npy", image_norm)
            np.save(output_path / "masks" / f"{base_name}.npy", mask)
            
            # Also save as PNG for visualization
            from skimage import io
            io.imsave(output_path / "images" / f"{base_name}.png", image_norm)
            io.imsave(output_path / "masks" / f"{base_name}_mask.png", 
                      (mask * (255 // max(1, mask.max()))).astype(np.uint8))
            
            extracted += 1
            
        except Exception as e:
            print(f"  [ERROR] {tiff_file.name}: {e}")
    
    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Total files processed: {len(tiff_files)}")
    print(f"  Successfully extracted: {extracted}")
    print(f"  Output directory: {output_path}")
    print()
    print("Files saved:")
    print(f"  - {output_path}/images/*.npy (normalized images)")
    print(f"  - {output_path}/masks/*.npy (instance masks)")
    print(f"  - {output_path}/images/*.png (visualization)")
    print(f"  - {output_path}/masks/*_mask.png (visualization)")
    print("=" * 60)

def normalize_image(image):
    """Normalize image to 0-255 uint8 range."""
    img_min = image.min()
    img_max = image.max()
    
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    return normalized

def analyze_data():
    """Analyze the extracted training data."""
    output_path = Path(OUTPUT_DIR)
    
    images = list((output_path / "images").glob("*.npy"))
    masks = list((output_path / "masks").glob("*.npy"))
    
    print("\n" + "=" * 60)
    print("Training Data Analysis")
    print("=" * 60)
    
    print(f"\nTotal training pairs: {len(images)}")
    
    for img_path, mask_path in zip(sorted(images), sorted(masks)):
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        n_cells = len(np.unique(mask)) - 1  # Exclude background (0)
        
        print(f"\n{img_path.stem}:")
        print(f"  Image: {img.shape}, dtype={img.dtype}")
        print(f"  Mask:  {mask.shape}, dtype={mask.dtype}")
        print(f"  Cells: {n_cells}")

if __name__ == "__main__":
    print("=" * 60)
    print("Allen Cell Training Data Extractor")
    print("=" * 60)
    print()
    
    extract_pairs()
    analyze_data()
