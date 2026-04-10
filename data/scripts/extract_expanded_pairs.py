"""
Extract training data from Allen Cell annotation dataset.
Outputs multi-channel images (BF + DAPI + Actn2) and masks, resized to 1024x1024.

Output structure:
  processed/
    images/*.npy   - 3-channel image: [BF, DAPI, Actn2], shape (3, 1024, 1024)
    masks/*.npy    - Instance mask, shape (1024, 1024)

Usage:
    python extract_expanded_pairs.py                    # Process all files
    python extract_expanded_pairs.py --limit 20         # Process first 20 files
"""

import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from skimage import transform

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "allen_segmented_fields_full"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Channel indices in TIFF
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9

# Target size for SAM model
TARGET_SIZE = (1024, 1024)


def normalize_channel(image):
    """Normalize image to 0-255 uint8 using P2-P98 percentile scaling."""
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    
    if p98 > p2:
        clipped = np.clip(image, p2, p98)
        normalized = ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    return normalized


def resize_image(image, target_size):
    """Resize image to target size with anti-aliasing."""
    return transform.resize(image, target_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)


def resize_mask(mask, target_size):
    """Resize mask using nearest neighbor (no interpolation) to preserve labels."""
    return transform.resize(mask, target_size, order=0, preserve_range=True).astype(np.int32)


def extract_data(limit=None):
    """
    Extract multi-channel images and masks from TIFF files.
    
    Output format:
    - images/*.npy: shape (3, 1024, 1024) - [BF, DAPI, Actn2]
    - masks/*.npy: shape (1024, 1024) - instance segmentation mask
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Clean output directory
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "masks").mkdir(exist_ok=True)
    
    # Find all TIFF files
    tiff_files = sorted(list(input_path.glob("*.tiff")) + list(input_path.glob("*.tif")))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_path}")
        return
    
    total_available = len(tiff_files)
    if limit is not None and limit > 0:
        tiff_files = tiff_files[:limit]
    
    print("=" * 60)
    print("CellSAM Multi-Channel Data Extraction")
    print("=" * 60)
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target size: {TARGET_SIZE}")
    print(f"  Channels: BF (Ch{CH_BRIGHTFIELD}) + DAPI (Ch{CH_DAPI}) + Actn2 (Ch{CH_ACTN2})")
    print(f"  Total files available: {total_available}")
    print(f"  Files to process: {len(tiff_files)}")
    print("=" * 60)
    
    extracted = 0
    total_cells = 0
    stats = []
    
    for tiff_file in tqdm(tiff_files, desc="Extracting"):
        try:
            data = tifffile.imread(tiff_file)
            
            if len(data.shape) != 3 or data.shape[0] < 10:
                print(f"  Skipping {tiff_file.name}: unexpected shape {data.shape}")
                continue
            
            # Extract all required channels
            bf = data[CH_BRIGHTFIELD]
            dapi = data[CH_DAPI]
            actn2 = data[CH_ACTN2]
            mask = data[CH_MASK]
            
            # Normalize each channel
            bf_norm = normalize_channel(bf)
            dapi_norm = normalize_channel(dapi)
            actn2_norm = normalize_channel(actn2)
            
            # Resize to 1024x1024
            bf_resized = resize_image(bf_norm, TARGET_SIZE)
            dapi_resized = resize_image(dapi_norm, TARGET_SIZE)
            actn2_resized = resize_image(actn2_norm, TARGET_SIZE)
            mask_resized = resize_mask(mask, TARGET_SIZE)
            
            # Stack channels: (3, 1024, 1024)
            multi_channel = np.stack([bf_resized, dapi_resized, actn2_resized], axis=0)
            
            # Count cells
            unique_labels = np.unique(mask_resized)
            n_cells = len(unique_labels[unique_labels > 0])
            total_cells += n_cells
            
            # Generate clean filename
            base_name = tiff_file.stem.replace("_annotations_corrected", "").replace("_rescaled.ome", "")
            
            # Save as .npy
            np.save(output_path / "images" / f"{base_name}.npy", multi_channel)
            np.save(output_path / "masks" / f"{base_name}.npy", mask_resized)
            
            stats.append({
                "filename": base_name,
                "original_shape": bf.shape,
                "n_cells": n_cells
            })
            
            extracted += 1
            
        except Exception as e:
            print(f"Error processing {tiff_file.name}: {e}")
    
    # Save stats
    if stats:
        import json
        stats_file = output_path / "extraction_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_files": len(tiff_files),
                "extracted": extracted,
                "total_cells": total_cells,
                "target_size": TARGET_SIZE,
                "channels": ["BF", "DAPI", "Actn2"],
                "files": stats
            }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"  Files processed: {len(tiff_files)}")
    print(f"  Successfully extracted: {extracted}")
    print(f"  Total cells: {total_cells}")
    print(f"  Output format:")
    print(f"    images/*.npy: (3, 1024, 1024) - [BF, DAPI, Actn2]")
    print(f"    masks/*.npy:  (1024, 1024)")
    print("=" * 60)
    
    return extracted, total_cells


def main():
    parser = argparse.ArgumentParser(description="Extract multi-channel training data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files to process (default: all)")
    
    args = parser.parse_args()
    extract_data(limit=args.limit)


if __name__ == "__main__":
    main()
