"""
Preprocessing script for Allen hiPSC-CM dataset.
Converts 3D OME-TIFF files to 2D projections for CellSAM training.

Usage:
    python preprocess_allen_data.py --input_dir /path/to/raw_images --output_dir /path/to/processed

Requirements:
    pip install tifffile numpy scikit-image
"""

import os
import argparse
import numpy as np
import tifffile
from pathlib import Path
from skimage import exposure
from typing import Dict, Optional
import json


def sum_projection(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Perform sum projection along specified axis.
    Better than max projection for preserving sarcomere texture intensity.
    
    Args:
        volume: 3D array (Z, H, W) or 4D array (Z, C, H, W)
        axis: Axis to project along (0 = Z-axis)
    
    Returns:
        2D projected image
    """
    projected = np.sum(volume, axis=axis, dtype=np.float64)
    # Normalize to 0-1 range
    projected = (projected - projected.min()) / (projected.max() - projected.min() + 1e-8)
    return projected.astype(np.float32)


def max_projection(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Perform max intensity projection along specified axis.
    Alternative to sum projection.
    """
    return np.max(volume, axis=axis)


def extract_channels_from_ome_tiff(
    filepath: str,
    channel_names: Optional[Dict[str, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract specific channels from an OME-TIFF file.
    
    Args:
        filepath: Path to OME-TIFF file
        channel_names: Dict mapping channel name to index, e.g. {"brightfield": 0, "dapi": 1}
                      If None, will try to auto-detect from metadata
    
    Returns:
        Dict of channel name -> 3D numpy array
    """
    with tifffile.TiffFile(filepath) as tif:
        # Read the full image stack
        data = tif.asarray()
        
        # Try to get metadata
        metadata = {}
        if tif.ome_metadata:
            metadata['ome'] = tif.ome_metadata
        
        print(f"  Loaded shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        
    # Handle different dimension orders
    # Common formats: (Z, C, H, W), (C, Z, H, W), (Z, H, W), (T, Z, C, H, W)
    
    if len(data.shape) == 3:
        # Assume (Z, H, W) - single channel
        return {"channel_0": data}
    
    elif len(data.shape) == 4:
        # Could be (Z, C, H, W) or (C, Z, H, W)
        # Heuristic: channel dimension is usually smaller
        if data.shape[1] < data.shape[0]:
            # Likely (Z, C, H, W)
            channels = {f"channel_{i}": data[:, i, :, :] for i in range(data.shape[1])}
        else:
            # Likely (C, Z, H, W)
            channels = {f"channel_{i}": data[i, :, :, :] for i in range(data.shape[0])}
        return channels
    
    elif len(data.shape) == 5:
        # Likely (T, Z, C, H, W) - take first timepoint
        print("  5D data detected, taking first timepoint")
        data = data[0]
        channels = {f"channel_{i}": data[:, i, :, :] for i in range(data.shape[1])}
        return channels
    
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def normalize_for_display(img: np.ndarray, percentile_low: float = 1, percentile_high: float = 99) -> np.ndarray:
    """
    Normalize image using percentile clipping for better visualization.
    """
    p_low, p_high = np.percentile(img, [percentile_low, percentile_high])
    img_clipped = np.clip(img, p_low, p_high)
    img_normalized = (img_clipped - p_low) / (p_high - p_low + 1e-8)
    return img_normalized.astype(np.float32)


def process_single_file(
    input_path: str,
    output_dir: str,
    projection_method: str = "sum",
    channel_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """
    Process a single OME-TIFF file.
    
    Args:
        input_path: Path to input OME-TIFF
        output_dir: Directory to save outputs
        projection_method: "sum" or "max"
        channel_mapping: Optional dict mapping semantic names to channel indices
                        e.g. {"brightfield": 0, "dapi": 3, "actn2": 1}
    
    Returns:
        Dict of output file paths
    """
    filename = Path(input_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {filename}")
    
    # Extract channels
    channels = extract_channels_from_ome_tiff(input_path, channel_mapping)
    
    # Choose projection function
    project_fn = sum_projection if projection_method == "sum" else max_projection
    
    output_paths = {}
    
    for channel_name, volume in channels.items():
        # Project to 2D
        projected = project_fn(volume)
        
        # Normalize for display
        projected = normalize_for_display(projected)
        
        # Save as NPY (for training) and PNG (for visualization)
        npy_path = os.path.join(output_dir, f"{filename}_{channel_name}.npy")
        png_path = os.path.join(output_dir, f"{filename}_{channel_name}.png")
        
        np.save(npy_path, projected)
        
        # Convert to uint8 for PNG
        img_uint8 = (projected * 255).astype(np.uint8)
        tifffile.imwrite(png_path, img_uint8)
        
        output_paths[channel_name] = npy_path
        print(f"  Saved {channel_name}: {projected.shape}")
    
    # Save metadata
    meta_path = os.path.join(output_dir, f"{filename}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "source_file": input_path,
            "channels": list(channels.keys()),
            "projection_method": projection_method,
            "output_paths": output_paths
        }, f, indent=2)
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Preprocess Allen hiPSC-CM dataset for CellSAM training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing OME-TIFF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed files")
    parser.add_argument("--projection", type=str, default="sum", choices=["sum", "max"],
                       help="Projection method (sum recommended for sarcomeres)")
    parser.add_argument("--brightfield_channel", type=int, default=None, help="Index of brightfield channel")
    parser.add_argument("--dapi_channel", type=int, default=None, help="Index of DAPI channel")
    parser.add_argument("--actn2_channel", type=int, default=None, help="Index of Alpha-actinin-2 channel")
    
    args = parser.parse_args()
    
    # Build channel mapping if provided
    channel_mapping = None
    if any([args.brightfield_channel, args.dapi_channel, args.actn2_channel]):
        channel_mapping = {}
        if args.brightfield_channel is not None:
            channel_mapping["brightfield"] = args.brightfield_channel
        if args.dapi_channel is not None:
            channel_mapping["dapi"] = args.dapi_channel
        if args.actn2_channel is not None:
            channel_mapping["actn2"] = args.actn2_channel
    
    # Find all TIFF files
    input_dir = Path(args.input_dir)
    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in {args.input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for tiff_path in tiff_files:
        try:
            process_single_file(
                str(tiff_path),
                args.output_dir,
                args.projection,
                channel_mapping
            )
        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
            continue
    
    print(f"\nDone! Processed files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
