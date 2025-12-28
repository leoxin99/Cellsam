"""
Generate bounding box prompts from DAPI channel images.
Detects nuclei and creates bounding boxes for CellSAM training.

Usage:
    python generate_prompts_from_dapi.py --input_dir /path/to/processed --output_dir /path/to/prompts

Requirements:
    pip install numpy scikit-image scipy
"""

import os
import argparse
import numpy as np
from pathlib import Path
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage
import json
from typing import List, Tuple, Dict


def detect_nuclei(
    dapi_image: np.ndarray,
    threshold_method: str = "otsu",
    min_size: int = 100,
    max_size: int = 50000,
    separate_touching: bool = True
) -> np.ndarray:
    """
    Detect nuclei from DAPI channel using thresholding and morphological operations.
    
    Args:
        dapi_image: 2D DAPI image (normalized 0-1 or 0-255)
        threshold_method: "otsu" or "adaptive"
        min_size: Minimum nucleus area in pixels
        max_size: Maximum nucleus area in pixels
        separate_touching: Use watershed to separate touching nuclei
    
    Returns:
        Labeled image where each nucleus has unique integer ID
    """
    # Normalize to 0-1 if needed
    if dapi_image.max() > 1:
        dapi_image = dapi_image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to reduce noise
    smoothed = filters.gaussian(dapi_image, sigma=2)
    
    # Threshold
    if threshold_method == "otsu":
        thresh = filters.threshold_otsu(smoothed)
    else:
        thresh = filters.threshold_local(smoothed, block_size=51)
    
    binary = smoothed > thresh
    
    # Morphological cleanup
    binary = morphology.binary_opening(binary, morphology.disk(2))
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    # Remove small objects
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    
    if separate_touching:
        # Watershed to separate touching nuclei
        distance = ndimage.distance_transform_edt(binary)
        local_max = morphology.h_maxima(distance, h=5)
        markers = measure.label(local_max)
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labels = measure.label(binary)
    
    # Filter by size
    props = measure.regionprops(labels)
    for prop in props:
        if prop.area < min_size or prop.area > max_size:
            labels[labels == prop.label] = 0
    
    # Relabel consecutively
    labels = measure.label(labels > 0)
    
    return labels


def extract_bounding_boxes(
    labels: np.ndarray,
    padding: int = 5,
    format: str = "xyxy"
) -> List[Dict]:
    """
    Extract bounding boxes from labeled nuclei image.
    
    Args:
        labels: Labeled image from detect_nuclei
        padding: Pixels to add around each box
        format: "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,width,height)
    
    Returns:
        List of dicts with box coordinates and metadata
    """
    props = measure.regionprops(labels)
    boxes = []
    
    h, w = labels.shape
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        
        # Add padding
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(h, maxr + padding)
        maxc = min(w, maxc + padding)
        
        # Centroid
        cy, cx = prop.centroid
        
        if format == "xyxy":
            box = {
                "box": [float(minc), float(minr), float(maxc), float(maxr)],
                "centroid": [float(cx), float(cy)],
                "area": int(prop.area),
                "label_id": int(prop.label)
            }
        else:  # xywh
            box = {
                "box": [float(minc), float(minr), float(maxc - minc), float(maxr - minr)],
                "centroid": [float(cx), float(cy)],
                "area": int(prop.area),
                "label_id": int(prop.label)
            }
        
        boxes.append(box)
    
    return boxes


def process_dapi_image(
    dapi_path: str,
    output_dir: str,
    threshold_method: str = "otsu",
    min_nucleus_size: int = 100,
    max_nucleus_size: int = 50000,
    padding: int = 10
) -> Tuple[str, int]:
    """
    Process a single DAPI image to generate bounding boxes.
    
    Returns:
        Tuple of (output_json_path, num_boxes_found)
    """
    filename = Path(dapi_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    if dapi_path.endswith('.npy'):
        dapi_image = np.load(dapi_path)
    else:
        from skimage import io
        dapi_image = io.imread(dapi_path)
    
    print(f"Processing: {filename}")
    print(f"  Image shape: {dapi_image.shape}, dtype: {dapi_image.dtype}")
    
    # Detect nuclei
    labels = detect_nuclei(
        dapi_image,
        threshold_method=threshold_method,
        min_size=min_nucleus_size,
        max_size=max_nucleus_size
    )
    
    num_nuclei = len(np.unique(labels)) - 1  # -1 for background
    print(f"  Detected {num_nuclei} nuclei")
    
    # Extract bounding boxes
    boxes = extract_bounding_boxes(labels, padding=padding, format="xyxy")
    
    # Save results
    output_data = {
        "source_file": dapi_path,
        "image_shape": list(dapi_image.shape),
        "num_nuclei": num_nuclei,
        "boxes": boxes,
        "format": "xyxy",
        "padding": padding
    }
    
    json_path = os.path.join(output_dir, f"{filename}_prompts.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Also save the nucleus labels for visualization
    labels_path = os.path.join(output_dir, f"{filename}_nuclei_labels.npy")
    np.save(labels_path, labels.astype(np.int32))
    
    print(f"  Saved: {json_path}")
    
    return json_path, num_nuclei


def main():
    parser = argparse.ArgumentParser(description="Generate bounding box prompts from DAPI images")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing DAPI images (NPY or image files)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for prompt JSON files")
    parser.add_argument("--pattern", type=str, default="*dapi*.npy",
                       help="Glob pattern to match DAPI files (default: *dapi*.npy)")
    parser.add_argument("--threshold", type=str, default="otsu", choices=["otsu", "adaptive"],
                       help="Thresholding method")
    parser.add_argument("--min_size", type=int, default=100,
                       help="Minimum nucleus area in pixels")
    parser.add_argument("--max_size", type=int, default=50000,
                       help="Maximum nucleus area in pixels")
    parser.add_argument("--padding", type=int, default=10,
                       help="Padding around bounding boxes")
    
    args = parser.parse_args()
    
    # Find DAPI files
    input_dir = Path(args.input_dir)
    dapi_files = list(input_dir.glob(args.pattern))
    
    if not dapi_files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        return
    
    print(f"Found {len(dapi_files)} DAPI files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_nuclei = 0
    for dapi_path in dapi_files:
        try:
            _, num_nuclei = process_dapi_image(
                str(dapi_path),
                args.output_dir,
                threshold_method=args.threshold,
                min_nucleus_size=args.min_size,
                max_nucleus_size=args.max_size,
                padding=args.padding
            )
            total_nuclei += num_nuclei
        except Exception as e:
            print(f"Error processing {dapi_path}: {e}")
            continue
    
    print(f"\nDone! Generated prompts for {total_nuclei} total nuclei")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
