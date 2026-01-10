"""
Napari Viewer with CellSAM Segmentation Overlay.
Opens preprocessed images and runs CellSAM inference to display segmentation results.

Usage:
    python view_with_cellsam.py --image path/to/image.npy
    python view_with_cellsam.py --dir path/to/processed_folder
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import napari
from napari.utils import CyclicLabelColormap

# Add CellSAM to path
sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))


def get_colormap():
    """Create a nice colormap for segmentation labels."""
    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]
    return CyclicLabelColormap(colors=colors, background_value=0)


def load_image(path: str) -> np.ndarray:
    """Load image from NPY or standard image format."""
    if path.endswith('.npy'):
        return np.load(path)
    else:
        from skimage import io
        return io.imread(path)


def run_cellsam_inference(image: np.ndarray, bbox_threshold: float = 0.3):
    """
    Run CellSAM inference on a single image.
    
    Returns:
        mask: Segmentation mask
        num_cells: Number of detected cells
    """
    from cellSAM import get_model
    from cellSAM.model import segment_cellular_image
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CellSAM model...")
    model = get_model()
    
    print("Running inference...")
    mask, embedding, boxes = segment_cellular_image(
        image, 
        model, 
        device=device,
        bbox_threshold=bbox_threshold
    )
    
    num_cells = len(np.unique(mask)) - 1  # Exclude background
    print(f"Detected {num_cells} cells")
    
    return mask, num_cells


def view_single_image(image_path: str, run_inference: bool = True, bbox_threshold: float = 0.3):
    """Open a single image in Napari with optional CellSAM overlay."""
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Normalize for display
    if image.dtype != np.uint8:
        if image.max() > 1:
            image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image_display = image
    else:
        image_display = image
    
    # Create viewer
    viewer = napari.Viewer(title=f"CellSAM Viewer - {Path(image_path).name}")
    
    # Add image layer
    viewer.add_image(image_display, name="Image", colormap="gray")
    
    if run_inference:
        mask, num_cells = run_cellsam_inference(image, bbox_threshold)
        
        # Add segmentation overlay
        viewer.add_labels(
            mask, 
            name=f"Segmentation ({num_cells} cells)",
            colormap=get_colormap(),
            opacity=0.5
        )
        
        # Add contour view
        viewer.add_labels(
            mask,
            name="Contours",
            colormap=get_colormap(),
            opacity=0.8,
            contour=2
        )
    
    print("\n=== Napari Controls ===")
    print("- Scroll wheel: Zoom in/out")
    print("- Click & drag: Pan view")
    print("- Toggle layers: Click eye icon in layer list")
    print("- Adjust opacity: Slider in layer controls")
    print("========================\n")
    
    napari.run()


def view_directory(dir_path: str, pattern: str = "*_channel_0.npy", run_inference: bool = True):
    """Open all images in a directory."""
    dir_path = Path(dir_path)
    image_files = sorted(dir_path.glob(pattern))
    
    if not image_files:
        print(f"No files matching '{pattern}' found in {dir_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create viewer
    viewer = napari.Viewer(title=f"CellSAM Viewer - {dir_path.name}")
    
    for i, img_path in enumerate(image_files):
        print(f"\nProcessing [{i+1}/{len(image_files)}]: {img_path.name}")
        
        image = load_image(str(img_path))
        
        # Normalize
        if image.max() > 1:
            image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image_display = image
        
        # Add image
        viewer.add_image(image_display, name=f"Img: {img_path.stem[:30]}", colormap="gray")
        
        if run_inference:
            mask, num_cells = run_cellsam_inference(image)
            viewer.add_labels(
                mask,
                name=f"Seg: {img_path.stem[:20]} ({num_cells})",
                colormap=get_colormap(),
                opacity=0.5
            )
    
    print("\nAll images loaded!")
    napari.run()


def main():
    parser = argparse.ArgumentParser(description="View images with CellSAM segmentation in Napari")
    parser.add_argument("--image", type=str, help="Path to single image file (NPY or PNG/TIFF)")
    parser.add_argument("--dir", type=str, help="Path to directory with processed images")
    parser.add_argument("--pattern", type=str, default="*_channel_0.npy", help="Glob pattern for directory mode")
    parser.add_argument("--no-inference", action="store_true", help="Skip CellSAM inference, just view images")
    parser.add_argument("--threshold", type=float, default=0.3, help="BBox detection threshold (lower = more cells)")
    
    args = parser.parse_args()
    
    run_inference = not args.no_inference
    
    if args.image:
        view_single_image(args.image, run_inference, args.threshold)
    elif args.dir:
        view_directory(args.dir, args.pattern, run_inference)
    else:
        # Default: view the processed brightfield images
        default_dir = "d:/AI/paper/CellSam/allen_brightfield_processed"
        if os.path.exists(default_dir):
            print(f"No arguments provided. Using default directory: {default_dir}")
            view_directory(default_dir, "*_brf_channel_0.npy", run_inference)
        else:
            print("Usage:")
            print("  python view_with_cellsam.py --image path/to/image.npy")
            print("  python view_with_cellsam.py --dir path/to/folder")
            print("\nOptions:")
            print("  --no-inference  : Just view images without running CellSAM")
            print("  --threshold 0.2 : Lower threshold = detect more cells")


if __name__ == "__main__":
    main()
