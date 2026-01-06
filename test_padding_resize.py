"""
Test Padding + Resize preprocessing method.
Compares original vs processed images in Napari.
"""

import numpy as np
from pathlib import Path
from skimage import transform
import napari


def preprocess_with_padding(image, mask, target_size=1024):
    """
    Padding + Resize preprocessing:
    1. Pad image to square (preserves aspect ratio)
    2. Resize to target size
    
    Returns:
        processed_image, processed_mask, padding_info
    """
    h, w = image.shape[:2]
    
    # Step 1: Pad to square
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    # Pad image
    padded_image = np.pad(
        image,
        ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
        mode='constant',
        constant_values=0
    )
    
    # Pad mask
    padded_mask = np.pad(
        mask,
        ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
        mode='constant',
        constant_values=0
    )
    
    # Step 2: Resize to target size
    resized_image = transform.resize(
        padded_image, 
        (target_size, target_size),
        preserve_range=True,
        anti_aliasing=True
    ).astype(image.dtype)
    
    resized_mask = transform.resize(
        padded_mask,
        (target_size, target_size),
        order=0,  # Nearest neighbor for masks
        preserve_range=True,
        anti_aliasing=False
    ).astype(mask.dtype)
    
    # Store padding info for potential reverse transformation
    padding_info = {
        'original_h': h,
        'original_w': w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'max_dim': max_dim,
        'target_size': target_size
    }
    
    return resized_image, resized_mask, padding_info


def preprocess_direct_resize(image, mask, target_size=1024):
    """
    Direct Resize (current method) for comparison.
    """
    resized_image = transform.resize(
        image,
        (target_size, target_size),
        preserve_range=True,
        anti_aliasing=True
    ).astype(image.dtype)
    
    resized_mask = transform.resize(
        mask,
        (target_size, target_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(mask.dtype)
    
    return resized_image, resized_mask


def main():
    print("=" * 60)
    print("Padding + Resize Preprocessing Test")
    print("=" * 60)
    
    # Load a test sample
    data_dir = Path("d:/AI/paper/CellSam/training_pairs_expanded")
    image_files = sorted(list((data_dir / "images").glob("*.npy")))
    
    if not image_files:
        print("No image files found!")
        return
    
    # Use first sample
    sample_path = image_files[0]
    sample_id = sample_path.stem
    
    print(f"\nLoading sample: {sample_id[:40]}...")
    
    image = np.load(sample_path)
    mask = np.load(data_dir / "masks" / f"{sample_id}.npy")
    
    print(f"Original shape: {image.shape}")
    
    # Method 1: Direct Resize (current)
    print("\nApplying Direct Resize...")
    direct_image, direct_mask = preprocess_direct_resize(image, mask)
    
    # Method 2: Padding + Resize (new)
    print("Applying Padding + Resize...")
    padded_image, padded_mask, info = preprocess_with_padding(image, mask)
    
    print(f"\nPadding info:")
    print(f"  Original: {info['original_h']}×{info['original_w']}")
    print(f"  Padded to: {info['max_dim']}×{info['max_dim']}")
    print(f"  Final: {info['target_size']}×{info['target_size']}")
    
    # Calculate aspect ratio change
    original_ratio = image.shape[1] / image.shape[0]
    direct_ratio = 1.0  # Square after direct resize
    padded_ratio = 1.0  # Square but content ratio preserved
    
    print(f"\nAspect ratio:")
    print(f"  Original: {original_ratio:.4f}")
    print(f"  Direct Resize: 1.0 (distorted)")
    print(f"  Padding + Resize: preserves original ratio inside")
    
    # Open in Napari
    print("\n" + "=" * 60)
    print("Opening Napari viewer...")
    print("=" * 60)
    
    viewer = napari.Viewer(title="Preprocessing Comparison")
    
    # Original
    viewer.add_image(image, name='1. Original Image', colormap='gray')
    viewer.add_labels(mask.astype(np.int32), name='1. Original Mask', opacity=0.5, visible=False)
    
    # Direct Resize
    viewer.add_image(direct_image, name='2. Direct Resize', colormap='gray', visible=False)
    viewer.add_labels(direct_mask.astype(np.int32), name='2. Direct Resize Mask', opacity=0.5, visible=False)
    
    # Padding + Resize
    viewer.add_image(padded_image, name='3. Padding+Resize', colormap='gray', visible=False)
    viewer.add_labels(padded_mask.astype(np.int32), name='3. Padding+Resize Mask', opacity=0.5, visible=False)
    
    print("\n📌 使用说明:")
    print("  - 点击图层名称左侧的眼睛图标切换显示")
    print("  - 对比以下三种处理:")
    print("    1. Original Image (原始 1736×1776)")
    print("    2. Direct Resize (直接缩放 1024×1024)")
    print("    3. Padding+Resize (填充后缩放 1024×1024)")
    print("\n  观察点:")
    print("    - 细胞形态是否变形")
    print("    - 肌节条纹是否清晰")
    print("    - 掩膜边界是否平滑")
    
    napari.run()


if __name__ == "__main__":
    main()
