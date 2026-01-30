"""
Quick visualization to confirm channel order in processed .npy files.
Run: python tools/visualize_npy_channels.py
"""
import numpy as np
from pathlib import Path

# Check if napari available
try:
    import napari
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    print("Napari not available, will save to image instead")

def main():
    # Find a sample .npy file
    data_dir = Path("d:/AI/paper/CellSam/data/processed/images")
    
    if not data_dir.exists():
        # Try alternate location
        data_dir = Path("d:/AI/paper/CellSam/training_pairs_expanded/images")
    
    npy_files = sorted(data_dir.glob("*.npy"))[:5]
    
    if not npy_files:
        print(f"No .npy files found in {data_dir}")
        return
    
    print(f"Found {len(npy_files)} samples, loading first one...")
    
    sample_path = npy_files[0]
    img = np.load(sample_path)
    print(f"Sample: {sample_path.name}")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min/Max: {img.min():.1f} / {img.max():.1f}")
    
    # Handle different shapes
    if img.ndim == 3:
        if img.shape[0] == 3:  # (3, H, W)
            ch0 = img[0]
            ch1 = img[1]
            ch2 = img[2]
        elif img.shape[-1] == 3:  # (H, W, 3)
            ch0 = img[..., 0]
            ch1 = img[..., 1]
            ch2 = img[..., 2]
        else:
            print(f"Unexpected shape: {img.shape}")
            return
    else:
        print(f"Expected 3D array, got {img.ndim}D")
        return
    
    print(f"\nChannel statistics:")
    print(f"  Ch0: min={ch0.min():.1f}, max={ch0.max():.1f}, mean={ch0.mean():.1f}")
    print(f"  Ch1: min={ch1.min():.1f}, max={ch1.max():.1f}, mean={ch1.mean():.1f}")
    print(f"  Ch2: min={ch2.min():.1f}, max={ch2.max():.1f}, mean={ch2.mean():.1f}")
    
    if HAS_NAPARI:
        viewer = napari.Viewer()
        viewer.add_image(ch0, name="Ch0 (BF?)", colormap="gray")
        viewer.add_image(ch1, name="Ch1 (DAPI? or Actn2?)", colormap="blue", blending="additive", visible=False)
        viewer.add_image(ch2, name="Ch2 (Actn2? or DAPI?)", colormap="red", blending="additive", visible=False)
        
        print("\n✅ Napari viewer opened!")
        print("Toggle layers to identify:")
        print("  - BF: grayscale cell outlines")
        print("  - DAPI: bright nuclei")
        print("  - Actn2: striped sarcomere pattern")
        
        napari.run()
    else:
        # Save as image
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ch0, cmap='gray')
        axes[0].set_title('Ch0')
        axes[1].imshow(ch1, cmap='Blues')
        axes[1].set_title('Ch1')
        axes[2].imshow(ch2, cmap='Reds')
        axes[2].set_title('Ch2')
        plt.savefig('channel_check.png', dpi=150)
        print("Saved to channel_check.png")

if __name__ == "__main__":
    main()
