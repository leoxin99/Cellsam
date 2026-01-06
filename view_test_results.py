"""
Napari visualization for comparing original images, GT masks, and model predictions.
Opens an interactive viewer with multiple layers for comparison.
"""

import numpy as np
from pathlib import Path
from skimage import io, transform

try:
    import napari
except ImportError:
    print("Installing napari...")
    import subprocess
    subprocess.run(["pip", "install", "napari[all]", "-q"])
    import napari


def load_test_data():
    """Load test images, GT masks, and predictions."""
    
    images_dir = Path("d:/AI/paper/CellSam/training_pairs_expanded/images")
    masks_dir = Path("d:/AI/paper/CellSam/training_pairs_expanded/masks")
    preds_dir = Path("d:/AI/paper/CellSam/test_results")
    
    # Get the 5 test samples
    pred_files = sorted(list(preds_dir.glob("*_pred.png")))
    
    data = []
    
    for pred_file in pred_files:
        # Extract sample ID (match the naming pattern)
        pred_name = pred_file.stem.replace("_pred", "")
        
        # Find matching original files
        matching_images = list(images_dir.glob(f"{pred_name}*.npy"))
        
        if not matching_images:
            # Try partial match
            for img_file in images_dir.glob("*.npy"):
                if pred_name in img_file.stem:
                    matching_images = [img_file]
                    break
        
        if matching_images:
            image_path = matching_images[0]
            sample_id = image_path.stem
            mask_path = masks_dir / f"{sample_id}.npy"
            
            if mask_path.exists():
                # Load data
                image = np.load(image_path)
                mask = np.load(mask_path)
                pred = io.imread(pred_file)
                
                # Resize all to same size (1024x1024)
                if image.shape != (1024, 1024):
                    image = transform.resize(image, (1024, 1024), preserve_range=True)
                if mask.shape != (1024, 1024):
                    mask = transform.resize(mask, (1024, 1024), order=0, preserve_range=True)
                if pred.shape != (1024, 1024):
                    pred = transform.resize(pred, (1024, 1024), order=0, preserve_range=True)
                
                data.append({
                    'id': sample_id[:30],
                    'image': image.astype(np.float32),
                    'mask': mask.astype(np.int32),
                    'pred': (pred > 128).astype(np.int32)
                })
                print(f"Loaded: {sample_id[:30]}")
    
    return data


def calculate_metrics(gt, pred):
    """Calculate Dice and IoU."""
    gt_binary = (gt > 0).astype(float)
    pred_binary = (pred > 0).astype(float)
    
    intersection = (gt_binary * pred_binary).sum()
    dice = 2 * intersection / (gt_binary.sum() + pred_binary.sum() + 1e-8)
    iou = intersection / (gt_binary.sum() + pred_binary.sum() - intersection + 1e-8)
    
    return dice, iou


def main():
    print("=" * 60)
    print("CellSAM Test Results Viewer (Napari)")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    test_data = load_test_data()
    
    if not test_data:
        print("No test data found!")
        return
    
    print(f"\nLoaded {len(test_data)} samples")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    for sample in test_data:
        dice, iou = calculate_metrics(sample['mask'], sample['pred'])
        print(f"  {sample['id']}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # Start Napari viewer
    print("\nOpening Napari viewer...")
    print("=" * 60)
    
    viewer = napari.Viewer(title="CellSAM Test Results")
    
    # Use first sample as default
    sample = test_data[0]
    
    # Add layers
    viewer.add_image(
        sample['image'],
        name='Original Image',
        colormap='gray'
    )
    
    viewer.add_labels(
        sample['mask'],
        name='Ground Truth (GT)',
        opacity=0.5
    )
    
    viewer.add_labels(
        sample['pred'],
        name='Model Prediction',
        opacity=0.5
    )
    
    # Create overlay (GT=green, Pred=blue, Overlap=cyan)
    overlay = np.zeros((*sample['image'].shape, 3), dtype=np.float32)
    gt_binary = (sample['mask'] > 0)
    pred_binary = (sample['pred'] > 0)
    
    overlay[:, :, 0] = 0  # Red channel
    overlay[:, :, 1] = gt_binary.astype(float)  # Green for GT
    overlay[:, :, 2] = pred_binary.astype(float)  # Blue for Pred
    
    viewer.add_image(
        overlay,
        name='Overlay (G=GT, B=Pred)',
        rgb=True,
        opacity=0.4
    )
    
    # Add all samples as a stack if multiple
    if len(test_data) > 1:
        all_images = np.stack([s['image'] for s in test_data])
        all_masks = np.stack([s['mask'] for s in test_data])
        all_preds = np.stack([s['pred'] for s in test_data])
        
        viewer.add_image(
            all_images,
            name='All Test Images (Stack)',
            colormap='gray',
            visible=False
        )
        
        viewer.add_labels(
            all_masks,
            name='All GT Masks (Stack)',
            opacity=0.5,
            visible=False
        )
        
        viewer.add_labels(
            all_preds,
            name='All Predictions (Stack)',
            opacity=0.5,
            visible=False
        )
    
    # Print instructions
    print("\n📌 Napari 使用说明:")
    print("  - 使用滚轮缩放图像")
    print("  - 左键拖拽平移")
    print("  - 点击图层名称切换显示/隐藏")
    print("  - 调整 Opacity 滑块改变透明度")
    print("  - 如有 Stack，使用滑块切换不同样本")
    print("\n按 Ctrl+C 或关闭窗口退出")
    
    napari.run()


if __name__ == "__main__":
    main()
