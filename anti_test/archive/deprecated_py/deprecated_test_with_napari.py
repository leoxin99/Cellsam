# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Test CellSAM model with napari visualization.
Analyzes class imbalance and visualizes predictions.

Usage:
    python test_with_napari.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import measure
from skimage import transform as skt

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
from cellSAM import get_model


# Configuration - use the latest fixed checkpoint
MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"
TEST_DIR = "d:/AI/paper/CellSam/data/processed"
OUTPUT_DIR = "d:/AI/paper/CellSam/test_results"


def normalize_image(img):
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    p_low, p_high = np.percentile(img, [1, 99])
    img = np.clip(img, p_low, p_high)
    return (img - p_low) / (p_high - p_low + 1e-8)


def mask_to_boxes(mask, max_boxes=20):
    boxes = []
    cell_ids = []
    for region in measure.regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1], x2 + pad)
        y2 = min(mask.shape[0], y2 + pad)
        boxes.append([x1, y1, x2, y2])
        cell_ids.append(region.label)
    return boxes[:max_boxes], cell_ids[:max_boxes]


def calculate_dice(pred, gt):
    intersection = (pred * gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)


def calculate_class_imbalance(mask):
    """
    Calculate the foreground/background ratio in the mask.
    Returns:
        fg_ratio: foreground pixels / total pixels
        bg_fg_ratio: background:foreground ratio
    """
    total_pixels = mask.size
    fg_pixels = (mask > 0).sum()
    bg_pixels = total_pixels - fg_pixels
    
    fg_ratio = fg_pixels / total_pixels
    bg_fg_ratio = bg_pixels / max(fg_pixels, 1)
    
    return fg_ratio, bg_fg_ratio


def analyze_class_imbalance_detailed(mask):
    """
    Detailed analysis of class imbalance per cell.
    The '19:1 background:foreground' ratio mentioned by Claude
    comes from analyzing individual cell regions, not the entire mask.
    """
    results = []
    props = measure.regionprops(mask.astype(np.int32))
    
    for prop in props:
        # Get the bounding box for this cell
        minr, minc, maxr, maxc = prop.bbox
        box_area = (maxr - minr) * (maxc - minc)
        cell_area = prop.area
        
        # Ratio of cell within its bounding box
        fg_ratio_in_box = cell_area / box_area if box_area > 0 else 0
        
        # Extended box (20% expansion as used in training)
        h, w = maxr - minr, maxc - minc
        expand = 0.2
        ext_minr = max(0, int(minr - h * expand))
        ext_minc = max(0, int(minc - w * expand))
        ext_maxr = min(mask.shape[0], int(maxr + h * expand))
        ext_maxc = min(mask.shape[1], int(maxc + w * expand))
        ext_box_area = (ext_maxr - ext_minr) * (ext_maxc - ext_minc)
        
        # Extract the mask region
        cell_mask_in_ext = (mask[ext_minr:ext_maxr, ext_minc:ext_maxc] == prop.label)
        cell_area_in_ext = cell_mask_in_ext.sum()
        
        fg_ratio_in_ext_box = cell_area_in_ext / ext_box_area if ext_box_area > 0 else 0
        bg_fg_ratio_in_ext_box = (ext_box_area - cell_area_in_ext) / max(cell_area_in_ext, 1)
        
        results.append({
            'cell_id': prop.label,
            'cell_area': cell_area,
            'box_area': box_area,
            'ext_box_area': ext_box_area,
            'fg_ratio_in_box': fg_ratio_in_box,
            'fg_ratio_in_ext_box': fg_ratio_in_ext_box,
            'bg_fg_ratio_in_ext_box': bg_fg_ratio_in_ext_box,
        })
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = get_model()
    
    # Load checkpoint (supports both full state_dict and model_state_dict)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from checkpoint with val_dice: {checkpoint.get('val_dice', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Get test samples
    image_dir = Path(TEST_DIR) / "images"
    mask_dir = Path(TEST_DIR) / "masks"
    
    samples = sorted(list(image_dir.glob("*.npy")))[:5]
    
    print(f"\nTesting on {len(samples)} samples...")
    print("=" * 70)
    
    # First, analyze class imbalance
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)
    print("\nThis explains the '19:1 background:foreground' ratio mentioned by Claude:")
    print("- Entire image: Most of the image IS cells (high foreground)")
    print("- Per-cell box: Each cell occupies only ~5-30% of its bounding box")
    print("- Extended box: After 20% expansion, cell occupies even less")
    print("\nWithout region-based loss, BCE optimizes for 'all background' prediction")
    print("-" * 70)
    
    all_cell_analyses = []
    
    for sample_path in samples[:3]:  # Analyze first 3 samples
        sample_id = sample_path.stem
        mask = np.load(mask_dir / f"{sample_id}.npy")
        
        # Resize to 1024x1024 to match training
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        
        # Overall image analysis
        fg_ratio, bg_fg_ratio = calculate_class_imbalance(mask_resized)
        print(f"\nSample: {sample_id[:40]}...")
        print(f"  Overall image: {fg_ratio*100:.1f}% foreground, {bg_fg_ratio:.1f}:1 bg:fg ratio")
        
        # Per-cell analysis
        cell_analyses = analyze_class_imbalance_detailed(mask_resized)
        all_cell_analyses.extend(cell_analyses)
        
        if len(cell_analyses) > 0:
            avg_fg_ratio = np.mean([c['fg_ratio_in_ext_box'] for c in cell_analyses])
            avg_bg_fg = np.mean([c['bg_fg_ratio_in_ext_box'] for c in cell_analyses])
            print(f"  Per-cell (extended box): {avg_fg_ratio*100:.1f}% fg, {avg_bg_fg:.1f}:1 bg:fg ratio")
            print(f"  Number of cells: {len(cell_analyses)}")
    
    # Summary across all analyzed cells
    if all_cell_analyses:
        avg_overall_fg = np.mean([c['fg_ratio_in_ext_box'] for c in all_cell_analyses])
        avg_overall_bg_fg = np.mean([c['bg_fg_ratio_in_ext_box'] for c in all_cell_analyses])
        print(f"\n" + "-" * 70)
        print(f"SUMMARY across {len(all_cell_analyses)} cells:")
        print(f"  Average foreground in extended box: {avg_overall_fg*100:.1f}%")
        print(f"  Average background:foreground ratio: {avg_overall_bg_fg:.1f}:1")
        print("-" * 70)
        print("\nConclusion: The '19:1' ratio is calculated WITHIN each cell's bounding box,")
        print("not across the entire image. This is why BCE Loss pushed the model to predict")
        print("'all background' - it was optimizing for the local region, not global coverage.")
    
    # Now run model inference
    print("\n" + "=" * 70)
    print("MODEL INFERENCE RESULTS")
    print("=" * 70)
    
    dice_scores = []
    results_for_napari = []
    
    for sample_path in samples:
        sample_id = sample_path.stem
        print(f"\nProcessing: {sample_id[:40]}...")
        
        # Load data
        image = np.load(sample_path)
        mask = np.load(mask_dir / f"{sample_id}.npy")
        
        # Resize to 1024x1024
        h, w = image.shape[:2]
        image_resized = skt.resize(image, (1024, 1024), preserve_range=True)
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        
        # Normalize
        image_norm = normalize_image(image_resized)
        
        # Get boxes
        boxes, cell_ids = mask_to_boxes(mask_resized.astype(np.int32))
        print(f"  Detected cells: {len(boxes)}")
        
        if len(boxes) == 0:
            print("  [SKIP] No cells detected")
            continue
        
        # Prepare tensor
        img_tensor = np.stack([image_norm, image_norm, image_norm], axis=0)
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            img_preprocessed = model.sam_preprocess(img_tensor)
            embedding = model.model.image_encoder(img_preprocessed)
            
            pred_masks = []
            logit_ranges = []
            
            for box in boxes:
                box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
                
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Record logit range to verify the fix worked
                logit_ranges.append((low_res_masks.min().item(), low_res_masks.max().item()))
                
                pred_mask = F.interpolate(
                    low_res_masks, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                )
                
                pred_masks.append(torch.sigmoid(pred_mask.squeeze()).cpu().numpy())
        
        # Combine predictions
        if pred_masks:
            combined = np.maximum.reduce(pred_masks)
            pred_binary = (combined > 0.5).astype(np.uint8)
            gt_binary = (mask_resized > 0).astype(np.float32)
            
            # Calculate Dice
            dice = calculate_dice(pred_binary.astype(np.float32), gt_binary)
            dice_scores.append(dice)
            
            # Analyze logit range (key indicator of the fix)
            min_logit = min([lr[0] for lr in logit_ranges])
            max_logit = max([lr[1] for lr in logit_ranges])
            
            print(f"  Dice Score: {dice:.4f}")
            print(f"  Logit range: [{min_logit:.2f}, {max_logit:.2f}]")
            
            # Check if logits are normal (both positive and negative values)
            if max_logit > 0 and min_logit < 0:
                print(f"  ✅ Logits are healthy (both positive and negative)")
            elif max_logit < 0:
                print(f"  ⚠️ All logits are negative (model may predict all-background)")
            
            # Store for napari
            results_for_napari.append({
                'sample_id': sample_id,
                'image': image_resized.astype(np.uint8),
                'gt_mask': mask_resized.astype(np.int32),
                'pred_mask': combined,
                'pred_binary': pred_binary,
                'dice': dice,
            })
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Samples tested: {len(dice_scores)}")
    if dice_scores:
        print(f"  Mean Dice: {np.mean(dice_scores):.4f}")
        print(f"  Min Dice:  {np.min(dice_scores):.4f}")
        print(f"  Max Dice:  {np.max(dice_scores):.4f}")
    print("=" * 70)
    
    # Launch napari visualization
    print("\n" + "=" * 70)
    print("LAUNCHING NAPARI VISUALIZATION")
    print("=" * 70)
    
    try:
        import napari
        
        viewer = napari.Viewer()
        
        for i, result in enumerate(results_for_napari):
            # Add image layer
            viewer.add_image(
                result['image'], 
                name=f"Image_{i+1} (Dice={result['dice']:.3f})",
                visible=(i == 0)  # Only first visible by default
            )
            
            # Add ground truth mask as labels
            viewer.add_labels(
                result['gt_mask'],
                name=f"GT_Mask_{i+1}",
                visible=(i == 0),
                opacity=0.5
            )
            
            # Add prediction probability
            viewer.add_image(
                result['pred_mask'],
                name=f"Pred_Prob_{i+1}",
                visible=False,
                colormap='viridis',
                opacity=0.6
            )
            
            # Add prediction binary
            viewer.add_labels(
                result['pred_binary'].astype(np.int32),
                name=f"Pred_Binary_{i+1}",
                visible=(i == 0),
                opacity=0.5
            )
        
        print("\nNapari viewer opened!")
        print("Tips:")
        print("  - Toggle layers on/off using the eye icon")
        print("  - Compare GT (colored cells) with Prediction (red overlay)")
        print("  - Check Pred_Prob layers for probability maps")
        
        napari.run()
        
    except ImportError:
        print("\nNapari not installed. Please install with:")
        print("  pip install napari[all]")
        print("\nSaving results as images instead...")
        
        from skimage import io
        for result in results_for_napari:
            save_path = output_path / f"{result['sample_id'][:30]}_pred.png"
            io.imsave(save_path, (result['pred_binary'] * 255).astype(np.uint8))
            print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
