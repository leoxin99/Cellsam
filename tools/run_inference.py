#!/usr/bin/env python
"""
[DEPRECATED] Legacy Inference Script for CellSAM.

WARNING: This script uses the legacy inference pipeline (src/inference/pipeline.py)
which uses 'first_write' conflict policy. This is INCONSISTENT with the unified
inference core (src/inference/core.py) which defaults to 'argmax_prob'.

Use the unified entry points instead:
  - Oracle evaluation:  python tools/standardized_inference.py
  - E2E evaluation:     python tools/evaluate_e2e.py
  - Comprehensive eval: python tools/comprehensive_eval.py

All unified scripts use segment_with_boxes() + InferenceConfig.default().

This file is retained only for backward compatibility with pre-Phase-0 experiments.
"""
import warnings
warnings.warn(
    "run_inference.py uses legacy pipeline with 'first_write' conflict policy. "
    "Use tools/evaluate_e2e.py or tools/standardized_inference.py instead.",
    DeprecationWarning, stacklevel=2
)
import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile
import napari

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import run_sam_inference, mask_to_rgb, load_model, visualize_results
from detection import detect_and_create_boxes


def normalize_channel(img):
    """Normalize image using percentile-based normalization."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)


def load_sample(tiff_path):
    """Load a sample from Allen dataset TIFF file."""
    data = tifffile.imread(tiff_path)
    return {
        'bf': data[0],           # Brightfield
        'dapi': data[4],         # DAPI (nuclei)
        'actn2': data[5],        # Actn2 (cardiomyocyte marker)
        'gt': data[9],           # GT segmentation
        'name': Path(tiff_path).stem
    }


def compute_metrics(pred, gt):
    """Compute pixel and instance metrics."""
    # Pixel Dice
    pred_bin = (pred > 0).astype(float)
    gt_bin = (gt > 0).astype(float)
    intersection = (pred_bin * gt_bin).sum()
    pixel_dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)
    
    # Instance Dice (simplified)
    gt_ids = [i for i in np.unique(gt) if i > 0]
    pred_ids = [i for i in np.unique(pred) if i > 0]
    
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return {'pixel_dice': pixel_dice, 'instance_dice': 0, 'n_gt': len(gt_ids), 'n_pred': len(pred_ids)}
    
    # Simple matching: best IoU for each GT cell
    dices = []
    for gt_id in gt_ids:
        gt_mask = (gt == gt_id)
        best_dice = 0
        for pred_id in pred_ids:
            pred_mask = (pred == pred_id)
            inter = (gt_mask & pred_mask).sum()
            dice = 2 * inter / (gt_mask.sum() + pred_mask.sum() + 1e-8)
            best_dice = max(best_dice, dice)
        dices.append(best_dice)
    
    return {
        'pixel_dice': pixel_dice,
        'instance_dice': np.mean(dices) if dices else 0,
        'n_gt': len(gt_ids),
        'n_pred': len(pred_ids)
    }


def main():
    parser = argparse.ArgumentParser(description='CellSAM Unified Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/e12_boundary_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/raw/allen_segmented_fields_full',
                        help='Directory with TIFF files')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to process')
    parser.add_argument('--output', type=str, default='experiments/inference_unified',
                        help='Output directory for results')
    parser.add_argument('--no-napari', action='store_true',
                        help='Skip Napari visualization')
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = sorted(data_dir.glob('*.tiff'))[:args.samples]
    print(f"Processing {len(tiff_files)} samples...")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint if Path(args.checkpoint).exists() else None)
    device = 'cuda'
    
    # Process samples
    results = []
    all_images = []
    
    for i, tiff_path in enumerate(tiff_files):
        print(f"\n[{i+1}/{len(tiff_files)}] {tiff_path.stem}")
        
        # Load sample
        sample = load_sample(tiff_path)
        print(f"  Loaded: BF={sample['bf'].shape}, GT cells={sample['gt'].max()}")
        
        # Detect nuclei and create boxes
        boxes, cell_groups, regions = detect_and_create_boxes(sample['dapi'])
        print(f"  Detected: {len(boxes)} boxes from {len(regions)} nuclei")
        
        # Run inference
        pred = run_sam_inference(
            model=model,
            image=sample['bf'],
            boxes=boxes,
            device=device,
            apply_postprocess=True,
            validate_size=True
        )
        print(f"  Predicted: {pred.max()} cells")
        
        # Compute metrics
        metrics = compute_metrics(pred, sample['gt'])
        print(f"  Metrics: Pixel Dice={metrics['pixel_dice']:.3f}, "
              f"Instance Dice={metrics['instance_dice']:.3f}")
        
        results.append({
            'name': sample['name'],
            **metrics
        })
        
        # Store for visualization
        all_images.append({
            'name': sample['name'],
            'bf': normalize_channel(sample['bf']),
            'dapi': normalize_channel(sample['dapi']),
            'gt': sample['gt'],
            'pred': pred,
            'boxes': boxes
        })
        
        # Save comparison image
        save_path = output_dir / f"{sample['name']}_comparison.png"
        visualize_results(sample['bf'], sample['gt'], pred, save_path=str(save_path))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    avg_pixel = np.mean([r['pixel_dice'] for r in results])
    avg_instance = np.mean([r['instance_dice'] for r in results])
    print(f"Average Pixel Dice: {avg_pixel:.4f}")
    print(f"Average Instance Dice: {avg_instance:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Napari visualization
    if not args.no_napari:
        print("\nLaunching Napari...")
        v = napari.Viewer()
        
        for img in all_images:
            n = img['name']
            v.add_image(img['bf'], name=f'{n}_0_BF', colormap='gray')
            v.add_image(img['dapi'], name=f'{n}_1_DAPI', colormap='blue', blending='additive', visible=False)
            v.add_labels(img['gt'].astype(np.int32), name=f'{n}_2_GT')
            v.add_labels(img['pred'].astype(np.int32), name=f'{n}_3_Pred')
            
            if img['boxes']:
                rectangles = [np.array([[b[1], b[0]], [b[3], b[2]]]) for b in img['boxes']]
                v.add_shapes(rectangles, shape_type='rectangle', edge_color='magenta',
                            face_color='transparent', edge_width=4, name=f'{n}_4_Boxes')
        
        napari.run()


if __name__ == '__main__':
    main()
