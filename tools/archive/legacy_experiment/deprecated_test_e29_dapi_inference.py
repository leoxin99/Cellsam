# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: One-off experiment/visualization script (Phase B cleanup)
# Replacement entry points:
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Run E29 Model Inference with DAPI Detection
Purpose: Test inference using DAPI-detected boxes instead of GT boxes
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cellSAM import get_model
from cellSAM.model import segment_cellular_image
from src.augmented_dataset import AugmentedAllenDataset, load_split_ids
from src.detection.dapi import detect_and_create_boxes
from skimage.measure import regionprops
from tqdm import tqdm


def run_dapi_inference(checkpoint_path, num_samples=71, device='cuda', 
                       use_morphology=True, merge_threshold=1.5):
    """
    Run inference with DAPI-detected boxes.
    
    Args:
        use_morphology: If True, use full DAPI detection with morphology cleaning
        merge_threshold: Binuclear merge threshold (1.5 = 50% expansion for merge)
    """
    
    print("="*60)
    print(f"E29 Inference with DAPI Detection")
    print(f"  Morphology cleaning: {use_morphology}")
    print(f"  Merge threshold: {merge_threshold}")
    print("="*60)
    
    # Load model
    print("\n1. Loading E29 model...")
    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✅ Loaded epoch {checkpoint.get('epoch', '?')}, best_dice={checkpoint.get('best_dice', 0):.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load raw data for DAPI detection (need full 3 channels)
    print("\n2. Loading validation data...")
    val_ids = load_split_ids(split='val')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        sample_ids=val_ids,
        use_bf_only=False,  # Need all 3 channels for DAPI
        is_training=False
    )
    print(f"   ✅ Loaded {len(dataset)} samples")
    
    # Metrics
    instance_dices = []
    pq_scores = []
    detection_stats = {'gt_cells': [], 'pred_cells': []}
    
    print(f"\n3. Running inference on {min(num_samples, len(dataset))} samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[idx]
            
            image = sample['image']  # (3, H, W) = [BF, DAPI, Actn2]
            gt_mask = sample['mask'].numpy()
            
            # Extract DAPI channel (index 1 in processed data)
            dapi_channel = image[1].numpy()  # DAPI is channel 1 in processed data
            
            # Detect nuclei using DAPI
            # detect_and_create_boxes returns (boxes, groups, stats)
            result = detect_and_create_boxes(
                dapi_channel,
                min_nucleus_area=200,
                max_nucleus_area=10000
            )
            boxes_list = result[0] if isinstance(result, tuple) else result
            
            if len(boxes_list) == 0:
                continue
            
            # Convert to numpy HWC for inference
            img_np = image.numpy().transpose(1, 2, 0)
            
            # Run inference
            try:
                pred_mask, _, _ = segment_cellular_image(
                    img_np, model, normalize=False,
                    bounding_boxes=boxes_list, device=device
                )
            except Exception as e:
                print(f"   Sample {idx} failed: {e}")
                continue
            
            # Stats
            gt_regions = regionprops(gt_mask)
            pred_regions = regionprops(pred_mask.astype(np.int32))
            detection_stats['gt_cells'].append(len(gt_regions))
            detection_stats['pred_cells'].append(len(pred_regions))
            
            # Compute Instance Dice per GT cell
            for gt_region in gt_regions:
                gt_cell = (gt_mask == gt_region.label)
                best_dice = 0
                for pred_region in pred_regions:
                    pred_cell = (pred_mask == pred_region.label)
                    intersection = np.sum(gt_cell & pred_cell)
                    union = np.sum(gt_cell) + np.sum(pred_cell)
                    if union > 0:
                        dice = 2 * intersection / union
                        best_dice = max(best_dice, dice)
                instance_dices.append(best_dice)
            
            # Compute PQ
            iou_threshold = 0.5
            tp, fp, fn = 0, 0, 0
            matched_ious = []
            pred_matched = set()
            
            for gt_region in gt_regions:
                gt_cell = (gt_mask == gt_region.label)
                best_iou = 0
                best_pred = None
                
                for pred_region in pred_regions:
                    if pred_region.label in pred_matched:
                        continue
                    pred_cell = (pred_mask == pred_region.label)
                    intersection = np.sum(gt_cell & pred_cell)
                    union = np.sum(gt_cell | pred_cell)
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_region.label
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_ious.append(best_iou)
                    pred_matched.add(best_pred)
                else:
                    fn += 1
            
            fp = len(pred_regions) - len(pred_matched)
            sq = np.mean(matched_ious) if matched_ious else 0
            rq = tp / (tp + 0.5*fp + 0.5*fn) if (tp + fp + fn) > 0 else 0
            pq = sq * rq
            pq_scores.append(pq)
    
    # Summary
    print("\n" + "="*60)
    print("Results:")
    print(f"  Instance Dice: {np.mean(instance_dices):.4f} ± {np.std(instance_dices):.4f}")
    print(f"  PQ@0.5: {np.mean(pq_scores):.4f} ± {np.std(pq_scores):.4f}")
    print(f"  Detection: GT={np.mean(detection_stats['gt_cells']):.1f}, "
          f"Pred={np.mean(detection_stats['pred_cells']):.1f} cells/image")
    print("="*60)
    
    return {
        'instance_dice_mean': np.mean(instance_dices),
        'instance_dice_std': np.std(instance_dices),
        'pq_mean': np.mean(pq_scores),
        'pq_std': np.std(pq_scores),
        'avg_gt_cells': np.mean(detection_stats['gt_cells']),
        'avg_pred_cells': np.mean(detection_stats['pred_cells'])
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/E29_bf_instance_best.pt')
    parser.add_argument('--samples', type=int, default=71)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-morphology', action='store_true', help='Disable morphology cleaning')
    parser.add_argument('--merge-threshold', type=float, default=1.5)
    args = parser.parse_args()
    
    run_dapi_inference(
        args.checkpoint, 
        args.samples, 
        args.device,
        use_morphology=not args.no_morphology,
        merge_threshold=args.merge_threshold
    )
