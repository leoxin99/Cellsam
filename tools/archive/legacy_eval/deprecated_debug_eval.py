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
Debug script to diagnose PQ=0 evaluation issue.
Visualize predictions vs GT to understand the mismatch.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset

def debug_single_sample():
    """Debug a single sample to understand prediction quality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = "checkpoints/boundary_enhanced_best.pt"
    print(f"Loading: {checkpoint_path}")
    
    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model_state_dict")
    else:
        model.model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded direct state_dict")
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    
    # Get first sample
    sample = dataset[0]
    image = sample['image'].numpy()
    gt_mask = sample['mask'].numpy()
    boxes = sample['boxes'].numpy()
    
    print(f"\nSample info:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {gt_mask.shape}, unique labels: {np.unique(gt_mask)}")
    print(f"  Boxes: {len(boxes)}")
    
    # Prepare input (BF only, 3 channels)
    img_bf = np.stack([image[0], image[0], image[0]], axis=0)
    img_tensor = torch.from_numpy(img_bf).float().unsqueeze(0).to(device)
    
    # Normalize
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    img_tensor = (img_tensor - img_min) / (img_max - img_min)
    
    print(f"\nInput tensor: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}")
    
    # SAM preprocess
    img_preprocessed = model.sam_preprocess(img_tensor)
    print(f"Preprocessed: shape={img_preprocessed.shape}")
    
    # Get embedding
    with torch.no_grad():
        image_embedding = model.model.image_encoder(img_preprocessed)
    print(f"Embedding: shape={image_embedding.shape}")
    
    # Segment first box
    if len(boxes) > 0:
        box = boxes[0]
        print(f"\nFirst box: {box}")
        
        box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sparse_emb, dense_emb = model.model.prompt_encoder(
                points=None, boxes=box_tensor, masks=None
            )
            low_res_masks, iou_pred = model.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
        
        print(f"Low-res mask: shape={low_res_masks.shape}, range=[{low_res_masks.min():.4f}, {low_res_masks.max():.4f}]")
        print(f"IoU prediction: {iou_pred}")
        
        # Resize
        pred = F.interpolate(
            low_res_masks, size=(1024, 1024),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        pred_sigmoid = torch.sigmoid(pred).cpu().numpy()
        pred_binary = (pred_sigmoid > 0.5).astype(np.int32)
        
        print(f"\nPrediction stats:")
        print(f"  Sigmoid range: [{pred_sigmoid.min():.4f}, {pred_sigmoid.max():.4f}]")
        print(f"  Binary area: {pred_binary.sum()} pixels")
        
        # Extract GT for this box region
        x1, y1, x2, y2 = [int(b) for b in box]
        gt_in_box = gt_mask[y1:y2, x1:x2]
        gt_labels_in_box = np.unique(gt_in_box[gt_in_box > 0])
        
        print(f"\nGT in box region:")
        print(f"  Labels: {gt_labels_in_box}")
        
        if len(gt_labels_in_box) > 0:
            target_label = gt_labels_in_box[0]
            gt_binary = (gt_mask == target_label).astype(np.int32)
            gt_area = gt_binary.sum()
            
            # Calculate IoU
            intersection = (pred_binary & gt_binary).sum()
            union = (pred_binary | gt_binary).sum()
            iou = intersection / (union + 1e-8)
            
            # Dice
            dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
            
            print(f"\nMetrics for first cell:")
            print(f"  GT area: {gt_area}")
            print(f"  Pred area: {pred_binary.sum()}")
            print(f"  Intersection: {intersection}")
            print(f"  IoU: {iou:.4f}")
            print(f"  Dice: {dice:.4f}")
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(image[0], cmap='gray')
        axes[0, 0].set_title("BF Channel")
        
        axes[0, 1].imshow(gt_mask, cmap='tab20')
        axes[0, 1].set_title(f"GT Mask ({len(np.unique(gt_mask))-1} cells)")
        
        axes[0, 2].imshow(pred_sigmoid)
        axes[0, 2].set_title(f"Prediction Sigmoid (box 0)")
        
        axes[1, 0].imshow(pred_binary)
        axes[1, 0].set_title(f"Prediction Binary (area={pred_binary.sum()})")
        
        if len(gt_labels_in_box) > 0:
            axes[1, 1].imshow(gt_binary)
            axes[1, 1].set_title(f"GT Cell (area={gt_area})")
            
            # Overlay
            overlay = np.zeros((*gt_mask.shape, 3))
            overlay[gt_binary > 0, 1] = 1  # Green for GT
            overlay[pred_binary > 0, 0] = 1  # Red for pred
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title(f"Overlay (IoU={iou:.4f})")
        
        # Draw box
        from matplotlib.patches import Rectangle
        for ax in [axes[0, 0], axes[0, 1]]:
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                            fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        plt.tight_layout()
        plt.savefig("results/debug_prediction.png", dpi=150)
        print(f"\nVisualization saved to: results/debug_prediction.png")
        plt.show()


if __name__ == "__main__":
    debug_single_sample()
