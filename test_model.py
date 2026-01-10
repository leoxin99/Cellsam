"""
Test the trained CellSAM model - simplified version without matplotlib.
Calculates Dice scores and saves binary mask results.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import measure, io

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
from cellSAM import get_model


# Configuration
MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260107_233506/checkpoint_epoch10.pt"
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
    for region in measure.regionprops(mask.astype(np.int32)):
        y1, x1, y2, x2 = region.bbox
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1], x2 + pad)
        y2 = min(mask.shape[0], y2 + pad)
        boxes.append([x1, y1, x2, y2])
    return boxes[:max_boxes]


def calculate_dice(pred, gt):
    intersection = (pred * gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model...")
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Get test samples
    image_dir = Path(TEST_DIR) / "images"
    mask_dir = Path(TEST_DIR) / "masks"
    
    samples = sorted(list(image_dir.glob("*.npy")))[:5]
    
    print(f"\nTesting on {len(samples)} samples...")
    print("=" * 60)
    
    dice_scores = []
    
    for sample_path in samples:
        sample_id = sample_path.stem
        print(f"\nProcessing: {sample_id[:30]}...")
        
        # Load data
        image = np.load(sample_path)
        mask = np.load(mask_dir / f"{sample_id}.npy")
        
        # Resize to 1024x1024
        from skimage import transform as skt
        h, w = image.shape[:2]
        image_resized = skt.resize(image, (1024, 1024), preserve_range=True)
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        
        # Normalize
        image_norm = normalize_image(image_resized)
        
        # Get boxes
        boxes = mask_to_boxes(mask_resized.astype(np.int32))
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
            
            print(f"  Dice Score: {dice:.4f}")
            
            # Save result
            save_path = output_path / f"{sample_id[:30]}_pred.png"
            io.imsave(save_path, (pred_binary * 255).astype(np.uint8))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Samples tested: {len(dice_scores)}")
    print(f"  Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"  Min Dice:  {np.min(dice_scores):.4f}")
    print(f"  Max Dice:  {np.max(dice_scores):.4f}")
    print(f"  Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
