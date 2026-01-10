"""
Evaluation script to compare old vs new model after boundary loss fine-tuning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
from tqdm import tqdm
import tifffile

sys.path.insert(0, str(Path(__file__).parent / "cellSAM_source"))
from cellSAM import get_model

sys.path.insert(0, str(Path(__file__).parent / "anti_test"))
from eval_metrics import evaluate_instance_segmentation, print_evaluation_report

# Configuration
OLD_MODEL = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"
NEW_MODEL = "d:/AI/paper/CellSam/checkpoints/boundary_20260111_012636/best_model.pt"
DATA_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
CH_BF = 0
CH_MASK = 9


def run_inference(model, image, boxes, device):
    """Run SAM inference."""
    # Prepare image
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=0)
    else:
        image_rgb = image
    
    img_tensor = torch.from_numpy(image_rgb).unsqueeze(0).float().to(device)
    
    # Preprocess and get embedding
    with torch.no_grad():
        img_preprocessed = model.sam_preprocess(img_tensor)
        embedding = model.model.image_encoder(img_preprocessed)
    
    # Predict each cell
    pred_mask = np.zeros((1024, 1024), dtype=np.int32)
    cell_id = 0
    
    for box in boxes:
        box_tensor = torch.tensor([box]).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
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
            
            pred = F.interpolate(
                low_res_masks, size=(1024, 1024),
                mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
        
        pred_binary = (pred > 0).astype(bool)
        
        cell_id += 1
        new_pixels = pred_binary & (pred_mask == 0)
        pred_mask[new_pixels] = cell_id
    
    return pred_mask


def get_gt_boxes(mask):
    """Get boxes from GT mask."""
    from skimage import measure
    boxes = []
    for region in measure.regionprops(mask):
        y1, x1, y2, x2 = region.bbox
        # Expand
        h, w = mask.shape
        bh, bw = y2 - y1, x2 - x1
        x1 = max(0, int(x1 - bw * 0.1))
        y1 = max(0, int(y1 - bh * 0.1))
        x2 = min(w, int(x2 + bw * 0.1))
        y2 = min(h, int(y2 + bh * 0.1))
        boxes.append([x1, y1, x2, y2])
    return boxes


def evaluate_model(model_path, test_files, device, name="Model"):
    """Evaluate a model on test files."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_results = []
    
    for path in tqdm(test_files, desc=f"Testing {name}"):
        with tifffile.TiffFile(path) as tif:
            data = np.squeeze(tif.asarray())
        
        bf = data[CH_BF]
        gt_mask = data[CH_MASK].astype(np.int32)
        
        # Normalize and resize
        p2, p98 = np.percentile(bf, [2, 98])
        bf_norm = np.clip((bf - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
        
        from skimage.transform import resize
        bf_resized = resize(bf_norm, (1024, 1024), preserve_range=True).astype(np.float32)
        gt_resized = resize(gt_mask, (1024, 1024), order=0, preserve_range=True).astype(np.int32)
        
        boxes = get_gt_boxes(gt_resized)
        
        if len(boxes) == 0:
            continue
        
        pred_mask = run_inference(model, bf_resized, boxes, device)
        
        results = evaluate_instance_segmentation(pred_mask, gt_resized)
        all_results.append(results)
    
    # Summary
    print(f"\n--- {name} Summary ({len(all_results)} samples) ---")
    for key in ['PQ@0.3', 'PQ@0.5', 'AJI', 'RI', 'Boundary_IoU', 'Dice', 'Max_IoU']:
        if key in all_results[0]:
            vals = [r[key] for r in all_results]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    return all_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get test files (use last 10)
    tiff_files = sorted(DATA_DIR.glob("*.tiff"))
    test_files = tiff_files[-10:]
    
    print(f"Testing on {len(test_files)} samples")
    
    # Evaluate old model
    old_results = evaluate_model(OLD_MODEL, test_files, device, "Old Model (no boundary loss)")
    
    # Evaluate new model
    new_results = evaluate_model(NEW_MODEL, test_files, device, "New Model (with boundary loss)")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON: Old vs New Model")
    print("="*60)
    
    for key in ['PQ@0.3', 'PQ@0.5', 'AJI', 'RI', 'Boundary_IoU', 'Dice', 'Max_IoU']:
        if key in old_results[0]:
            old_mean = np.mean([r[key] for r in old_results])
            new_mean = np.mean([r[key] for r in new_results])
            delta = new_mean - old_mean
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {key}: {old_mean:.4f} → {new_mean:.4f} ({arrow} {abs(delta):.4f})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
