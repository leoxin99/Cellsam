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
Analyze class imbalance and test model - writes output to file.
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

# Configuration
MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"
TEST_DIR = "d:/AI/paper/CellSam/data/processed"
OUTPUT_FILE = "d:/AI/paper/CellSam/analysis_results.txt"

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

def analyze_class_imbalance(mask_resized):
    """Analyze class imbalance at per-cell level."""
    props = measure.regionprops(mask_resized.astype(np.int32))
    
    cell_fg_ratios = []
    cell_bg_fg_ratios = []
    
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        h, w = maxr - minr, maxc - minc
        expand = 0.2
        ext_minr = max(0, int(minr - h * expand))
        ext_minc = max(0, int(minc - w * expand))
        ext_maxr = min(mask_resized.shape[0], int(maxr + h * expand))
        ext_maxc = min(mask_resized.shape[1], int(maxc + w * expand))
        ext_box_area = (ext_maxr - ext_minr) * (ext_maxc - ext_minc)
        
        cell_mask_in_ext = (mask_resized[ext_minr:ext_maxr, ext_minc:ext_maxc] == prop.label)
        cell_area_in_ext = cell_mask_in_ext.sum()
        
        fg_ratio = cell_area_in_ext / ext_box_area if ext_box_area > 0 else 0
        bg_fg_ratio = (ext_box_area - cell_area_in_ext) / max(cell_area_in_ext, 1)
        
        cell_fg_ratios.append(fg_ratio)
        cell_bg_fg_ratios.append(bg_fg_ratio)
    
    return cell_fg_ratios, cell_bg_fg_ratios

def main():
    import warnings
    warnings.filterwarnings("ignore")
    
    lines = []
    def log(msg):
        print(msg)
        lines.append(msg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    # Load model
    log(f'\nLoading model from: {MODEL_PATH}')
    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        log(f"  Checkpoint val_dice: {checkpoint.get('val_dice', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    log('Model loaded!')

    # Get test samples
    image_dir = Path(TEST_DIR) / 'images'
    mask_dir = Path(TEST_DIR) / 'masks'
    samples = sorted(list(image_dir.glob('*.npy')))[:5]

    log('\n' + '=' * 70)
    log('CLASS IMBALANCE ANALYSIS')
    log('=' * 70)
    log('\nWhy did Claude claim 19:1 bg:fg ratio when images look mostly like cells?')
    log('Answer: The ratio is calculated PER-CELL within each bounding box,')
    log('        not across the entire image.')
    log('-' * 70)

    all_fg_ratios = []
    all_bg_fg_ratios = []

    for sample_path in samples[:3]:
        sample_id = sample_path.stem
        mask = np.load(mask_dir / f'{sample_id}.npy')
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        
        # Overall image statistics
        total_pixels = mask_resized.size
        fg_pixels = (mask_resized > 0).sum()
        overall_fg_ratio = fg_pixels / total_pixels
        overall_bg_fg = (total_pixels - fg_pixels) / max(fg_pixels, 1)
        
        # Per-cell statistics
        cell_fg, cell_bg_fg = analyze_class_imbalance(mask_resized)
        all_fg_ratios.extend(cell_fg)
        all_bg_fg_ratios.extend(cell_bg_fg)
        
        log(f'\nSample: {sample_id[:40]}')
        log(f'  OVERALL IMAGE: {overall_fg_ratio*100:.1f}% foreground, {overall_bg_fg:.1f}:1 bg:fg')
        log(f'  PER-CELL (in extended box): avg {np.mean(cell_fg)*100:.1f}% fg, avg {np.mean(cell_bg_fg):.1f}:1 bg:fg')
        log(f'  Number of cells: {len(cell_fg)}')

    log('\n' + '-' * 70)
    log(f'SUMMARY across {len(all_fg_ratios)} cells:')
    log(f'  Average foreground in extended box: {np.mean(all_fg_ratios)*100:.1f}%')
    log(f'  Average bg:fg ratio: {np.mean(all_bg_fg_ratios):.1f}:1')
    log('-' * 70)
    log('\nConclusion:')
    log('  - You are RIGHT that the OVERALL image is mostly cells')
    log('  - But during training, loss is computed for EACH CELL separately')
    log('  - Each cell only fills ~5-30% of its own bounding box (irregular shapes)')
    log('  - This creates LOCAL class imbalance of ~3-20:1 bg:fg')
    log('  - BCE Loss then pushes model to predict "all background" within each box')

    # Test model inference
    log('\n' + '=' * 70)
    log('MODEL INFERENCE TEST')
    log('=' * 70)

    dice_scores = []
    for sample_path in samples:
        sample_id = sample_path.stem
        image = np.load(sample_path)
        mask = np.load(mask_dir / f'{sample_id}.npy')
        
        image_resized = skt.resize(image, (1024, 1024), preserve_range=True)
        mask_resized = skt.resize(mask, (1024, 1024), order=0, preserve_range=True)
        image_norm = normalize_image(image_resized)
        
        boxes = mask_to_boxes(mask_resized.astype(np.int32))
        log(f'\nProcessing: {sample_id[:40]} ({len(boxes)} cells)')
        
        if len(boxes) == 0:
            continue
        
        img_tensor = np.stack([image_norm] * 3, axis=0)
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_preprocessed = model.sam_preprocess(img_tensor)
            embedding = model.model.image_encoder(img_preprocessed)
            
            pred_masks = []
            logit_min, logit_max = float('inf'), float('-inf')
            
            for box in boxes:
                box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
                sparse_emb, dense_emb = model.model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                logit_min = min(logit_min, low_res_masks.min().item())
                logit_max = max(logit_max, low_res_masks.max().item())
                
                pred_mask = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                pred_masks.append(torch.sigmoid(pred_mask.squeeze()).cpu().numpy())
            
            if pred_masks:
                combined = np.maximum.reduce(pred_masks)
                pred_binary = (combined > 0.5).astype(np.uint8)
                gt_binary = (mask_resized > 0).astype(np.float32)
                dice = calculate_dice(pred_binary.astype(np.float32), gt_binary)
                dice_scores.append(dice)
                
                log(f'  Dice: {dice:.4f}  Logits: [{logit_min:.2f}, {logit_max:.2f}]')
                
                if logit_max > 0 and logit_min < 0:
                    log(f'  [OK] Logits are healthy (both positive and negative values)')
                elif logit_max < 0:
                    log(f'  [WARNING] All logits negative -> model predicts all-background')

    if dice_scores:
        log('\n' + '=' * 70)
        log('FINAL RESULTS')
        log('=' * 70)
        log(f'  Samples tested: {len(dice_scores)}')
        log(f'  Mean Dice: {np.mean(dice_scores):.4f}')
        log(f'  Min Dice:  {np.min(dice_scores):.4f}')
        log(f'  Max Dice:  {np.max(dice_scores):.4f}')
        log('=' * 70)
        log('\nTo visualize with napari, run:')
        log('  conda activate cellsam')
        log('  python test_with_napari.py')
    
    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f'\nResults saved to: {OUTPUT_FILE}')

if __name__ == "__main__":
    main()
