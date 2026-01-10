"""
Full Pipeline Test: DAPI Detection + SAM Segmentation

Uses DAPI-based detection (F1=0.75) to replace CellFinder, then uses the trained
SAM model for segmentation.

Test set: 10 random samples from 478 - 50 = 428 unseen data
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import random
from pathlib import Path
from skimage import measure, filters, morphology
from skimage import transform as skt
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

import tifffile

# Add CellSAM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
from cellSAM.model import get_model

# Configuration
RAW_TIFF_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
PROCESSED_DIR = Path("d:/AI/paper/CellSam/data/processed/images")
MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/expanded_20260108_034352/best_model.pt"
OUTPUT_DIR = Path("d:/AI/paper/CellSam/anti_test")

# Channel mapping
CH_BRIGHTFIELD = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9

# Random seed for reproducibility
RANDOM_SEED = 42
N_TEST_SAMPLES = 10


def normalize_channel(img):
    """Normalize a channel to [0, 1] using percentile."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
    return img_norm.astype(np.float32)


def detect_nuclei_dapi(dapi_channel, min_nucleus_area=500, max_nucleus_area=15000,
                        relative_size_threshold=0.2):
    """Detect nuclei from DAPI with relative size filtering."""
    img_norm = normalize_channel(dapi_channel)
    
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = morphology.remove_small_objects(binary, min_size=min_nucleus_area)
    binary = ndimage.binary_fill_holes(binary)
    
    labels = measure.label(binary)
    
    all_regions = []
    all_areas = []
    
    for region in measure.regionprops(labels):
        area = region.area
        if area <= max_nucleus_area:
            all_regions.append(region)
            all_areas.append(area)
    
    if len(all_areas) > 0:
        median_area = np.median(all_areas)
        min_relative_area = median_area * relative_size_threshold
    else:
        return [], []
    
    valid_regions = []
    for region in all_regions:
        if region.area >= min_nucleus_area and region.area >= min_relative_area:
            valid_regions.append(region)
    
    return valid_regions, labels


def merge_close_nuclei(regions, merge_distance=100):
    """Merge nuclei that are close (binucleated cells)."""
    if len(regions) <= 1:
        return [[r] for r in regions]
    
    centroids = np.array([r.centroid for r in regions])
    n = len(centroids)
    distances = cdist(centroids, centroids)
    
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < merge_distance:
                union(i, j)
    
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    return [[regions[i] for i in indices] for indices in groups.values()]


def is_on_edge(region, image_shape, margin=30):
    """Check if region touches image edge."""
    y1, x1, y2, x2 = region.bbox
    h, w = image_shape
    return x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin


def create_bounding_boxes(cell_region_groups, image_shape, expansion_factor=6.0, 
                           exclude_edges=True, margin=30):
    """Create expanded bounding boxes from nucleus regions."""
    boxes = []
    h, w = image_shape
    
    for regions in cell_region_groups:
        if exclude_edges and any(is_on_edge(r, image_shape, margin) for r in regions):
            continue
        
        y_min = min(r.bbox[0] for r in regions)
        x_min = min(r.bbox[1] for r in regions)
        y_max = max(r.bbox[2] for r in regions)
        x_max = max(r.bbox[3] for r in regions)
        
        box_h, box_w = y_max - y_min, x_max - x_min
        center_y, center_x = (y_min + y_max) / 2, (x_min + x_max) / 2
        
        new_h, new_w = box_h * expansion_factor, box_w * expansion_factor
        
        x1 = int(max(0, center_x - new_w / 2))
        y1 = int(max(0, center_y - new_h / 2))
        x2 = int(min(w, center_x + new_w / 2))
        y2 = int(min(h, center_y + new_h / 2))
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes


def dapi_detect_cells(dapi_channel, image_shape):
    """Full DAPI detection pipeline."""
    # Detect nuclei
    regions, _ = detect_nuclei_dapi(dapi_channel)
    
    # Merge close nuclei
    cell_groups = merge_close_nuclei(regions)
    
    # Create boxes
    boxes = create_bounding_boxes(cell_groups, image_shape)
    
    return boxes


def compute_dice(pred, target):
    """Compute Dice score."""
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)


def get_training_sample_ids():
    """Get the IDs of training samples from processed directory."""
    training_ids = set()
    for f in PROCESSED_DIR.glob("*.npy"):
        # Extract the core ID (first part before underscore)
        sample_id = f.stem.split('_')[0]
        training_ids.add(sample_id)
    return training_ids


def get_unseen_test_samples(n_samples=10, seed=42):
    """Get n random test samples not in training set."""
    training_ids = get_training_sample_ids()
    print(f"Training samples: {len(training_ids)}")
    
    all_tiff_files = list(RAW_TIFF_DIR.glob("*.tiff"))
    
    # Filter out training samples
    unseen_files = []
    for f in all_tiff_files:
        sample_id = f.stem.split('_')[0]
        if sample_id not in training_ids:
            unseen_files.append(f)
    
    print(f"Unseen samples: {len(unseen_files)}")
    
    # Random selection
    random.seed(seed)
    selected = random.sample(unseen_files, min(n_samples, len(unseen_files)))
    
    return selected


def main():
    print('='*70)
    print('FULL PIPELINE TEST: DAPI Detection + SAM Segmentation')
    print('='*70)
    
    # Setup device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get test samples
    print("\n" + "-"*70)
    print("Selecting test samples...")
    test_files = get_unseen_test_samples(N_TEST_SAMPLES, RANDOM_SEED)
    print(f"Selected {len(test_files)} test samples")
    
    # Results storage
    all_results = []
    results_for_napari = []
    
    for idx, tiff_path in enumerate(test_files):
        sample_id = tiff_path.stem[:45]
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(test_files)}] Processing: {sample_id}...")
        
        # Load TIFF
        with tifffile.TiffFile(tiff_path) as tif:
            data = np.squeeze(tif.asarray())
        
        if len(data.shape) != 3 or data.shape[0] < 10:
            print(f"  [SKIP] Unexpected shape: {data.shape}")
            continue
        
        brightfield = data[CH_BRIGHTFIELD]
        dapi = data[CH_DAPI]
        gt_mask = data[CH_MASK]
        
        image_shape = brightfield.shape
        print(f"  Image shape: {image_shape}")
        
        # Get GT info
        gt_cell_ids = [cid for cid in np.unique(gt_mask) if cid > 0]
        n_gt = len(gt_cell_ids)
        print(f"  GT cells: {n_gt}")
        
        # Step 1: DAPI Detection
        detected_boxes = dapi_detect_cells(dapi, image_shape)
        n_detected = len(detected_boxes)
        print(f"  DAPI detected: {n_detected} cells")
        
        if n_detected == 0:
            print("  [SKIP] No cells detected")
            continue
        
        # Step 2: Prepare image for SAM
        bf_norm = normalize_channel(brightfield)
        # Resize to 1024x1024 for SAM
        bf_resized = skt.resize(bf_norm, (1024, 1024), preserve_range=True)
        bf_rgb = np.stack([bf_resized] * 3, axis=-1)  # Convert to RGB
        
        # Prepare image tensor
        img_tensor = torch.from_numpy(bf_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Scale boxes to 1024x1024
        scale_y = 1024 / image_shape[0]
        scale_x = 1024 / image_shape[1]
        
        scaled_boxes = []
        for box in detected_boxes:
            x1, y1, x2, y2 = box
            scaled_boxes.append([
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y
            ])
        
        # Step 3: SAM Segmentation for each detected cell
        combined_pred = np.zeros((1024, 1024), dtype=np.float32)
        cell_dice_scores = []
        
        # Prepare tensor (H, W, C) -> (B, C, H, W) and scale to 0-255
        # bf_rgb is 0-1, so * 255. Matches training range [0, 255].
        img_tensor = torch.from_numpy(bf_rgb * 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # DEBUG: Check input stats
        if idx == 0:
            print(f"  DEBUG: Input Tensor Stats: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}, mean={img_tensor.mean():.2f}")
        
        with torch.no_grad():
            # 1. Preprocess
            img_preprocessed = model.sam_preprocess(img_tensor)
            
            # 2. Get Image Embedding (run once per image)
            image_embedding = model.model.image_encoder(img_preprocessed)
            
            # 3. Prompt Encoder + Mask Decoder for each box
            for box_idx, box in enumerate(scaled_boxes):
                # SAM expects boxes as (B, 4) tensor
                box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
                
                try:
                    # Encode box prompt
                    sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )
                    
                    # Decode mask
                    low_res_masks, _ = model.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # Upscale mask to 1024x1024
                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(1024, 1024),
                        mode="bilinear",
                        align_corners=False,
                    )
                    
                    # Convert to binary
                    probs = torch.sigmoid(pred_mask)
                    # DEBUG prediction stats for first box
                    if idx == 0 and box_idx == 0:
                        print(f"  DEBUG: Box 0 Pred Stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
                    
                    pred_mask_binary = (probs > 0.5).float().cpu().numpy()[0, 0]
                    
                    # Add to combined prediction
                    combined_pred = np.maximum(combined_pred, pred_mask_binary)
                    
                except Exception as e:
                    print(f"    Box {box_idx} segmentation failed: {e}")
                    continue
        
        # Resize prediction back to original size
        pred_resized = skt.resize(combined_pred, image_shape, preserve_range=True) > 0.5
        
        # Calculate overall Dice
        gt_binary = (gt_mask > 0).astype(np.float32)
        overall_dice = compute_dice(pred_resized.astype(np.float32), gt_binary)
        
        print(f"  Overall Dice: {overall_dice:.4f}")
        
        # Calculate per-cell Dice
        for cell_id in gt_cell_ids:
            gt_cell = (gt_mask == cell_id).astype(np.float32)
            
            # Find overlapping prediction region
            overlap = pred_resized * gt_cell
            if overlap.sum() > 0:
                # Create prediction mask for this cell region
                pred_cell = pred_resized * (gt_cell > 0)
                cell_dice = compute_dice(pred_cell.astype(np.float32), gt_cell)
                cell_dice_scores.append(cell_dice)
        
        mean_cell_dice = np.mean(cell_dice_scores) if cell_dice_scores else 0
        print(f"  Mean per-cell Dice: {mean_cell_dice:.4f} ({len(cell_dice_scores)} cells)")
        
        all_results.append({
            'sample_id': sample_id,
            'gt_cells': n_gt,
            'detected_cells': n_detected,
            'overall_dice': overall_dice,
            'mean_cell_dice': mean_cell_dice,
            'n_evaluated_cells': len(cell_dice_scores)
        })
        
        # Store for napari
        results_for_napari.append({
            'sample_id': sample_id,
            'brightfield': bf_norm,
            'dapi': normalize_channel(dapi),
            'gt_mask': gt_mask,
            'pred_mask': pred_resized.astype(np.int32),
            'detected_boxes': detected_boxes,
        })
    
    # Summary
    print('\n' + '='*70)
    print('OVERALL RESULTS')
    print('='*70)
    
    if all_results:
        mean_overall_dice = np.mean([r['overall_dice'] for r in all_results])
        mean_cell_dice_all = np.mean([r['mean_cell_dice'] for r in all_results])
        total_gt = sum(r['gt_cells'] for r in all_results)
        total_detected = sum(r['detected_cells'] for r in all_results)
        
        print(f"Test samples: {len(all_results)}")
        print(f"Total GT cells: {total_gt}")
        print(f"Total detected cells: {total_detected}")
        print(f"Mean Overall Dice: {mean_overall_dice:.4f}")
        print(f"Mean Per-Cell Dice: {mean_cell_dice_all:.4f}")
        
        print('\n' + '-'*70)
        print('Per-sample results:')
        for r in all_results:
            print(f"  {r['sample_id'][:40]}: GT={r['gt_cells']}, Det={r['detected_cells']}, "
                  f"Dice={r['overall_dice']:.3f}, CellDice={r['mean_cell_dice']:.3f}")
    
    # Save results
    results_file = OUTPUT_DIR / 'full_pipeline_test_results.txt'
    with open(results_file, 'w') as f:
        f.write('Full Pipeline Test Results (DAPI Detection + SAM Segmentation)\n')
        f.write('='*70 + '\n\n')
        f.write(f'Model: {MODEL_PATH}\n')
        f.write(f'Test samples: {len(all_results)} (unseen data)\n\n')
        
        for r in all_results:
            f.write(f"Sample: {r['sample_id']}\n")
            f.write(f"  GT cells: {r['gt_cells']}, Detected: {r['detected_cells']}\n")
            f.write(f"  Overall Dice: {r['overall_dice']:.4f}\n")
            f.write(f"  Mean Cell Dice: {r['mean_cell_dice']:.4f}\n\n")
        
        if all_results:
            f.write('='*70 + '\n')
            f.write(f'OVERALL: Mean Dice={mean_overall_dice:.4f}, Mean Cell Dice={mean_cell_dice_all:.4f}\n')
    
    print(f'\nResults saved to: {results_file}')
    
    # Launch napari
    print('\n' + '='*70)
    print('LAUNCHING NAPARI VISUALIZATION')
    print('='*70)
    
    try:
        import napari
        
        viewer = napari.Viewer()
        
        for i, result in enumerate(results_for_napari):
            viewer.add_image(
                result['brightfield'],
                name=f"BF_{i+1}_{result['sample_id'][:15]}",
                visible=(i == 0)
            )
            viewer.add_image(
                result['dapi'],
                name=f"DAPI_{i+1}",
                visible=False,
                colormap='blue',
                blending='additive'
            )
            viewer.add_labels(
                result['gt_mask'].astype(np.int32),
                name=f"GT_{i+1}",
                visible=(i == 0),
                opacity=0.4
            )
            viewer.add_labels(
                result['pred_mask'],
                name=f"Pred_{i+1}",
                visible=(i == 0),
                opacity=0.4
            )
            
            # Add detection boxes
            if result['detected_boxes']:
                rects = [np.array([[b[1], b[0]], [b[1], b[2]], [b[3], b[2]], [b[3], b[0]]]) 
                         for b in result['detected_boxes']]
                viewer.add_shapes(rects, shape_type='polygon', edge_color='yellow', 
                                  edge_width=2, face_color='transparent', 
                                  name=f"DetBoxes_{i+1}", visible=(i == 0))
        
        print('\nNapari opened!')
        print('Legend: GT=Ground Truth, Pred=SAM Prediction, Yellow=DAPI Detection Boxes')
        
        napari.run()
        
    except ImportError:
        print('Napari not installed.')


if __name__ == "__main__":
    main()
