"""
Visualize and SAVE test results.
Creates timestamped experiment folder with:
- Segmentation images (GT, Pred, overlay)
- Experiment log document
- Metrics summary
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from skimage import transform as skt, io
from scipy import ndimage
from skimage import measure, filters, morphology
from scipy.spatial.distance import cdist
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
from cellSAM.model import get_model

# Configuration
RAW_TIFF_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
MODEL_PATH = "d:/AI/paper/CellSam/checkpoints/boundary_20260111_012636/best_model.pt"
RESULTS_BASE_DIR = Path("d:/AI/paper/CellSam/experiments")



# Test sample IDs
TEST_SAMPLES = [
    "cf4fb0e8_5500000013_63X_20190807_S1_P27_B4",
    "3a3cf60a_5500000014_63X_20190816_S2_P14_C4",
    "27e55ff3_5500000013_63X_20190807_S1_P10_B5",
    "ec4c125c_5500000013_63X_20190807_S1_P5_B4",
    "60f3d143_5500000014_63X_20190816_S2_P12_C4",
    "5c2b8632_5500000013_63X_20190807_S2_P28_C3",
    "570acc96_5500000013_63X_20190807_S1_P14_B3",
    "43283e18_5500000013_63X_20190807_S1_P20_B3",
    "ebfc8c4d_5500000013_63X_20190807_S1_P30_B4",
    "39531263_5500000013_63X_20190807_S1_P13_B3",
]

CH_BRIGHTFIELD = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9


def normalize_channel(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)
    return img_norm.astype(np.float32)


def detect_nuclei_dapi(dapi_channel, min_nucleus_area=500, max_nucleus_area=15000,
                        relative_size_threshold=0.2):
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
        if region.area <= max_nucleus_area:
            all_regions.append(region)
            all_areas.append(region.area)
    
    if len(all_areas) > 0:
        median_area = np.median(all_areas)
        min_relative_area = median_area * relative_size_threshold
    else:
        return [], []
    
    valid_regions = [r for r in all_regions if r.area >= min_nucleus_area and r.area >= min_relative_area]
    return valid_regions, labels


def merge_close_nuclei(regions, merge_distance=100):
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
    y1, x1, y2, x2 = region.bbox
    h, w = image_shape
    return x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin


def create_bounding_boxes(cell_region_groups, image_shape, expansion_factor=6.0, 
                           exclude_edges=True, margin=30):
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
    regions, _ = detect_nuclei_dapi(dapi_channel)
    cell_groups = merge_close_nuclei(regions)
    boxes = create_bounding_boxes(cell_groups, image_shape)
    return boxes


def compute_dice(pred, gt):
    intersection = (pred * gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)


def run_sam_segmentation(model, device, bf_image, boxes):
    """Run SAM segmentation with post-processing for solid regions."""
    bf_norm = normalize_channel(bf_image)
    bf_resized = skt.resize(bf_norm, (1024, 1024), preserve_range=True)
    bf_rgb = np.stack([bf_resized] * 3, axis=-1)
    
    img_tensor = torch.from_numpy(bf_rgb * 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    h, w = bf_image.shape
    scale_y, scale_x = 1024 / h, 1024 / w
    
    scaled_boxes = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in boxes]
    
    instance_mask = np.zeros((1024, 1024), dtype=np.int32)
    cell_id = 0
    
    with torch.no_grad():
        img_preprocessed = model.sam_preprocess(img_tensor)
        image_embedding = model.model.image_encoder(img_preprocessed)
        
        for box in scaled_boxes:
            box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
            
            try:
                sparse_emb, dense_emb = model.model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                pred_mask = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()[0, 0].astype(bool)
                
                # Post-processing
                pred_binary = morphology.binary_closing(pred_binary, morphology.disk(5))
                pred_binary = ndimage.binary_fill_holes(pred_binary)
                pred_binary = morphology.remove_small_objects(pred_binary, min_size=500)
                
                labeled = measure.label(pred_binary)
                if labeled.max() > 0:
                    regions = measure.regionprops(labeled)
                    largest = max(regions, key=lambda r: r.area)
                    pred_binary = (labeled == largest.label)
                
                cell_id += 1
                new_pixels = pred_binary & (instance_mask == 0)
                instance_mask[new_pixels] = cell_id
                
            except Exception as e:
                print(f"    Box failed: {e}")
                continue
    
    instance_resized = skt.resize(instance_mask, (h, w), order=0, preserve_range=True).astype(np.int32)
    return instance_resized


def mask_to_rgb(mask, cmap=None):
    """Convert instance mask to RGB for visualization."""
    if cmap is None:
        cmap = np.array([
            [0, 0, 0],        # Background
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [128, 0, 0],      # Dark Red
            [0, 128, 0],      # Dark Green
            [0, 0, 128],      # Dark Blue
            [255, 128, 0],    # Orange
            [128, 0, 255],    # Purple
        ], dtype=np.uint8)
    
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(1, mask.max() + 1):
        color_idx = i % (len(cmap) - 1) + 1
        rgb[mask == i] = cmap[color_idx]
    return rgb


def save_comparison_image(bf, gt_mask, pred_mask, save_path, sample_id):
    """Save side-by-side comparison image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(bf, cmap='gray')
    axes[0].set_title('Brightfield')
    axes[0].axis('off')
    
    gt_rgb = mask_to_rgb(gt_mask)
    axes[1].imshow(bf, cmap='gray')
    axes[1].imshow(gt_rgb, alpha=0.5)
    axes[1].set_title(f'GT ({gt_mask.max()} cells)')
    axes[1].axis('off')
    
    pred_rgb = mask_to_rgb(pred_mask)
    axes[2].imshow(bf, cmap='gray')
    axes[2].imshow(pred_rgb, alpha=0.5)
    axes[2].set_title(f'Pred ({pred_mask.max()} cells)')
    axes[2].axis('off')
    
    plt.suptitle(sample_id[:40], fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Skip Napari visualization')
    args = parser.parse_args()
    
    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_BASE_DIR / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    images_dir = exp_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f'='*70)
    print(f'EXPERIMENT: {timestamp}')
    print(f'Output: {exp_dir}')
    print(f'='*70)
    
    # Load model
    device = torch.device("cpu")
    print(f'\nLoading model from: {MODEL_PATH}')
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded!')
    
    # Process samples
    all_results = []
    
    for i, sample_id in enumerate(TEST_SAMPLES):
        print(f'\n[{i+1}/10] Processing {sample_id[:40]}...')
        
        tiff_files = list(RAW_TIFF_DIR.glob(f"{sample_id}*.tiff"))
        if not tiff_files:
            print(f'  SKIP: TIFF not found')
            continue
        
        with tifffile.TiffFile(tiff_files[0]) as tif:
            data = np.squeeze(tif.asarray())
        
        brightfield = data[CH_BRIGHTFIELD]
        dapi = data[CH_DAPI]
        gt_mask = data[CH_MASK].astype(np.int32)
        
        # Detect and segment
        detected_boxes = dapi_detect_cells(dapi, brightfield.shape)
        print(f'  Detected: {len(detected_boxes)} cells')
        
        if len(detected_boxes) > 0:
            pred_mask = run_sam_segmentation(model, device, brightfield, detected_boxes)
        else:
            pred_mask = np.zeros_like(brightfield, dtype=np.int32)
        
        # Calculate metrics
        gt_binary = (gt_mask > 0).astype(np.float32)
        pred_binary = (pred_mask > 0).astype(np.float32)
        dice = compute_dice(pred_binary, gt_binary)
        
        n_gt = len(np.unique(gt_mask)) - 1
        n_pred = len(np.unique(pred_mask)) - 1
        
        print(f'  GT cells: {n_gt}, Pred cells: {n_pred}, Dice: {dice:.4f}')
        
        # Save comparison image
        bf_norm = normalize_channel(brightfield)
        save_path = images_dir / f"{i+1:02d}_{sample_id[:30]}.png"
        save_comparison_image(bf_norm, gt_mask, pred_mask, save_path, sample_id)
        
        # Save individual masks
        np.save(images_dir / f"{i+1:02d}_gt_mask.npy", gt_mask)
        np.save(images_dir / f"{i+1:02d}_pred_mask.npy", pred_mask)
        
        all_results.append({
            'sample_id': sample_id,
            'gt_cells': n_gt,
            'pred_cells': n_pred,
            'dice': dice,
        })
    
    # Calculate overall metrics
    mean_dice = np.mean([r['dice'] for r in all_results])
    total_gt = sum(r['gt_cells'] for r in all_results)
    total_pred = sum(r['pred_cells'] for r in all_results)
    
    # Generate experiment log
    log_path = exp_dir / "experiment_log.md"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"# 实验记录 Experiment Log\n\n")
        f.write(f"**实验时间**: {timestamp}\n\n")
        f.write(f"**模型**: `{MODEL_PATH}`\n\n")
        f.write(f"**测试样本数**: {len(all_results)}\n\n")
        f.write(f"---\n\n")
        f.write(f"## 总体结果 Overall Results\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| Mean Dice | **{mean_dice:.4f}** |\n")
        f.write(f"| Total GT Cells | {total_gt} |\n")
        f.write(f"| Total Pred Cells | {total_pred} |\n\n")
        f.write(f"---\n\n")
        f.write(f"## 逐样本结果 Per-Sample Results\n\n")
        f.write(f"| # | Sample ID | GT | Pred | Dice |\n")
        f.write(f"|---|-----------|----:|-----:|------|\n")
        for i, r in enumerate(all_results):
            f.write(f"| {i+1} | {r['sample_id'][:35]} | {r['gt_cells']} | {r['pred_cells']} | {r['dice']:.4f} |\n")
        f.write(f"\n---\n\n")
        f.write(f"## 实验配置 Configuration\n\n")
        f.write(f"- Detection: DAPI-based nucleus detection\n")
        f.write(f"- Segmentation: CellSAM (fine-tuned)\n")
        f.write(f"- Post-processing: binary_closing, fill_holes, largest_component\n")
    
    print(f'\n{"="*70}')
    print(f'EXPERIMENT COMPLETE')
    print(f'{"="*70}')
    print(f'Mean Dice: {mean_dice:.4f}')
    print(f'Results saved to: {exp_dir}')
    print(f'Log: {log_path}')
    
    # Launch napari
    if not args.no_gui:
        print(f'\nLaunching Napari...')
        import napari
        viewer = napari.Viewer()
        
        for i, sample_id in enumerate(TEST_SAMPLES[:len(all_results)]):
            gt = np.load(images_dir / f"{i+1:02d}_gt_mask.npy")
            pred = np.load(images_dir / f"{i+1:02d}_pred_mask.npy")
            
            viewer.add_labels(gt, name=f"GT_{i+1}", visible=(i==0), opacity=0.6)
            viewer.add_labels(pred, name=f"Pred_{i+1}", visible=(i==0), opacity=0.6)
        
        napari.run()
    else:
        print("\nSkipping Napari visualization (--no-gui specified)")


if __name__ == "__main__":
    main()
