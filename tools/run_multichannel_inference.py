"""
Comprehensive inference and visualization for CellSAM.
Shows all channels (BF, DAPI, Actn2) and saves detailed results.

Experiment ID: E16_inference_20260115
Model: E15b multi-channel (base_20260115_021255)
"""
import sys
import numpy as np
import torch
import tifffile
import napari
from pathlib import Path
from datetime import datetime
from skimage import measure, morphology, filters, transform as skt
from scipy import ndimage
import json

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cellSAM import get_model

# Configuration
RAW_DIR = Path("d:/AI/paper/CellSam/data/raw/allen_segmented_fields_full")
MODEL_PATH = Path("d:/AI/paper/CellSam/checkpoints/base_20260115_021255/best_model.pt")
OUTPUT_DIR = Path("d:/AI/paper/CellSam/experiments")
EXPERIMENT_ID = f"E16_multichannel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
NUM_SAMPLES = 10

# Channel indices
CH_BF = 0
CH_ACTN2 = 1
CH_DAPI = 4
CH_MASK = 9


def normalize_channel(img):
    """P2-P98 normalization."""
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)


def detect_nuclei_dapi(dapi_channel, min_nucleus_area=500, max_nucleus_area=30000):
    """DAPI-based nucleus detection with current parameters."""
    img_norm = normalize_channel(dapi_channel)
    try:
        thresh = filters.threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    binary = img_norm > thresh
    binary = morphology.binary_opening(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    labels = measure.label(binary)
    valid_regions = []
    
    for region in measure.regionprops(labels):
        if min_nucleus_area <= region.area <= max_nucleus_area:
            valid_regions.append(region)
    
    return valid_regions


def create_boxes_from_nuclei(regions, image_shape, margin=30, expand_ratio=3.0):
    """Convert nuclei to cell bounding boxes with smart expansion."""
    boxes = []
    h, w = image_shape
    
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        
        # Skip edge nuclei
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            continue
        
        # Get nucleus size and orientation
        nuc_h = y2 - y1
        nuc_w = x2 - x1
        
        # Use orientation-aware expansion
        if hasattr(region, 'orientation'):
            orientation = region.orientation
        else:
            orientation = 0
        
        # Calculate expansion
        major_expand = 5.0
        minor_expand = 3.0
        
        # Apply expansion based on orientation
        if abs(orientation) < np.pi / 4:  # Horizontal nucleus
            expand_x = major_expand
            expand_y = minor_expand
        else:  # Vertical nucleus
            expand_x = minor_expand
            expand_y = major_expand
        
        # Calculate new box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        new_w = nuc_w * expand_x
        new_h = nuc_h * expand_y
        
        bx1 = max(0, int(center_x - new_w / 2))
        by1 = max(0, int(center_y - new_h / 2))
        bx2 = min(w, int(center_x + new_w / 2))
        by2 = min(h, int(center_y + new_h / 2))
        
        boxes.append([bx1, by1, bx2, by2])
    
    return boxes


def run_sam_segmentation(model, device, bf, dapi, actn2, boxes):
    """Run SAM segmentation with multi-channel input using correct API."""
    import torch.nn.functional as F
    
    # Normalize each channel
    bf_norm = normalize_channel(bf)
    dapi_norm = normalize_channel(dapi)
    actn2_norm = normalize_channel(actn2)
    
    # Resize to 1024x1024
    bf_resized = skt.resize(bf_norm, (1024, 1024), preserve_range=True)
    dapi_resized = skt.resize(dapi_norm, (1024, 1024), preserve_range=True)
    actn2_resized = skt.resize(actn2_norm, (1024, 1024), preserve_range=True)
    
    # Stack as 3-channel input [BF, DAPI, Actn2] - (H, W, 3)
    multi_channel = np.stack([bf_resized, dapi_resized, actn2_resized], axis=-1)
    
    # Convert to tensor: (B, 3, H, W)
    img_tensor = torch.from_numpy(multi_channel * 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    h, w = bf.shape
    scale_y, scale_x = 1024 / h, 1024 / w
    
    # Scale boxes
    scaled_boxes = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in boxes]
    
    # Run segmentation using correct SAM API
    instance_mask = np.zeros((1024, 1024), dtype=np.int32)
    cell_id = 0
    
    with torch.no_grad():
        # Preprocess and encode image
        img_preprocessed = model.sam_preprocess(img_tensor)
        image_embedding = model.model.image_encoder(img_preprocessed)
        
        for box in scaled_boxes:
            box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
            
            try:
                # Encode prompt
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, 
                    boxes=box_tensor, 
                    masks=None
                )
                
                # Decode mask
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Upsample to 1024x1024
                pred_mask = F.interpolate(
                    low_res_masks, 
                    size=(1024, 1024), 
                    mode="bilinear", 
                    align_corners=False
                )
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()[0, 0].astype(bool)
                
                # Post-processing
                pred_binary = morphology.binary_closing(pred_binary, morphology.disk(5))
                pred_binary = ndimage.binary_fill_holes(pred_binary)
                pred_binary = morphology.remove_small_objects(pred_binary, min_size=500)
                
                # Smoothing
                from scipy.ndimage import gaussian_filter
                smoothed = gaussian_filter(pred_binary.astype(float), sigma=3)
                pred_binary = smoothed > 0.5
                pred_binary = ndimage.binary_fill_holes(pred_binary)
                
                if pred_binary.sum() > 0:
                    cell_id += 1
                    instance_mask[pred_binary] = cell_id
            except Exception as e:
                print(f"Error processing box {box}: {e}")
    
    # Resize back
    instance_mask_original = skt.resize(instance_mask, (h, w), order=0, preserve_range=True).astype(np.int32)
    
    return instance_mask_original


def compute_dice(pred, gt):
    """Compute Dice score between prediction and GT."""
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    intersection = (pred_binary * gt_binary).sum()
    return (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)


def main():
    print("=" * 60)
    print("CellSAM Multi-Channel Inference")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples: {NUM_SAMPLES}")
    print("=" * 60)
    
    # Create output directory
    exp_dir = OUTPUT_DIR / EXPERIMENT_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Get samples
    tiff_files = sorted(RAW_DIR.glob("*.tiff"))[:NUM_SAMPLES]
    
    # Results storage
    results = []
    total_gt_cells = 0
    total_pred_cells = 0
    dice_scores = []
    
    # For Napari
    all_images = []
    
    for idx, tiff_file in enumerate(tiff_files):
        print(f"\n[{idx+1}/{NUM_SAMPLES}] Processing: {tiff_file.name}")
        
        # Load data
        data = tifffile.imread(tiff_file)
        bf = data[CH_BF]
        dapi = data[CH_DAPI]
        actn2 = data[CH_ACTN2]
        gt_mask = data[CH_MASK]
        
        # Detect nuclei
        nuclei = detect_nuclei_dapi(dapi)
        boxes = create_boxes_from_nuclei(nuclei, bf.shape)
        
        print(f"  Detected: {len(boxes)} cells")
        
        # Run segmentation
        pred_mask = run_sam_segmentation(model, device, bf, dapi, actn2, boxes)
        
        # Compute metrics
        dice = compute_dice(pred_mask, gt_mask)
        gt_cells = len(np.unique(gt_mask)) - 1
        pred_cells = len(np.unique(pred_mask)) - 1
        
        total_gt_cells += gt_cells
        total_pred_cells += pred_cells
        dice_scores.append(dice)
        
        print(f"  GT: {gt_cells}, Pred: {pred_cells}, Dice: {dice:.4f}")
        
        # Store for Napari
        all_images.append({
            'name': tiff_file.stem,
            'bf': normalize_channel(bf),
            'dapi': normalize_channel(dapi),
            'actn2': normalize_channel(actn2),
            'gt_mask': gt_mask,
            'pred_mask': pred_mask,
            'boxes': boxes
        })
        
        # Store results
        results.append({
            'sample_id': tiff_file.stem,
            'gt_cells': gt_cells,
            'pred_cells': pred_cells,
            'dice': float(dice)
        })
    
    # Calculate mean dice
    mean_dice = np.mean(dice_scores)
    
    # Save results
    experiment_log = f"""# 实验记录 Experiment Log

**实验ID**: {EXPERIMENT_ID}

**模型**: `{MODEL_PATH}`

**实验描述**: E15b 多通道模型推理 (BF+DAPI+Actn2)

**测试样本数**: {NUM_SAMPLES}

---

## Prompt 生成方式

当前 prompt (边界框) 通过 **DAPI 核检测方案** 生成：

```
DAPI → Otsu → Opening → Fill Holes → Size Filter (500-30000)
    → Edge Exclude (30px) → Smart Expand (5x/3x) → Box
```

**注意**: 训练时 prompt 来自 GT Mask，推理时来自 DAPI 检测。

---

## Mean Dice 含义

Mean Dice = 所有样本的 **像素级 Dice** 均值

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
Mean Dice = ∑ Dice(i) / N
```

衡量的是**分割质量**，不是检测准确度。

---

## 总体结果 Overall Results

| 指标 | 值 |
|------|-----|
| Mean Dice | **{mean_dice:.4f}** |
| Total GT Cells | {total_gt_cells} |
| Total Pred Cells | {total_pred_cells} |

---

## 逐样本结果 Per-Sample Results

| # | Sample ID | GT | Pred | Dice |
|---|-----------|---:|-----:|------|
"""
    
    for i, r in enumerate(results):
        experiment_log += f"| {i+1} | {r['sample_id'][:40]} | {r['gt_cells']} | {r['pred_cells']} | {r['dice']:.4f} |\n"
    
    experiment_log += f"""
---

## 实验配置 Configuration

- **Input Channels**: BF + DAPI + Actn2 (3-channel)
- **Detection**: DAPI-based nucleus detection
- **Segmentation**: CellSAM (E15b fine-tuned)
- **Post-processing**: binary_closing, fill_holes, gaussian_smooth

---

## 模型对通道的理解

**问**: 模型知道三个通道代表什么吗？

**答**: **不知道**。模型只看到 (3, 1024, 1024) 的数值数组，通过训练**隐式学习**：
- 第1通道 (BF) 提供细胞结构
- 第2通道 (DAPI) 提供核位置
- 第3通道 (Actn2) 提供心肌细胞区域

模型没有显式的通道语义理解，只是学到了数值模式与分割目标的关联。

"""
    
    # Save log
    log_path = exp_dir / "experiment_log.md"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(experiment_log)
    
    # Save results JSON
    results_json = {
        'experiment_id': EXPERIMENT_ID,
        'model_path': str(MODEL_PATH),
        'mean_dice': float(mean_dice),
        'total_gt_cells': total_gt_cells,
        'total_pred_cells': total_pred_cells,
        'samples': results
    }
    with open(exp_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {exp_dir}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")
    
    # Launch Napari with first sample
    print("\nLaunching Napari...")
    viewer = napari.Viewer()
    
    for img_data in all_images:
        # Add layers for each sample
        name_prefix = img_data['name'][:20]
        
        viewer.add_image(img_data['bf'], name=f'{name_prefix}_0_BF', colormap='gray')
        viewer.add_image(img_data['dapi'], name=f'{name_prefix}_1_DAPI', colormap='blue', 
                        blending='additive', visible=False)
        viewer.add_image(img_data['actn2'], name=f'{name_prefix}_2_Actn2', colormap='green',
                        blending='additive', visible=False)
        viewer.add_labels(img_data['gt_mask'].astype(np.int32), name=f'{name_prefix}_3_GT')
        viewer.add_labels(img_data['pred_mask'].astype(np.int32), name=f'{name_prefix}_4_Pred')
        
        # Add boxes
        if img_data['boxes']:
            boxes = []
            for bx1, by1, bx2, by2 in img_data['boxes']:
                boxes.append(np.array([[by1, bx1], [by1, bx2], [by2, bx2], [by2, bx1]]))
            viewer.add_shapes(boxes, shape_type='polygon', edge_color='magenta',
                             face_color='transparent', edge_width=2, name=f'{name_prefix}_5_Boxes')
    
    napari.run()


if __name__ == "__main__":
    main()
