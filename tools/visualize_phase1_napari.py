"""
Phase 1 分割结果 Napari 可视化

展示 3 个样本 × 3 种框 (GT, DAPI, Z-line) × 3 通道 (BF, Actn2, DAPI)
使用 Phase 1 best checkpoint 推理

Usage:
    conda activate cellsam
    python tools/visualize_phase1_napari.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from augmented_dataset import AugmentedAllenDataset, load_split_ids
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
)
from detection.dapi import detect_and_create_boxes, detect_with_adaptive_box


# ============== Configuration ==============
CHECKPOINT = "checkpoints/E_phase1_rebalance_l4/best_model.pt"
N_SAMPLES = 5
SPLIT = "test"  # use test set for supervisor demo

# Colors for different box types (RGBA, 0-1)
BOX_COLORS = {
    'GT':      [0.0, 1.0, 0.0, 0.8],   # green
    'DAPI':    [1.0, 0.5, 0.0, 0.8],   # orange
    'Z-line':  [0.0, 0.7, 1.0, 0.8],   # cyan
}


def boxes_to_rectangles(boxes):
    """Convert list of [x1, y1, x2, y2] boxes to napari rectangle format.
    Returns a list of (4,2) arrays — napari expects this for 'rectangle' shape_type.
    """
    rects = []
    for b in boxes:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # napari uses (row, col) = (y, x) coordinates
        rect = np.array([
            [y1, x1],  # top-left
            [y1, x2],  # top-right
            [y2, x2],  # bottom-right
            [y2, x1],  # bottom-left
        ], dtype=np.float64)
        rects.append(rect)
    return rects


def run_inference(model, image_tensor, boxes, config, device):
    """Run inference with given boxes, return instance mask."""
    if len(boxes) == 0:
        H, W = image_tensor.shape[-2:]
        return np.zeros((H, W), dtype=np.int32)
    
    boxes_np = np.array(boxes, dtype=np.float32)
    boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)
    result = segment_with_boxes(
        model=model,
        image=image_tensor,
        boxes=boxes_tensor,
        config=config,
        device=device,
    )
    return result.instance_mask


def get_raw_channels(sample_id: str):
    """Load raw (un-normalized) image for DAPI/Actn2 detection.
    Returns (3, H, W) raw uint16/uint8 array.
    """
    raw_path = PROJECT_ROOT / "data" / "processed" / "images" / f"{sample_id}.npy"
    raw = np.load(str(raw_path))
    if raw.ndim == 3 and raw.shape[0] == 3:
        return raw  # (3, H, W)
    return raw.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def detect_dapi_boxes(raw_image):
    """DAPI detection boxes from raw image channels.
    raw_image: (3, H, W)
    """
    dapi_channel = raw_image[1]
    if dapi_channel.max() > 255:
        # Normalize to uint8 for detection
        dapi_u8 = ((dapi_channel.astype(np.float32) / dapi_channel.max()) * 255).astype(np.uint8)
    elif dapi_channel.max() <= 1.0 and dapi_channel.dtype in [np.float32, np.float64]:
        dapi_u8 = (dapi_channel * 255).astype(np.uint8)
    else:
        dapi_u8 = dapi_channel.astype(np.uint8)
    
    boxes, _, _ = detect_and_create_boxes(dapi_u8)
    return boxes if boxes else []


def detect_zline_boxes(raw_image):
    """Z-line adaptive boxes from raw image channels.
    raw_image: (3, H, W)
    """
    dapi_channel = raw_image[1]
    actn2_channel = raw_image[2]
    
    # Normalize to uint8
    def to_uint8(ch):
        if ch.max() > 255:
            return ((ch.astype(np.float32) / ch.max()) * 255).astype(np.uint8)
        elif ch.max() <= 1.0 and ch.dtype in [np.float32, np.float64]:
            return (ch * 255).astype(np.uint8)
        return ch.astype(np.uint8)
    
    dapi_u8 = to_uint8(dapi_channel)
    actn2_u8 = to_uint8(actn2_channel)
    
    boxes, _, _ = detect_with_adaptive_box(
        dapi_channel=dapi_u8,
        actn2_channel=actn2_u8,
    )
    return boxes if boxes else []


def main():
    import napari
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    ckpt_path = str(PROJECT_ROOT / CHECKPOINT)
    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    model, adapter, ckpt_info = load_cellsam_checkpoint(ckpt_path, device)
    print(f"✅ Model loaded: {ckpt_info}")
    
    infer_cfg = InferenceConfig.default()
    
    # Load test data (first N_SAMPLES, deterministic)
    split_ids = load_split_ids(SPLIT, str(PROJECT_ROOT / "data/splits"))
    selected_ids = split_ids[:N_SAMPLES]
    print(f"Using {SPLIT} set, first {len(selected_ids)} samples: {selected_ids}")
    
    dataset = AugmentedAllenDataset(
        data_dir=str(PROJECT_ROOT / "data/processed"),
        is_training=False,
        sample_ids=selected_ids,
        use_bf_only=False,
    )
    indices = list(range(len(dataset)))
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        sample_id = sample['sample_id']
        
        image = sample['image'].numpy()        # [3, H, W] normalized
        gt_mask = sample['mask'].numpy()        # [H, W] instance mask
        gt_boxes_t = sample['boxes'][:sample['num_boxes']]
        
        # Filter zero boxes
        valid = gt_boxes_t.sum(dim=1) > 0
        gt_boxes = gt_boxes_t[valid].numpy().tolist()
        
        # Load raw image for detection
        raw_image = get_raw_channels(sample_id)
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{N_SAMPLES}: {sample_id} (idx={idx})")
        print(f"  GT boxes: {len(gt_boxes)}")
        
        prefix = f"[{i+1}] "
        vis = (i == 0)  # only first sample visible by default
        
        # === Channels ===
        viewer.add_image(image[0], name=f"{prefix}BF",
                         colormap='gray', visible=vis, blending='additive')
        viewer.add_image(image[1], name=f"{prefix}DAPI",
                         colormap='blue', visible=vis, blending='additive')
        viewer.add_image(image[2], name=f"{prefix}Actn2",
                         colormap='green', visible=vis, blending='additive')
        
        # === GT Mask ===
        viewer.add_labels(gt_mask.astype(np.int32),
                          name=f"{prefix}GT Mask", visible=vis, opacity=0.3)
        
        # === BF-only input for Phase 1 inference ===
        bf = image[0]
        bf_3ch = np.stack([bf, bf, bf], axis=0)
        bf_tensor = torch.from_numpy(bf_3ch).float()
        
        # ------ A. GT boxes → Segmentation ------
        print(f"  Inference: GT boxes...")
        gt_seg = run_inference(model, bf_tensor, gt_boxes, infer_cfg, device)
        if gt_boxes:
            viewer.add_shapes(boxes_to_rectangles(gt_boxes),
                              shape_type='rectangle', name=f"{prefix}GT Boxes",
                              edge_color=BOX_COLORS['GT'], face_color=[0,0,0,0],
                              edge_width=2, visible=vis)
        viewer.add_labels(gt_seg.astype(np.int32),
                          name=f"{prefix}Seg(GT)", visible=False, opacity=0.4)
        
        # ------ B. DAPI boxes → Segmentation ------
        print(f"  Detecting: DAPI boxes...")
        dapi_boxes = detect_dapi_boxes(raw_image)
        print(f"  DAPI boxes: {len(dapi_boxes)}")
        
        dapi_seg = run_inference(model, bf_tensor, dapi_boxes, infer_cfg, device)
        if dapi_boxes:
            viewer.add_shapes(boxes_to_rectangles(dapi_boxes),
                              shape_type='rectangle', name=f"{prefix}DAPI Boxes",
                              edge_color=BOX_COLORS['DAPI'], face_color=[0,0,0,0],
                              edge_width=2, visible=False)
        viewer.add_labels(dapi_seg.astype(np.int32),
                          name=f"{prefix}Seg(DAPI)", visible=False, opacity=0.4)
        
        # ------ C. Z-line boxes → Segmentation ------
        print(f"  Detecting: Z-line boxes...")
        zline_boxes = detect_zline_boxes(raw_image)
        print(f"  Z-line boxes: {len(zline_boxes)}")
        
        zline_seg = run_inference(model, bf_tensor, zline_boxes, infer_cfg, device)
        if zline_boxes:
            viewer.add_shapes(boxes_to_rectangles(zline_boxes),
                              shape_type='rectangle', name=f"{prefix}Z-line Boxes",
                              edge_color=BOX_COLORS['Z-line'], face_color=[0,0,0,0],
                              edge_width=2, visible=False)
        viewer.add_labels(zline_seg.astype(np.int32),
                          name=f"{prefix}Seg(Z-line)", visible=False, opacity=0.4)
        
        print(f"  ✅ Done: GT={len(gt_boxes)}, DAPI={len(dapi_boxes)}, Z-line={len(zline_boxes)}")
    
    print(f"\n{'='*60}")
    print("✅ Napari viewer ready.")
    print("\nUsage Tips:")
    print("  - Toggle layer visibility with the eye icon")
    print("  - Compare segmentation by toggling Seg(GT) / Seg(DAPI) / Seg(Z-line)")
    print("  - Overlay BF + DAPI + Actn2 channels in additive blending")
    print("  - Switch samples by toggling [1] / [2] / [3] layer groups")
    
    napari.run()


if __name__ == "__main__":
    main()
