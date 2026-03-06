#!/usr/bin/env python3
"""Napari comparison: Cellpose d=250 vs T27a vs GT
Shows first 3 test samples with: BF, DAPI, Actn2, GT masks, Cellpose pred, T27a pred.
Uses GT boxes for T27a to avoid DAPI detection dependency.
"""
import sys, warnings
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

# ── Config ──
N_SAMPLES = 3
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
T27A_CKPT = PROJECT_ROOT / "checkpoints" / "T27a_PlanB_DecoderOnly_20260302_033621" / "best_model.pt"

# ── Load IDs ──
ids = [l.strip() for l in open(SPLITS_DIR / "test_ids.txt") if l.strip()][:N_SAMPLES]
print(f"Loading {len(ids)} samples: {ids}")

# ── Load T27a model ──
print("Loading T27a model...")
import torch
from inference.core import load_cellsam_checkpoint, segment_with_boxes, InferenceConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, adapter, ckpt_info = load_cellsam_checkpoint(str(T27A_CKPT), device=device)
model.eval()
config = InferenceConfig.default()
print(f"T27a loaded: {ckpt_info}")

# ── Load Cellpose ──
print("Loading Cellpose model...")
from cellpose import models as cp_models
try:
    cp_model = cp_models.Cellpose(model_type="cyto3", gpu=True)
except AttributeError:
    cp_model = cp_models.CellposeModel(model_type="cyto3", gpu=True)

# ── Helpers ──
def normalize(ch):
    ch_min, ch_max = ch.min(), ch.max()
    if ch_max - ch_min > 1e-8:
        return (ch - ch_min) / (ch_max - ch_min)
    return np.zeros_like(ch)

def gt_mask_to_boxes(mask):
    """Extract bounding boxes from GT instance mask."""
    from skimage.measure import regionprops
    boxes = []
    for prop in regionprops(mask.astype(int)):
        y1, x1, y2, x2 = prop.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes

# ── Process each sample ──
import napari
viewer = napari.Viewer()

for idx, img_id in enumerate(ids):
    print(f"\n{'='*40}")
    print(f"Processing {idx+1}/{len(ids)}: {img_id}")
    
    # Load processed image (C, H, W) -> (H, W, C)
    img = np.load(PROCESSED_DIR / "images" / f"{img_id}.npy")
    if img.ndim == 3 and img.shape[0] in (3, 4, 5):
        img = np.transpose(img, (1, 2, 0))
    
    bf = img[:, :, 0].astype(np.float32)
    dapi = img[:, :, 1].astype(np.float32)
    actn2 = img[:, :, 2].astype(np.float32)
    
    gt_mask = np.load(PROCESSED_DIR / "masks" / f"{img_id}.npy")
    n_gt = len(np.unique(gt_mask)) - 1
    
    # ── Cellpose ──
    blank = np.zeros_like(bf)
    rgb_input = np.stack([blank, normalize(dapi), normalize(bf)], axis=-1).astype(np.float32)
    result = cp_model.eval(rgb_input, channels=[3, 2], diameter=250.0)
    cp_masks = result[0]
    n_cp = len(np.unique(cp_masks)) - 1
    
    # ── T27a (using GT boxes for fair visual comparison) ──
    boxes = gt_mask_to_boxes(gt_mask)
    print(f"  GT boxes: {len(boxes)}")
    
    if len(boxes) > 0:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        img_chw = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_chw).float()
        
        t27a_result = segment_with_boxes(
            model, img_tensor, boxes_tensor, config, device=device
        )
        t27a_masks = t27a_result.instance_mask
        n_t27a = t27a_result.n_instances
    else:
        t27a_masks = np.zeros_like(gt_mask, dtype=np.int32)
        n_t27a = 0
    
    print(f"  GT={n_gt}, Cellpose={n_cp}, T27a={n_t27a}")
    
    # ── Add to napari ──
    prefix = f"S{idx+1}"
    vis = (idx == 0)
    
    viewer.add_image(bf, name=f"{prefix} BF", colormap="gray", visible=vis)
    viewer.add_image(dapi, name=f"{prefix} DAPI", colormap="blue", visible=vis, blending="additive")
    viewer.add_image(actn2, name=f"{prefix} Actn2", colormap="green", visible=vis, blending="additive")
    
    viewer.add_labels(gt_mask.astype(np.int32), name=f"{prefix} GT ({n_gt})", visible=vis)
    viewer.add_labels(cp_masks.astype(np.int32), name=f"{prefix} Cellpose ({n_cp})", visible=vis)
    viewer.add_labels(t27a_masks.astype(np.int32), name=f"{prefix} T27a ({n_t27a})", visible=vis)

print("\n=== All samples loaded! Toggle layers to compare. ===")
napari.run()
