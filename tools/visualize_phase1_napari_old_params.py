"""
Phase 1 分割结果 Napari 可视化 — 旧参数版 (runtime_default)
用于与 locked_eval 版本对比

Usage:
    conda activate cellsam
    python tools/visualize_phase1_napari_old_params.py
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
SPLIT = "test"

# OLD runtime_default params (for comparison)
_OLD_DAPI = {
    "min_nucleus_area": 200,
    "max_nucleus_area": 10000,
    "edge_margin": 32,
    "size_ratio_threshold": 3.0,
    "merge_coeff": 1.2,
}
_OLD_ADAPTIVE = {
    "min_nucleus_area": 200,
    "max_nucleus_area": 10000,
    "search_radius": 256,
    "min_zlines": 15,
    "zline_threshold": 0.03,
    "edge_margin": 32,
    "size_ratio_threshold": 3.0,
    "merge_coeff": 1.2,
}

BOX_COLORS = {
    'GT':      [0.0, 1.0, 0.0, 0.8],
    'DAPI':    [1.0, 0.5, 0.0, 0.8],
    'Z-line':  [0.0, 0.7, 1.0, 0.8],
}


def boxes_to_rectangles(boxes):
    rects = []
    for b in boxes:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        rect = np.array([
            [y1, x1], [y1, x2], [y2, x2], [y2, x1]
        ])
        rects.append(rect)
    return rects


def run_inference(model, image, boxes, config, device):
    if not boxes:
        return np.zeros((1024, 1024), dtype=np.int32)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    result = segment_with_boxes(
        model=model, image=image, boxes=boxes_tensor,
        config=config, device=device,
    )
    return result.instance_mask


def get_raw_channels(sample_id):
    raw_path = PROJECT_ROOT / "data" / "processed" / "images" / f"{sample_id}.npy"
    raw = np.load(str(raw_path))
    if raw.ndim == 3 and raw.shape[0] == 3:
        return raw
    return raw.transpose(2, 0, 1)


def detect_dapi_boxes(raw_image):
    dapi_channel = raw_image[1]
    if dapi_channel.max() > 255:
        dapi_u8 = ((dapi_channel.astype(np.float32) / dapi_channel.max()) * 255).astype(np.uint8)
    elif dapi_channel.max() <= 1.0 and dapi_channel.dtype in [np.float32, np.float64]:
        dapi_u8 = (dapi_channel * 255).astype(np.uint8)
    else:
        dapi_u8 = dapi_channel.astype(np.uint8)

    # OLD params (no overrides = runtime_default)
    boxes, _, _ = detect_and_create_boxes(dapi_u8)
    return boxes if boxes else []


def detect_zline_boxes(raw_image):
    dapi_channel = raw_image[1]
    actn2_channel = raw_image[2]

    def to_uint8(ch):
        if ch.max() > 255:
            return ((ch.astype(np.float32) / ch.max()) * 255).astype(np.uint8)
        elif ch.max() <= 1.0 and ch.dtype in [np.float32, np.float64]:
            return (ch * 255).astype(np.uint8)
        return ch.astype(np.uint8)

    dapi_u8 = to_uint8(dapi_channel)
    actn2_u8 = to_uint8(actn2_channel)

    # OLD params (no overrides = runtime_default)
    boxes, _, _ = detect_with_adaptive_box(
        dapi_channel=dapi_u8,
        actn2_channel=actn2_u8,
    )
    return boxes if boxes else []


def main():
    import napari

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("⚠️ Using OLD runtime_default params (for comparison)")

    ckpt_path = str(PROJECT_ROOT / CHECKPOINT)
    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    model, adapter, ckpt_info = load_cellsam_checkpoint(ckpt_path, device)
    print(f"✅ Model loaded: {ckpt_info}")

    infer_cfg = InferenceConfig.default()

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

    viewer = napari.Viewer(title="OLD params (runtime_default)")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        sample_id = sample['sample_id']
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy()
        gt_boxes_t = sample['boxes'][:sample['num_boxes']]
        valid = gt_boxes_t.sum(dim=1) > 0
        gt_boxes = gt_boxes_t[valid].numpy().tolist()
        raw_image = get_raw_channels(sample_id)

        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{N_SAMPLES}: {sample_id} (idx={idx})")
        print(f"  GT boxes: {len(gt_boxes)}")

        prefix = f"[{i+1}] "
        vis = (i == 0)

        viewer.add_image(image[0], name=f"{prefix}BF",
                         colormap='gray', visible=vis, blending='additive')
        viewer.add_image(image[1], name=f"{prefix}DAPI",
                         colormap='blue', visible=vis, blending='additive')
        viewer.add_image(image[2], name=f"{prefix}Actn2",
                         colormap='green', visible=vis, blending='additive')
        viewer.add_labels(gt_mask.astype(np.int32),
                          name=f"{prefix}GT Mask", visible=vis, opacity=0.3)

        bf = image[0]
        bf_3ch = np.stack([bf, bf, bf], axis=0)
        bf_tensor = torch.from_numpy(bf_3ch).float()

        print(f"  Inference: GT boxes...")
        gt_seg = run_inference(model, bf_tensor, gt_boxes, infer_cfg, device)
        if gt_boxes:
            viewer.add_shapes(boxes_to_rectangles(gt_boxes),
                              shape_type='rectangle', name=f"{prefix}GT Boxes",
                              edge_color=BOX_COLORS['GT'], face_color=[0,0,0,0],
                              edge_width=2, visible=vis)
        viewer.add_labels(gt_seg.astype(np.int32),
                          name=f"{prefix}Seg(GT)", visible=False, opacity=0.4)

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
    print("✅ OLD params Napari viewer ready.")
    napari.run()


if __name__ == "__main__":
    main()
