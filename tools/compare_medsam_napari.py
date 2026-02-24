"""
MedSAM vs Ours Oracle — Napari 对比可视化

对比测试集前 3 个样本:
  - BF 图像
  - GT Mask
  - Ours Oracle (Phase1_L4) 分割结果
  - MedSAM Oracle 分割结果
  - GT Boxes

Usage:
    conda activate cellsam
    python tools/compare_medsam_napari.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from augmented_dataset import AugmentedAllenDataset, load_split_ids
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint,
    resolve_conflicts
)
from metrics.instance_metrics import compute_all_metrics

# ============== Config ==============
N_SAMPLES = 3
OURS_CKPT = "checkpoints/E_phase1_rebalance_l4/best_model.pt"
MEDSAM_CKPT = "checkpoints/medsam_vit_b_real.pth"


def boxes_to_rectangles(boxes):
    """Convert [x1,y1,x2,y2] boxes to napari rectangle format."""
    rects = []
    for b in boxes:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        rect = np.array([
            [y1, x1], [y1, x2], [y2, x2], [y2, x1],
        ], dtype=np.float64)
        rects.append(rect)
    return rects


def medsam_predict(sam_model, image_tensor, boxes, device):
    """MedSAM Oracle inference (same as baseline_eval.py eval_medsam)."""
    infer_cfg = InferenceConfig.default()
    
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        img_preprocessed = sam_model.preprocess(img)
        image_embedding = sam_model.image_encoder(img_preprocessed)
        
        all_masks = []
        for i in range(len(boxes)):
            box = boxes[i:i+1].unsqueeze(0).to(device)
            
            sparse_emb, dense_emb = sam_model.prompt_encoder(
                points=None, boxes=box, masks=None
            )
            low_res_masks, iou_pred = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            upscaled = F.interpolate(
                low_res_masks, size=(1024, 1024),
                mode="bilinear", align_corners=False
            )
            pred_sigmoid = torch.sigmoid(upscaled[0, 0]).cpu()
            all_masks.append(pred_sigmoid)
    
    del img, img_preprocessed, image_embedding
    torch.cuda.empty_cache()
    
    if not all_masks:
        return np.zeros((1024, 1024), dtype=np.int32)
    
    stacked = torch.stack(all_masks, dim=0).numpy()
    instance_mask, _ = resolve_conflicts(
        stacked, infer_cfg.mask_threshold, infer_cfg.conflict_policy
    )
    return instance_mask


def main():
    import napari
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Load Ours model
    ours_path = str(PROJECT_ROOT / OURS_CKPT)
    ours_model, adapter, ckpt_info = load_cellsam_checkpoint(ours_path, device)
    print(f"✅ Ours model loaded: {ckpt_info}")
    
    # 2. Load MedSAM model
    medsam_path = str(PROJECT_ROOT / MEDSAM_CKPT)
    from segment_anything import sam_model_registry
    medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_path)
    medsam_model = medsam_model.to(device).eval()
    print("✅ MedSAM model loaded")
    
    infer_cfg = InferenceConfig.default()
    
    # 3. Load test dataset
    split_ids = load_split_ids("test", str(PROJECT_ROOT / "data/splits"))
    selected_ids = split_ids[:N_SAMPLES]
    print(f"Samples: {selected_ids}")
    
    dataset = AugmentedAllenDataset(
        data_dir=str(PROJECT_ROOT / "data/processed"),
        is_training=False,
        sample_ids=selected_ids,
    )
    
    # 4. Create viewer
    viewer = napari.Viewer()
    
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample['sample_id']
        image = sample['image']  # [3, 1024, 1024] tensor, BF replicated
        gt_mask = sample['mask'].numpy().astype(np.int32)
        gt_boxes_t = sample['boxes'][:sample['num_boxes']]
        valid = gt_boxes_t.sum(dim=1) > 0
        gt_boxes = gt_boxes_t[valid]
        
        prefix = f"[{i+1}] "
        vis = (i == 0)
        
        # BF image
        bf = image[0].numpy()
        viewer.add_image(bf, name=f"{prefix}BF", colormap='gray',
                         visible=vis, blending='additive')
        
        # GT Mask
        viewer.add_labels(gt_mask, name=f"{prefix}GT Mask",
                          visible=vis, opacity=0.35)
        
        # GT Boxes
        if len(gt_boxes) > 0:
            viewer.add_shapes(boxes_to_rectangles(gt_boxes.numpy()),
                              shape_type='rectangle', name=f"{prefix}GT Boxes",
                              edge_color=[0, 1, 0, 0.8], face_color=[0, 0, 0, 0],
                              edge_width=2, visible=vis)
        
        # === Ours Oracle ===
        print(f"  [{i+1}] Ours Oracle inference...")
        ours_result = segment_with_boxes(
            model=ours_model, image=image, boxes=gt_boxes,
            config=infer_cfg, device=device
        )
        ours_mask = ours_result.instance_mask
        ours_m = compute_all_metrics(ours_mask, gt_mask)
        print(f"      Ours: PQ={ours_m['pq']:.3f}, BM-Dice={ours_m['bm_1to1_dice']:.3f}")
        
        viewer.add_labels(ours_mask, name=f"{prefix}Ours Oracle (PQ={ours_m['pq']:.3f})",
                          visible=vis, opacity=0.4)
        
        # === MedSAM Oracle ===
        print(f"  [{i+1}] MedSAM Oracle inference...")
        medsam_mask = medsam_predict(medsam_model, image, gt_boxes, device)
        medsam_m = compute_all_metrics(medsam_mask, gt_mask)
        print(f"      MedSAM: PQ={medsam_m['pq']:.3f}, BM-Dice={medsam_m['bm_1to1_dice']:.3f}")
        
        viewer.add_labels(medsam_mask, name=f"{prefix}MedSAM Oracle (PQ={medsam_m['pq']:.3f})",
                          visible=False, opacity=0.4)
        
        print(f"  [{i+1}] {sample_id} done. Ours PQ={ours_m['pq']:.3f} vs MedSAM PQ={medsam_m['pq']:.3f}")
    
    print(f"\n{'='*60}")
    print("✅ Napari viewer ready.")
    print("\nUsage:")
    print("  - Toggle 'Ours Oracle' vs 'MedSAM Oracle' to compare")
    print("  - Use 'GT Mask' as ground truth reference")
    print("  - Switch samples by toggling [1]/[2]/[3] groups")
    
    napari.run()


if __name__ == "__main__":
    main()
