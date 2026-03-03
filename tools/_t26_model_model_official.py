#!/usr/bin/env python3
"""
T26: CellSAM 官方推理管线 — model.model (adv_mode=False) baseline

目的: 用官方完整预处理管线 (prep_2 + forward + predict)，
     但强制使用 model.model (Stage 1 权重) 而非 model_cp (Stage 2 权重)。
     对比 T24 的 model_cp 结果 (PQ=0.434) 来隔离权重差异的影响。

方法: 设 model.adv_mode = False
      → forward() 用 model.image_encoder (而非 model_cp.image_encoder)
      → predict() 用 model.prompt_encoder + model.mask_decoder
      → 预处理管线 (prep_2) 不变
"""
import sys, json
from pathlib import Path

project = Path(__file__).parent.parent
sys.path.insert(0, str(project / "src"))
sys.path.insert(0, str(project / "cellSAM_source"))

import numpy as np
import torch
from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset
from metrics.instance_metrics import compute_all_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model
    model = get_model()
    
    # ★ 关键: 强制使用 model.model (Stage 1 权重)
    print(f"adv_mode before: {model.adv_mode}")
    model.adv_mode = False
    print(f"adv_mode after:  {model.adv_mode}")
    print("★ Using model.model (Stage 1 weights) with official predict() pipeline")
    
    model = model.to(device)
    model.eval()
    
    # 2. Load test set
    test_ids_file = project / "data/splits/test_ids.txt"
    test_ids = test_ids_file.read_text().strip().split("\n")
    dataset = AugmentedAllenDataset(
        data_dir=str(project / "data/processed"),
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Test samples: {len(dataset)}")
    
    # 3. Evaluate using official predict()
    per_sample = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_id = sample["sample_id"]
        gt_mask = sample["mask"].numpy().astype(np.int32)
        
        # Get boxes
        boxes = sample["boxes"][:sample["num_boxes"]]
        valid = boxes[boxes.sum(dim=1) > 0]
        
        if len(valid) == 0:
            print(f"  {sample_id}: no valid boxes, skipping")
            continue
        
        # Prepare image for official predict()
        # predict() expects [C, H, W] tensor, range [0,255] ideally
        # But our dataset outputs [0,1]. Official predict() calls prep_2() which
        # applies PercentileThreshold + Normalize + Standardize, so the input range
        # matters less than for sam_preprocess.
        img = sample["image"]  # [3, H, W] float [0,1]
        
        # Scale to [0, 255] for consistency with official usage
        img_255 = (img * 255.0).clamp(0, 255)
        
        # Scale boxes to 1024x1024 (predict expects boxes in 1024x1024 space)
        H, W = img.shape[1], img.shape[2]
        scale_x = 1024.0 / W
        scale_y = 1024.0 / H
        scaled_boxes = valid.clone()
        scaled_boxes[:, 0] *= scale_x  # x1
        scaled_boxes[:, 1] *= scale_y  # y1
        scaled_boxes[:, 2] *= scale_x  # x2
        scaled_boxes[:, 3] *= scale_y  # y2
        
        boxes_list = [scaled_boxes.to(device)]
        
        try:
            with torch.no_grad():
                result = model.predict(
                    [img_255.to(device)],
                    boxes_per_heatmap=boxes_list
                )
            
            if result[0] is None:
                pred_mask = np.zeros_like(gt_mask, dtype=np.int32)
            else:
                pred_mask = result[0].astype(np.int32)
                # Resize pred to match GT if needed
                if pred_mask.shape != gt_mask.shape:
                    from skimage.transform import resize
                    pred_mask = resize(pred_mask, gt_mask.shape, order=0, 
                                      preserve_range=True).astype(np.int32)
            
            metrics = compute_all_metrics(pred_mask, gt_mask)
            per_sample.append({"sample_id": sample_id, **metrics})
            print(f"  [{idx+1}/{len(dataset)}] {sample_id}: PQ={metrics['pq']:.4f}")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(dataset)}] {sample_id}: ERROR - {e}")
            per_sample.append({
                "sample_id": sample_id,
                "pq": 0.0, "sq": 0.0, "rq": 0.0,
                "bm_1to1_dice": 0.0, "aji": 0.0, "semantic_dice": 0.0
            })
    
    # 4. Aggregate
    metric_keys = ["pq", "sq", "rq", "bm_1to1_dice", "bm_coverage_dice", 
                   "aji", "semantic_dice", "tp", "fp", "fn"]
    
    print("\n" + "="*60)
    print("T26 Results: model.model (Stage 1) via official predict()")
    print("="*60)
    
    aggregated = {}
    for key in metric_keys:
        vals = [s[key] for s in per_sample if key in s]
        if vals:
            aggregated[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            print(f"  {key:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Compare with T24 model_cp result
    print("\n--- Comparison ---")
    t24_pq = 0.4339
    t26_pq = aggregated.get("pq", {}).get("mean", 0)
    print(f"  T24 (model_cp, official predict):  PQ = {t24_pq:.4f}")
    print(f"  T26 (model.model, official predict): PQ = {t26_pq:.4f}")
    print(f"  Difference (model_cp - model.model): {t24_pq - t26_pq:+.4f}")
    
    # 5. Save results
    output = {
        "experiment": "T26_model_model_official_pipeline",
        "description": "CellSAM official predict() with adv_mode=False (model.model Stage 1 weights)",
        "n_samples": len(per_sample),
        "aggregated": aggregated,
        "per_sample": per_sample
    }
    
    out_path = project / "experiments/baseline_comparison/per_sample_cellsam_model_official.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
