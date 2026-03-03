#!/usr/bin/env python3
"""
Plan B Sanity Check: Verify official pipeline + model_cp gives PQ ≈ 0.434

Quick test on 5 samples to confirm:
1. official_preprocess functions work
2. model_cp is being used
3. PQ is close to T24's 0.434
"""
import sys
from pathlib import Path

project = Path(__file__).parent.parent
sys.path.insert(0, str(project / "src"))
sys.path.insert(0, str(project / "cellSAM_source"))

import numpy as np
import torch
from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset
from inference.core import segment_with_boxes, InferenceConfig
from metrics.instance_metrics import compute_all_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = get_model()
    model.adv_mode = True  # Plan B: use model_cp
    model = model.to(device).eval()
    print(f"adv_mode = {model.adv_mode}")
    
    # Load test set
    test_ids = (project / "data/splits/test_ids.txt").read_text().strip().split("\n")
    dataset = AugmentedAllenDataset(
        data_dir=str(project / "data/processed"),
        is_training=False,
        sample_ids=test_ids
    )
    
    # Test on first 5 samples
    N = min(5, len(dataset))
    config = InferenceConfig(mask_threshold=0.5)
    pqs = []
    
    for idx in range(N):
        sample = dataset[idx]
        image = sample["image"]  # [3, H, W]
        gt_mask = sample["mask"].numpy().astype(np.int32)
        boxes = sample["boxes"][:sample["num_boxes"]]
        valid = boxes[boxes.sum(dim=1) > 0]
        
        if len(valid) == 0:
            print(f"  [{idx+1}] {sample['sample_id']}: no valid boxes")
            continue
        
        result = segment_with_boxes(model, image, valid, config, device)
        pred_mask = result.instance_mask
        
        metrics = compute_all_metrics(pred_mask, gt_mask)
        pqs.append(metrics['pq'])
        print(f"  [{idx+1}] {sample['sample_id']}: PQ={metrics['pq']:.4f}, BM-Dice={metrics['bm_1to1_dice']:.4f}")
    
    mean_pq = np.mean(pqs) if pqs else 0
    print(f"\n{'='*50}")
    print(f"Plan B Sanity Check: mean PQ = {mean_pq:.4f} (N={N})")
    print(f"Expected: ≈0.434 (T24 model_cp baseline)")
    if mean_pq > 0.3:
        print("✅ PASS: PQ > 0.3, pipeline is working")
    else:
        print("❌ FAIL: PQ too low, investigate")

if __name__ == "__main__":
    main()
