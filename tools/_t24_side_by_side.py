#!/usr/bin/env python3
"""T24 Step 2: Side-by-side eval — official CellSAM path vs unified path.

Uses same data loading as baseline_eval.py (AugmentedAllenDataset).
Compares:
  A) Official: model_cp + predict() flow (adv_mode=True)
  B) Unified:  model.model + segment_with_boxes() (our baseline_eval path)
"""
import sys, json, torch, numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset
from metrics.instance_metrics import compute_all_metrics
from inference.core import segment_with_boxes, InferenceConfig
from cellSAM import get_model


def load_test_dataset():
    test_ids = (PROJECT_ROOT / "data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir=str(PROJECT_ROOT / "data/processed"),
        is_training=False,
        sample_ids=test_ids
    )
    return dataset


def run_official_predict(model, image_3ch_torch, boxes_np, device):
    """
    Run official CellSAM.predict() which uses model_cp when adv_mode=True.
    image_3ch_torch: [3, H, W] float [0,1]
    boxes_np: [N, 4] numpy array
    """
    model.eval()
    # predict() expects list of [C, H, W] tensors and list of box tensors
    # boxes need to be in original image coords (predict() scales to 1024 internally)
    images = [image_3ch_torch]
    boxes_list = [torch.tensor(boxes_np).float()]
    
    with torch.no_grad():
        result = model.predict(images, boxes_per_heatmap=boxes_list)
    
    if result[0] is not None:
        return result[0].astype(np.int32)
    else:
        H, W = image_3ch_torch.shape[1], image_3ch_torch.shape[2]
        return np.zeros((H, W), dtype=np.int32)


def run_unified_path(model, image_torch, boxes_torch, device):
    """
    Run our unified segment_with_boxes() which uses model.model.
    image_torch: [3, H, W] float [0,1]
    boxes_torch: [N, 4] tensor
    """
    config = InferenceConfig.default()  # uses same config as baseline_eval
    result = segment_with_boxes(
        model=model, image=image_torch, boxes=boxes_torch,
        config=config, device=str(device)
    )
    return result.instance_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading CellSAM pretrained model...")
    model = get_model()
    model = model.to(device)
    model.eval()
    print(f"  adv_mode={model.adv_mode}")
    
    dataset = load_test_dataset()
    print(f"Test set: {len(dataset)} samples")
    
    official_metrics = []
    unified_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc="T24 Side-by-side"):
        sample = dataset[idx]
        image = sample['image']       # [3, 1024, 1024] BF replicated 3x, float [0,1]
        gt_mask = sample['mask'].numpy().astype(np.int32)
        boxes = sample['boxes']
        num_boxes = sample['num_boxes']
        sid = sample['sample_id']
        
        # Filter valid boxes
        valid_mask = boxes[:num_boxes].sum(dim=1) > 0
        valid_boxes = boxes[:num_boxes][valid_mask]
        
        if len(valid_boxes) == 0:
            continue
        
        try:
            # Path A: Official predict (model_cp)
            # predict() expects image in [0, 255] range based on sam_preprocess div_255=True
            image_255 = image * 255.0
            mask_official = run_official_predict(
                model, image_255, valid_boxes.numpy(), device
            )
            m_off = compute_all_metrics(mask_official, gt_mask)
            m_off['sample_id'] = sid
            official_metrics.append(m_off)
        except Exception as e:
            print(f"  Official ERROR {sid}: {e}")
        
        try:
            # Path B: Unified (model.model) — same as baseline_eval
            mask_unified = run_unified_path(model, image, valid_boxes, device)
            m_uni = compute_all_metrics(mask_unified, gt_mask)
            m_uni['sample_id'] = sid
            unified_metrics.append(m_uni)
        except Exception as e:
            print(f"  Unified ERROR {sid}: {e}")
    
    # Report
    print("\n" + "=" * 70)
    print("T24 RESULTS: Official vs Unified CellSAM Inference")
    print("=" * 70)
    
    for label, metrics in [("Official (model_cp)", official_metrics), 
                           ("Unified (model.model)", unified_metrics)]:
        n = len(metrics)
        if n == 0:
            print(f"\n{label}: NO VALID RESULTS")
            continue
        pq = np.mean([m["pq"] for m in metrics])
        dice = np.mean([m["bm_1to1_dice"] for m in metrics])
        aji = np.mean([m["aji"] for m in metrics])
        sem = np.mean([m["semantic_dice"] for m in metrics])
        rq = np.mean([m["rq"] for m in metrics])
        sq = np.mean([m["sq"] for m in metrics])
        print(f"\n{label} (n={n}):")
        print(f"  PQ:      {pq:.4f}  (SQ={sq:.4f}, RQ={rq:.4f})")
        print(f"  BM-Dice: {dice:.4f}")
        print(f"  AJI:     {aji:.4f}")
        print(f"  Sem:     {sem:.4f}")
    
    # Delta
    if official_metrics and unified_metrics:
        pq_off = np.mean([m["pq"] for m in official_metrics])
        pq_uni = np.mean([m["pq"] for m in unified_metrics])
        d_pq = pq_off - pq_uni
        print(f"\n>>> PQ Delta (Official - Unified): {d_pq:+.4f} ({d_pq*100:+.1f}pp)")
        if abs(d_pq) < 0.02:
            print(">>> Impact: MINOR (<2pp) — footnote")
        elif abs(d_pq) < 0.05:
            print(">>> Impact: MODERATE (2-5pp) — update main table")
        else:
            print(">>> Impact: MAJOR (>5pp) — P0 review all conclusions")
    
    # Save
    results = {}
    for label, metrics in [("official_model_cp", official_metrics), ("unified_model", unified_metrics)]:
        if metrics:
            results[label] = {k: float(np.mean([m[k] for m in metrics])) 
                            for k in metrics[0] if k != 'sample_id'}
    
    out = PROJECT_ROOT / "experiments" / "t24_inference_path_audit.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
