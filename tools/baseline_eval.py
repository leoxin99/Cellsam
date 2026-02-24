"""
T16 Baseline Comparison — Unified Evaluation Script

Evaluates external baselines on test(73) using compute_all_metrics.
All baselines produce instance masks → metrics computed identically.

Experiments:
  Group A (Oracle, GT boxes):
    E-B3: CellSAM Ours Phase 1 (already have results)
    E-B4: CellSAM Pretrained (no fine-tuning)
    E-B5: MedSAM

  Group B (E2E, auto-detect):
    E-B1a: Cellpose v4 (default unified model)
    E-B1b: Cellpose cyto3
    E-B1c: Cellpose cyto2
    E-B2a: StarDist 2D_versatile_fluo (DAPI) — requires stardist env
    E-B2b: StarDist 2D_versatile_he (BF RGB) — requires stardist env
    E-B6:  SAMCell — TODO
    E-B7:  CellSAM Ours E2E (use evaluate_e2e.py separately)

Usage:
  python tools/baseline_eval.py --method cellpose_v4
  python tools/baseline_eval.py --method cellpose_cyto3
  python tools/baseline_eval.py --method cellpose_cyto2
  python tools/baseline_eval.py --method cellsam_pretrained
  python tools/baseline_eval.py --method medsam
  python tools/baseline_eval.py --method all

Updated: 2026-02-21
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))

from augmented_dataset import AugmentedAllenDataset
from metrics.instance_metrics import compute_all_metrics
from inference.core import (
    segment_with_boxes, InferenceConfig, load_cellsam_checkpoint
)


# ============================================================
# Data loading
# ============================================================

def load_test_dataset():
    """Load test(73) dataset."""
    project_root = Path(__file__).parent.parent
    test_ids = (project_root / "data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir=str(project_root / "data/processed"),
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Loaded test dataset: {len(dataset)} samples")
    return dataset


def load_raw_image(dataset, idx):
    """
    Load raw 3-channel image (BF/DAPI/Actn2) before BF-only replication.
    Returns: (image_3ch, gt_mask, boxes, num_boxes, sample_id)
      image_3ch: [3, H, W] numpy float32 [0,1] — C0=BF, C1=DAPI, C2=Actn2
    """
    sample = dataset[idx]
    # The dataset with use_bf_only=True gives BF replicated 3x
    # We need to reload raw .npy to get original channels
    raw_path = dataset.samples[idx]['image_path']
    raw_img = np.load(raw_path)  # [3, H, W] or [H, W, 3] uint16/uint8
    
    # Normalize to [0, 1]
    if raw_img.ndim == 3 and raw_img.shape[0] == 3:
        pass  # already [3, H, W]
    elif raw_img.ndim == 3 and raw_img.shape[2] == 3:
        raw_img = raw_img.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    
    # Resize to 1024x1024 if needed
    from skimage import transform as sktransform
    if raw_img.shape[1] != 1024 or raw_img.shape[2] != 1024:
        resized = np.zeros((3, 1024, 1024), dtype=np.float32)
        for c in range(3):
            resized[c] = sktransform.resize(raw_img[c], (1024, 1024), preserve_range=True)
        raw_img = resized
    
    raw_img = raw_img.astype(np.float32)
    for c in range(3):
        cmin, cmax = raw_img[c].min(), raw_img[c].max()
        if cmax > cmin:
            raw_img[c] = (raw_img[c] - cmin) / (cmax - cmin)
        else:
            raw_img[c] = 0.0
    
    gt_mask = sample['mask'].numpy()
    boxes = sample['boxes']
    num_boxes = sample['num_boxes']
    sample_id = sample['sample_id']
    
    return raw_img, gt_mask, boxes, num_boxes, sample_id


# ============================================================
# Cellpose evaluation
# ============================================================

def eval_cellpose(dataset, pretrained_model=None, diameter=None):
    """
    Run Cellpose on test(73).
    
    Args:
        pretrained_model: None for v4 default, 'cyto3' or 'cyto2' for specific
        diameter: cell diameter (None = auto-estimate)
    """
    from cellpose.models import CellposeModel
    
    model_name = pretrained_model or "v4_default"
    print(f"\n{'='*60}")
    print(f"Cellpose ({model_name}) — E2E on BF grayscale")
    print(f"{'='*60}")
    
    if pretrained_model:
        model = CellposeModel(pretrained_model=pretrained_model, gpu=True)
    else:
        model = CellposeModel(gpu=True)
    
    all_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc=f"Cellpose-{model_name}"):
        raw_img, gt_mask, _, _, sample_id = load_raw_image(dataset, idx)
        
        # Use BF channel (C0), convert to uint8 for cellpose
        bf = raw_img[0]  # [1024, 1024] float32 [0,1]
        bf_uint8 = (bf * 255).astype(np.uint8)
        
        # Run cellpose — single channel grayscale
        try:
            masks, flows, styles = model.eval(
                bf_uint8,
                diameter=diameter,
            )
            
            m = compute_all_metrics(masks.astype(np.int32), gt_mask.astype(np.int32))
            m['sample_id'] = sample_id
            m['n_cellpose_cells'] = int(masks.max())
            all_metrics.append(m)
        except Exception as e:
            print(f"  ⚠️ Sample {idx} ({sample_id}) failed: {e}")
            all_metrics.append({
                'sample_id': sample_id, 'error': str(e),
                'pq': 0, 'bm_1to1_dice': 0, 'aji': 0
            })
    
    return f"cellpose_{model_name}", all_metrics


# ============================================================
# CellSAM Pretrained (no fine-tuning) — Oracle
# ============================================================

def eval_cellsam_pretrained(dataset):
    """
    CellSAM pretrained (E-B4): Oracle evaluation with GT boxes.
    Uses get_model() without loading any checkpoint = original pretrained weights.
    """
    print(f"\n{'='*60}")
    print(f"CellSAM Pretrained — Oracle (GT boxes)")
    print(f"{'='*60}")
    
    from cellSAM import get_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model = model.to(device)
    model.eval()
    
    infer_cfg = InferenceConfig.default()
    print(f"Inference config: threshold={infer_cfg.mask_threshold}, "
          f"conflict={infer_cfg.conflict_policy}")
    
    all_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc="CellSAM-Pretrained"):
        sample = dataset[idx]
        image = sample['image']  # [3, 1024, 1024] BF replicated 3x
        gt_mask = sample['mask'].numpy()
        boxes = sample['boxes']
        num_boxes = sample['num_boxes']
        sample_id = sample['sample_id']
        
        # Filter valid boxes
        valid_mask = boxes[:num_boxes].sum(dim=1) > 0
        valid_boxes = boxes[:num_boxes][valid_mask]
        
        if len(valid_boxes) == 0:
            continue
        
        try:
            result = segment_with_boxes(
                model=model,
                image=image,
                boxes=valid_boxes,
                config=infer_cfg,
                device=str(device),
            )
            
            m = compute_all_metrics(result.instance_mask, gt_mask.astype(np.int32))
            m['sample_id'] = sample_id
            m['conflict_pixels'] = result.conflict_pixels
            all_metrics.append(m)
        except Exception as e:
            print(f"  ⚠️ Sample {idx} ({sample_id}) failed: {e}")
            continue
    
    return "cellsam_pretrained", all_metrics


# ============================================================
# MedSAM — Oracle (GT boxes)
# ============================================================

def eval_medsam(dataset):
    """
    MedSAM (E-B5): Oracle evaluation with GT boxes.
    Uses medsam_vit_b.pth weights.
    """
    print(f"\n{'='*60}")
    print(f"MedSAM — Oracle (GT boxes)")
    print(f"{'='*60}")
    
    import torch.nn.functional as F
    
    medsam_path = Path(__file__).parent.parent / "checkpoints" / "medsam_vit_b_real.pth"
    if not medsam_path.exists():
        print(f"❌ MedSAM weights not found at {medsam_path}")
        print(f"   Download from: https://zenodo.org/records/10691958")
        print(f"   Place at: {medsam_path}")
        return "medsam", []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MedSAM uses the same SAM ViT-B architecture but with medical fine-tuning
    # We use segment_anything directly
    from segment_anything import sam_model_registry
    
    sam = sam_model_registry["vit_b"](checkpoint=str(medsam_path))
    sam = sam.to(device)
    sam.eval()
    
    infer_cfg = InferenceConfig.default()
    all_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc="MedSAM"):
        sample = dataset[idx]
        image = sample['image']  # [3, 1024, 1024]
        gt_mask = sample['mask'].numpy()
        boxes = sample['boxes']
        num_boxes = sample['num_boxes']
        sample_id = sample['sample_id']
        
        valid_mask = boxes[:num_boxes].sum(dim=1) > 0
        valid_boxes = boxes[:num_boxes][valid_mask]
        
        if len(valid_boxes) == 0:
            continue
        
        try:
            with torch.no_grad():
                # MedSAM inference: same as SAM but with MedSAM weights
                img = image.unsqueeze(0).to(device)  # [1, 3, 1024, 1024]
                
                # SAM preprocessing + encoding (whole image, once)
                img_preprocessed = sam.preprocess(img)
                image_embedding = sam.image_encoder(img_preprocessed)
                
                all_masks_list = []
                
                for i in range(len(valid_boxes)):
                    box = valid_boxes[i:i+1].unsqueeze(0).to(device)  # [1, 1, 4]
                    
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None, boxes=box, masks=None
                    )
                    
                    low_res_masks, iou_pred = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    
                    upscaled = F.interpolate(
                        low_res_masks, size=(1024, 1024),
                        mode="bilinear", align_corners=False
                    )
                    pred_sigmoid = torch.sigmoid(upscaled[0, 0]).cpu()
                    all_masks_list.append(pred_sigmoid)
            
            # Free VRAM between samples
            del img, img_preprocessed, image_embedding
            torch.cuda.empty_cache()
            
            # Conflict resolution (same as our pipeline)
            if all_masks_list:
                stacked = torch.stack(all_masks_list, dim=0).numpy()
                from inference.core import resolve_conflicts
                instance_mask, conflict_pixels = resolve_conflicts(
                    stacked, infer_cfg.mask_threshold, infer_cfg.conflict_policy
                )
                
                m = compute_all_metrics(instance_mask, gt_mask.astype(np.int32))
                m['sample_id'] = sample_id
                m['conflict_pixels'] = int(conflict_pixels)
                all_metrics.append(m)
        except Exception as e:
            print(f"  ⚠️ Sample {idx} ({sample_id}) failed: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    return "medsam", all_metrics


# ============================================================
# Aggregation + saving
# ============================================================

METRIC_KEYS = [
    'bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
    'pq', 'sq', 'rq', 'aji', 'semantic_dice',
    'tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells'
]


def aggregate_metrics(all_metrics):
    """Compute mean/std for each metric."""
    agg = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in all_metrics if key in m and not isinstance(m.get(key), str)]
        if values:
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'n': len(values),
            }
    return agg


def print_results(method_name, agg):
    """Pretty-print aggregated results."""
    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    for key in ['pq', 'bm_1to1_dice', 'aji', 'semantic_dice', 'sq', 'rq']:
        if key in agg:
            print(f"  {key:20s}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}  (n={agg[key]['n']})")
    for key in ['tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells']:
        if key in agg:
            print(f"  {key:20s}: {agg[key]['mean']:.1f} ± {agg[key]['std']:.1f}")


def save_results(results_dict, output_dir):
    """Save all results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "task": "T16 Baseline Comparison",
        "test_set": "test(73)",
        "methods": {}
    }
    
    for method_name, (agg, per_sample) in results_dict.items():
        output["methods"][method_name] = {
            "aggregated": agg,
            "n_samples": len(per_sample),
        }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Also save per-sample details
    for method_name, (agg, per_sample) in results_dict.items():
        detail_path = output_dir / f"per_sample_{method_name}.json"
        with open(detail_path, 'w') as f:
            json.dump(per_sample, f, indent=2, default=str)
    
    return results_path


# ============================================================
# Main
# ============================================================

METHODS = {
    'cellpose_v4': lambda ds: eval_cellpose(ds, pretrained_model=None),
    'cellpose_cyto3': lambda ds: eval_cellpose(ds, pretrained_model='cyto3'),
    'cellpose_cyto2': lambda ds: eval_cellpose(ds, pretrained_model='cyto2'),
    'cellsam_pretrained': lambda ds: eval_cellsam_pretrained(ds),
    'medsam': lambda ds: eval_medsam(ds),
}


def main():
    parser = argparse.ArgumentParser(description='T16 Baseline Comparison Evaluation')
    parser.add_argument('--method', type=str, default='all',
                        choices=list(METHODS.keys()) + ['all', 'cellpose_all', 'oracle_all'],
                        help='Which baseline method to evaluate')
    parser.add_argument('--output', type=str, default='experiments/baseline_comparison',
                        help='Output directory for results')
    args = parser.parse_args()
    
    dataset = load_test_dataset()
    results = {}
    
    if args.method == 'all':
        methods_to_run = list(METHODS.keys())
    elif args.method == 'cellpose_all':
        methods_to_run = ['cellpose_v4', 'cellpose_cyto3', 'cellpose_cyto2']
    elif args.method == 'oracle_all':
        methods_to_run = ['cellsam_pretrained', 'medsam']
    else:
        methods_to_run = [args.method]
    
    for method_key in methods_to_run:
        method_name, per_sample = METHODS[method_key](dataset)
        agg = aggregate_metrics(per_sample)
        print_results(method_name, agg)
        results[method_name] = (agg, per_sample)
    
    save_results(results, args.output)
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<25s} {'PQ':>8s} {'BM-Dice':>10s} {'AJI':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    for method_name, (agg, _) in results.items():
        pq = agg.get('pq', {}).get('mean', 0)
        dice = agg.get('bm_1to1_dice', {}).get('mean', 0)
        aji = agg.get('aji', {}).get('mean', 0)
        print(f"  {method_name:<25s} {pq:>8.4f} {dice:>10.4f} {aji:>8.4f}")
    
    # Reference: Our Phase 1
    print(f"  {'--- Reference ---':<25s}")
    print(f"  {'CellSAM Ours (Phase1)':<25s} {'0.4640':>8s} {'0.6950':>10s} {'0.5190':>8s}")


if __name__ == "__main__":
    main()
