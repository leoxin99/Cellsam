# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: One-off experiment/visualization script (Phase B cleanup)
# Replacement entry points:
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
E21 重新评估: E12 Baseline vs Semantic Adapter (修复后)

使用修复后的 checkpoint (包含 adapter_state_dict)
使用完整评估指标体系 (E09)

运行:
    conda activate cellsam
    python tools/compare_models_v2.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project paths
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "cellSAM_source"))
sys.path.insert(0, str(project_dir / "src"))
sys.path.insert(0, str(project_dir / "anti_test"))

from cellSAM.model import get_model
from adapters.channel_adapter import IndependentChannelAdapter
from detection.dapi import detect_and_create_boxes
from eval_metrics import evaluate_instance_segmentation, print_evaluation_report


def load_samples(data_dir: Path, split_file: Path, n_samples: int = None):
    """Load test samples."""
    with open(split_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    if n_samples:
        sample_ids = sample_ids[:n_samples]
    
    samples = []
    for sid in tqdm(sample_ids, desc="Loading samples"):
        img_path = data_dir / "images" / f"{sid}.npy"
        mask_path = data_dir / "masks" / f"{sid}.npy"
        
        if not img_path.exists() or not mask_path.exists():
            continue
            
        samples.append({
            'id': sid,
            'image': np.load(img_path),  # (3, 1024, 1024)
            'mask': np.load(mask_path),  # (1024, 1024) instance mask
        })
    
    return samples


def load_model_and_adapter(checkpoint_path: Path, device, use_adapter: bool = False):
    """Load model and optionally adapter from checkpoint."""
    model = get_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"    Loaded model from dict format")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"    Loaded model from direct state_dict")
    
    model = model.to(device)
    model.eval()
    
    # Load adapter if present and requested
    adapter = None
    if use_adapter and isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
        adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
        adapter = adapter.to(device)
        adapter.eval()
        print(f"    Loaded adapter ({adapter.get_param_count()} params)")
    elif use_adapter:
        print(f"    WARNING: No adapter_state_dict in checkpoint!")
    
    return model, adapter


def segment_with_model(model, image, boxes, device, adapter=None):
    """Run segmentation with model."""
    # Prepare image for SAM
    if adapter is not None:
        # Semantic channel mapping: [Actn2, BF, DAPI]
        semantic_img = np.stack([image[2], image[0], image[1]], axis=0)  # Actn2, BF, DAPI
        img_tensor = torch.from_numpy(semantic_img).float().unsqueeze(0).to(device)
        img_tensor = adapter(img_tensor)
    else:
        # BF only (3 copies)
        img_np = np.stack([image[0], image[0], image[0]], axis=0)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    
    # Normalize to [0, 1] range
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    if img_max > img_min:
        img_tensor = (img_tensor - img_min) / (img_max - img_min)
    
    # SAM preprocess expects tensor input (B, C, H, W)
    img_preprocessed = model.sam_preprocess(img_tensor)
    
    # Run SAM
    pred_mask = np.zeros((1024, 1024), dtype=np.int32)
    
    with torch.no_grad():
        for i, box in enumerate(boxes):
            try:
                box_tensor = torch.tensor([box], dtype=torch.float32, device=device)
                
                # Get image embedding
                image_embedding = model.model.image_encoder(img_preprocessed.to(device))
                
                # Get prompt embedding
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None
                )
                
                # Decode - CellSAM returns 2 values: (low_res_masks, iou_predictions)
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Resize from 256x256 to 1024x1024
                import torch.nn.functional as F
                pred_mask_resized = F.interpolate(
                    low_res_masks, size=(1024, 1024),
                    mode='bilinear', align_corners=False
                ).squeeze()
                
                # Convert to binary mask
                mask = (torch.sigmoid(pred_mask_resized) > 0.5).cpu().numpy()
                
                # Assign instance ID
                pred_mask[mask] = i + 1
                
            except Exception as e:
                print(f"    Error on box {i}: {e}")
                continue
    
    return pred_mask


def main():
    print("=" * 70)
    print("E21 重新评估: E12 Baseline vs Semantic Adapter (修复后)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Paths
    data_dir = project_dir / "data" / "processed"
    split_file = project_dir / "data" / "splits" / "test_ids.txt"
    output_dir = project_dir / "experiments" / "e21_rerun"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoints
    e12_ckpt = project_dir / "checkpoints" / "boundary_20260111_012636" / "best_model.pt"
    adapter_ckpt = project_dir / "checkpoints" / "semantic_adapter_v2_best.pt"
    
    # Verify checkpoints
    print("\n[1] Checking checkpoints...")
    for name, path in [("E12", e12_ckpt), ("Adapter", adapter_ckpt)]:
        if path.exists():
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ['state_dict']
            has_adapter = 'adapter_state_dict' in keys if isinstance(ckpt, dict) else False
            print(f"    {name}: ✅ exists, adapter={has_adapter}")
        else:
            print(f"    {name}: ❌ NOT FOUND at {path}")
            return
    
    # Load samples
    print("\n[2] Loading test samples...")
    samples = load_samples(data_dir, split_file, n_samples=20)
    print(f"    Loaded {len(samples)} samples")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'n_samples': len(samples),
        'models': {}
    }
    
    # Test both models
    for model_name, ckpt_path, use_adapter in [
        ("E12_Baseline", e12_ckpt, False),
        ("Semantic_Adapter", adapter_ckpt, True),
    ]:
        print(f"\n[3] Testing {model_name}...")
        model, adapter = load_model_and_adapter(ckpt_path, device, use_adapter)
        
        all_metrics = []
        for sample in tqdm(samples, desc=f"  {model_name}"):
            # Detect boxes
            boxes, _, _ = detect_and_create_boxes(
                sample['image'][1],  # DAPI channel
                min_nucleus_area=2500,
                max_nucleus_area=20000
            )
            
            if not boxes:
                continue
            
            # Segment
            pred_mask = segment_with_model(model, sample['image'], boxes, device, adapter)
            
            # Evaluate with comprehensive metrics
            metrics = evaluate_instance_segmentation(pred_mask, sample['mask'])
            metrics['sample_id'] = sample['id']
            all_metrics.append(metrics)
        
        # Aggregate
        if all_metrics:
            agg = {
                'mean_dice': np.mean([m['Dice'] for m in all_metrics]),
                'mean_pq_0.5': np.mean([m['PQ@0.5'] for m in all_metrics]),
                'mean_pq_0.3': np.mean([m['PQ@0.3'] for m in all_metrics]),
                'mean_aji': np.mean([m['AJI'] for m in all_metrics]),
                'mean_boundary_iou': np.mean([m['Boundary_IoU'] for m in all_metrics]),
                'mean_hd95': np.mean([m.get('HD95', 0) for m in all_metrics if m.get('HD95') is not None and m.get('HD95') != float('inf')]),
                'n_samples': len(all_metrics),
            }
            results['models'][model_name] = agg
            
            print(f"\n    Results for {model_name}:")
            print(f"      Dice:        {agg['mean_dice']:.4f}")
            print(f"      PQ@0.5:      {agg['mean_pq_0.5']:.4f}")
            print(f"      PQ@0.3:      {agg['mean_pq_0.3']:.4f}")
            print(f"      AJI:         {agg['mean_aji']:.4f}")
            print(f"      Boundary IoU:{agg['mean_boundary_iou']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    print(f"{'Model':<20} {'Dice':>10} {'PQ@0.5':>10} {'AJI':>10} {'Boundary':>10}")
    print("-" * 60)
    for name, data in results['models'].items():
        print(f"{name:<20} {data['mean_dice']:>10.4f} {data['mean_pq_0.5']:>10.4f} {data['mean_aji']:>10.4f} {data['mean_boundary_iou']:>10.4f}")
    
    # Save
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
