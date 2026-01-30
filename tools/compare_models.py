"""
Semantic Adapter vs E12 Baseline 模型对比测试 (修正版)

使用正确的 SAM 推理流程：
1. sam_preprocess() 预处理图像
2. image_encoder() 编码特征
3. prompt_encoder() + mask_decoder() 解码

运行:
    conda activate cellsam
    python tools/compare_models.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Setup environment
os.environ["DEEPCELL_ACCESS_TOKEN"] = "X2Od0tJX.te0hEWOzZlRXoJzh5pkvw7l4S5GdpPxs"

# Add project paths
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "cellSAM_source"))
sys.path.insert(0, str(project_dir / "src"))

from cellSAM import get_model
from adapters import IndependentChannelAdapter
from augmented_dataset import AugmentedAllenDataset, load_split_ids


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    
    if union == 0:
        return 1.0 if np.sum(gt) == 0 else 0.0
    
    return 2.0 * intersection / union


def load_model_and_adapter(checkpoint_path: Path, device: torch.device):
    """Load CellSAM model and adapter from checkpoint."""
    # Load base model
    model = get_model()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # Load model state
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load adapter if present
    adapter = None
    if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
        adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
        adapter = adapter.to(device)
        adapter.eval()
        print(f"    Loaded adapter ({adapter.get_param_count()} params)")
    
    return model, adapter


def evaluate_model(model, dataset, device, adapter=None, model_name: str = "Model"):
    """Evaluate model on dataset using per-cell Dice calculation."""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
            sample = dataset[idx]
            
            image = sample['image'].unsqueeze(0).to(device)  # (1, 3, 1024, 1024)
            boxes = sample['boxes'].to(device)
            cell_ids = sample['cell_ids'].to(device)
            gt_mask = sample['mask'].to(device)
            num_boxes = sample['num_boxes']
            
            if num_boxes == 0:
                continue
            
            # Apply adapter if available
            if adapter is not None:
                image = adapter(image)
            
            # Preprocess and encode
            img_preprocessed = model.sam_preprocess(image)
            embedding = model.model.image_encoder(img_preprocessed)
            
            # Evaluate each cell (max 20 boxes)
            for box_idx in range(min(num_boxes, 20)):
                box = boxes[box_idx:box_idx+1].unsqueeze(0)  # (1, 1, 4)
                cell_id = cell_ids[box_idx].item()
                
                # Encode prompt
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box, masks=None
                )
                
                # Decode mask
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Upsample to original size
                pred_mask = F.interpolate(
                    low_res_masks, 
                    size=(gt_mask.shape[0], gt_mask.shape[1]),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                # Get GT for this cell
                gt_cell_mask = (gt_mask == cell_id).float()
                
                # Calculate Dice
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                intersection = (pred_binary * gt_cell_mask).sum()
                dice = (2 * intersection) / (pred_binary.sum() + gt_cell_mask.sum() + 1e-8)
                dice_scores.append(dice.item())
    
    return {
        'model': model_name,
        'mean_dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'std_dice': float(np.std(dice_scores)) if dice_scores else 0.0,
        'min_dice': float(np.min(dice_scores)) if dice_scores else 0.0,
        'max_dice': float(np.max(dice_scores)) if dice_scores else 0.0,
        'n_cells': len(dice_scores),
        'scores': [float(s) for s in dice_scores]
    }


def main():
    print("=" * 60)
    print("Semantic Adapter vs E12 Baseline 模型对比")
    print("=" * 60)
    
    # Paths
    data_dir = project_dir / "data" / "processed"
    splits_dir = project_dir / "data" / "splits"
    output_dir = project_dir / "experiments" / "model_comparison"
    
    e12_checkpoint = project_dir / "checkpoints" / "boundary_20260111_012636" / "best_model.pt"
    semantic_checkpoint = project_dir / "checkpoints" / "semantic_adapter_20260130_033141" / "best_model.pt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Check checkpoints
    print("\n[1] Checking checkpoints...")
    if not e12_checkpoint.exists():
        print(f"    ❌ E12 checkpoint not found: {e12_checkpoint}")
        return
    if not semantic_checkpoint.exists():
        print(f"    ❌ Semantic checkpoint not found: {semantic_checkpoint}")
        return
    print(f"    ✅ E12: {e12_checkpoint.parent.name}")
    print(f"    ✅ Semantic: {semantic_checkpoint.parent.name}")
    
    # Load test IDs
    print("\n[2] Loading test dataset...")
    test_ids = load_split_ids(split="test", splits_dir=str(splits_dir))[:10]  # Use first 10
    print(f"    Test samples: {len(test_ids)}")
    
    # Create test dataset
    test_dataset = AugmentedAllenDataset(
        data_dir=str(data_dir),
        is_training=False,
        sample_ids=test_ids,
        use_bf_only=True  # E12 uses BF×3
    )
    print(f"    Dataset created: {len(test_dataset)} samples")
    
    # Evaluate E12 Baseline
    print("\n[3] Evaluating E12 Baseline (BF×3)...")
    e12_model, _ = load_model_and_adapter(e12_checkpoint, device)
    e12_results = evaluate_model(e12_model, test_dataset, device, None, 'E12_Baseline')
    print(f"    Mean Dice: {e12_results['mean_dice']:.4f} ({e12_results['n_cells']} cells)")
    del e12_model
    torch.cuda.empty_cache()
    
    # Create dataset for Semantic Adapter (3-channel with semantic mapping)
    semantic_dataset = AugmentedAllenDataset(
        data_dir=str(data_dir),
        is_training=False,
        sample_ids=test_ids,
        use_bf_only=False,
        use_semantic_mapping=True
    )
    
    # Evaluate Semantic Adapter
    print("\n[4] Evaluating Semantic Adapter (3通道 + Adapter)...")
    semantic_model, adapter = load_model_and_adapter(semantic_checkpoint, device)
    semantic_results = evaluate_model(semantic_model, semantic_dataset, device, adapter, 'Semantic_Adapter')
    print(f"    Mean Dice: {semantic_results['mean_dice']:.4f} ({semantic_results['n_cells']} cells)")
    del semantic_model, adapter
    torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"{'Model':<20} {'Mean Dice':>12} {'Std':>10} {'Cells':>10}")
    print("-" * 52)
    print(f"{'E12 Baseline':<20} {e12_results['mean_dice']:>12.4f} {e12_results['std_dice']:>10.4f} {e12_results['n_cells']:>10}")
    print(f"{'Semantic Adapter':<20} {semantic_results['mean_dice']:>12.4f} {semantic_results['std_dice']:>10.4f} {semantic_results['n_cells']:>10}")
    print("-" * 52)
    
    delta = semantic_results['mean_dice'] - e12_results['mean_dice']
    winner = "Semantic Adapter" if delta > 0 else "E12 Baseline"
    print(f"\n{'Winner:':<20} {winner} (Dice 差异: {delta:+.4f})")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'n_samples': len(test_ids),
        'e12_baseline': e12_results,
        'semantic_adapter': semantic_results,
        'delta_dice': delta,
        'winner': winner
    }
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
