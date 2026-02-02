"""
全面评估脚本：BF Baseline vs Semantic Adapter
使用完整评估指标 (PQ, AJI, Boundary IoU, HD95)

问题分析：
1. 训练验证只用了 Val Dice - 仅像素级指标
2. 需要实例级指标 (PQ, AJI) 来评估分割质量
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from adapters.channel_adapter import IndependentChannelAdapter
from augmented_dataset import AugmentedAllenDataset

# Import comprehensive metrics
sys.path.insert(0, str(Path(__file__).parent.parent / "anti_test"))
from eval_metrics import evaluate_instance_segmentation

# Checkpoints
CHECKPOINTS = {
    "BF_Baseline_Full": {
        "path": "checkpoints/bf_baseline_full_best.pt",
        "adapter": False,
        "semantic": False,
    },
    "Semantic_Adapter": {
        "path": "checkpoints/semantic_adapter_v2_best.pt", 
        "adapter": True,
        "semantic": True,
    },
}

def load_model_with_config(config, device):
    """Load model with optional adapter."""
    model = get_model()
    
    checkpoint = torch.load(config["path"], map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    adapter = None
    if config["adapter"]:
        adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
        # Try to load adapter weights from checkpoint
        if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
        adapter = adapter.to(device)
        adapter.eval()
    
    return model, adapter


def segment_image(model, image, boxes, device, adapter=None, semantic=False):
    """Segment image using model."""
    # Prepare input
    if semantic and adapter is not None:
        # Use semantic channel mapping: [Actn2, BF, DAPI]
        img_semantic = np.stack([image[2], image[0], image[1]], axis=0)
        img_tensor = torch.from_numpy(img_semantic).float().unsqueeze(0).to(device)
        img_tensor = adapter(img_tensor)
    else:
        # BF only (copy 3 times)
        img_bf = np.stack([image[0], image[0], image[0]], axis=0)
        img_tensor = torch.from_numpy(img_bf).float().unsqueeze(0).to(device)
    
    # Normalize
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    if img_max > img_min:
        img_tensor = (img_tensor - img_min) / (img_max - img_min)
    
    # SAM preprocess
    img_preprocessed = model.sam_preprocess(img_tensor)
    
    # Get embedding
    with torch.no_grad():
        image_embedding = model.model.image_encoder(img_preprocessed)
    
    # Segment each box
    pred_mask = np.zeros((1024, 1024), dtype=np.int32)
    
    for i, box in enumerate(boxes):
        try:
            box_tensor = torch.tensor([box], dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sparse_emb, dense_emb = model.model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                low_res_masks, _ = model.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
            
            # Resize and threshold
            pred = F.interpolate(
                low_res_masks, size=(1024, 1024),
                mode='bilinear', align_corners=False
            ).squeeze()
            
            mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            
            # === FIX: Clip mask to box region (matching training loss) ===
            x1, y1, x2, y2 = [int(b) for b in box]
            h, w = mask.shape
            bw, bh = x2 - x1, y2 - y1
            expand = 0.1  # 10% expansion (reduced from 20% to fix over-segmentation)
            x1_clip = max(0, int(x1 - bw * expand))
            y1_clip = max(0, int(y1 - bh * expand))
            x2_clip = min(w, int(x2 + bw * expand))
            y2_clip = min(h, int(y2 + bh * expand))
            
            # Create clipped mask
            mask_clipped = np.zeros_like(mask)
            mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]
            
            pred_mask[mask_clipped] = i + 1
            
        except Exception as e:
            continue
    
    return pred_mask


def run_comprehensive_evaluation():
    """Run evaluation with all metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Test samples: {len(dataset)}")
    
    results = {}
    
    for model_name, config in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        if not Path(config["path"]).exists():
            print(f"  ⚠️ Checkpoint not found: {config['path']}")
            continue
        
        model, adapter = load_model_with_config(config, device)
        
        all_metrics = []
        
        for idx in tqdm(range(len(dataset)), desc=model_name):
            sample = dataset[idx]
            
            # Get image and GT
            image = sample['image'].numpy()  # (3, H, W)
            gt_mask = sample['mask'].numpy()  # (H, W)
            boxes = sample['boxes'].numpy()
            num_boxes = sample['num_boxes']
            
            valid_boxes = [boxes[i] for i in range(num_boxes) if boxes[i].sum() > 0]
            
            if len(valid_boxes) == 0:
                continue
            
            # Segment
            pred_mask = segment_image(
                model, image, valid_boxes, device,
                adapter=adapter, semantic=config["semantic"]
            )
            
            # Evaluate
            metrics = evaluate_instance_segmentation(pred_mask, gt_mask)
            all_metrics.append(metrics)
        
        # Aggregate results
        if all_metrics:
            aggregated = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if not np.isinf(m[key]) and not np.isnan(m[key])]
                if values:
                    aggregated[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }
            
            results[model_name] = aggregated
            
            print(f"\nResults for {model_name}:")
            print(f"  Dice:        {aggregated.get('Dice', {}).get('mean', 0):.4f} ± {aggregated.get('Dice', {}).get('std', 0):.4f}")
            print(f"  PQ@0.5:      {aggregated.get('PQ@0.5', {}).get('mean', 0):.4f} ± {aggregated.get('PQ@0.5', {}).get('std', 0):.4f}")
            print(f"  PQ@0.3:      {aggregated.get('PQ@0.3', {}).get('mean', 0):.4f} ± {aggregated.get('PQ@0.3', {}).get('std', 0):.4f}")
            print(f"  AJI:         {aggregated.get('AJI', {}).get('mean', 0):.4f} ± {aggregated.get('AJI', {}).get('std', 0):.4f}")
            print(f"  Boundary IoU:{aggregated.get('Boundary_IoU', {}).get('mean', 0):.4f} ± {aggregated.get('Boundary_IoU', {}).get('std', 0):.4f}")
    
    # Save results
    output_dir = Path("experiments/comprehensive_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'results.json'}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()
