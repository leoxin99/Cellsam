# [DEPRECATED] This script has been archived.
#
# Archived: 2026-02-10
# Reason: Superseded by unified inference core (Phase 0)
# Replacement entry points:
#   - Training:           src/train.py
#   - Oracle evaluation:  tools/standardized_inference.py
#   - E2E evaluation:     tools/evaluate_e2e.py
#   - Multi-model eval:   tools/comprehensive_eval.py
#   - Regression test:    tools/test_phase0_regression.py
#
import warnings as _warnings
_warnings.warn(
    "This script is deprecated. See header for replacement entry points.",
    DeprecationWarning, stacklevel=2
)
"""
Evaluation script for E24-E28 models with PQ and Actn2 masking.
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

# Data-driven Actn2 threshold from analyze_data_params.py
ACTN2_THRESHOLD = 0.007  # P25 mean from data analysis

# E24-E28 Checkpoints
CHECKPOINTS = {
    "E25_Boundary_Enhanced": {
        "path": "checkpoints/boundary_enhanced_best.pt",
        "adapter": False,
        "semantic": False,
    },
    "E27_3ch_Semantic_Adapter": {
        "path": "checkpoints/3ch_semantic_adapter_best.pt",
        "adapter": True,
        "semantic": True,
    },
    "E28_BF_Adapter": {
        "path": "checkpoints/bf_adapter_best.pt",
        "adapter": True,
        "semantic": False,
    },
}

def load_model_with_config(config, device):
    """Load model with optional adapter."""
    model = get_model()
    
    checkpoint = torch.load(config["path"], map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    adapter = None
    if config["adapter"]:
        adapter = IndependentChannelAdapter(kernel_size=3, use_relu=True)
        if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
        adapter = adapter.to(device)
        adapter.eval()
    
    return model, adapter


def segment_image(model, image, boxes, device, adapter=None, semantic=False, 
                  use_actn2_mask=False, actn2_threshold=ACTN2_THRESHOLD):
    """Segment image using model with optional Actn2 masking."""
    # Prepare Actn2 mask
    actn2_region_mask = None
    if use_actn2_mask:
        actn2_channel = image[2].astype(np.float32)
        actn2_normalized = (actn2_channel - actn2_channel.min()) / (actn2_channel.max() - actn2_channel.min() + 1e-8)
        actn2_region_mask = actn2_normalized > actn2_threshold
    
    # Prepare input
    if semantic and adapter is not None:
        img_semantic = np.stack([image[2], image[0], image[1]], axis=0)
        img_tensor = torch.from_numpy(img_semantic).float().unsqueeze(0).to(device)
        img_tensor = adapter(img_tensor)
    else:
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
            
            pred = F.interpolate(
                low_res_masks, size=(1024, 1024),
                mode='bilinear', align_corners=False
            ).squeeze()
            
            mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            
            # Box clipping
            x1, y1, x2, y2 = [int(b) for b in box]
            h, w = mask.shape
            bw, bh = x2 - x1, y2 - y1
            expand = 0.1
            x1_clip = max(0, int(x1 - bw * expand))
            y1_clip = max(0, int(y1 - bh * expand))
            x2_clip = min(w, int(x2 + bw * expand))
            y2_clip = min(h, int(y2 + bh * expand))
            
            mask_clipped = np.zeros_like(mask)
            mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]
            
            # Apply Actn2 mask
            if use_actn2_mask and actn2_region_mask is not None:
                mask_clipped = mask_clipped & actn2_region_mask
            
            pred_mask[mask_clipped] = i + 1
            
        except Exception as e:
            continue
    
    return pred_mask


def run_evaluation():
    """Run evaluation on E24-E28 models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Actn2 Threshold (data-driven): {ACTN2_THRESHOLD}")
    
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
        
        # Evaluate with and without Actn2 mask
        for use_actn2 in [False, True]:
            tag = f"{model_name}{'_Actn2Mask' if use_actn2 else ''}"
            all_metrics = []
            
            for idx in tqdm(range(len(dataset)), desc=tag):
                sample = dataset[idx]
                image = sample['image'].numpy()
                gt_mask = sample['mask'].numpy()
                boxes = sample['boxes'].numpy()
                
                pred_mask = segment_image(
                    model, image, boxes, device, adapter,
                    semantic=config["semantic"],
                    use_actn2_mask=use_actn2
                )
                
                metrics = evaluate_instance_segmentation(pred_mask, gt_mask)
                all_metrics.append(metrics)
            
            # Aggregate
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            
            results[tag] = avg_metrics
            
            print(f"\n{tag}:")
            print(f"  PQ@0.5: {avg_metrics.get('pq_0.5', 0):.4f}")
            print(f"  PQ@0.3: {avg_metrics.get('pq_0.3', 0):.4f}")
            print(f"  AJI: {avg_metrics.get('aji', 0):.4f}")
            print(f"  Dice: {avg_metrics.get('dice', 0):.4f}")
    
    # Save results
    output_file = f"results/e24_e28_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_evaluation()
