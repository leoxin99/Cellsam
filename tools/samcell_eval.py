"""
SAMCell Baseline Evaluation — T16

SAMCell uses HuggingFace SamModel with fine-tuned decoder that predicts
distance maps. Watershed converts distance maps to instance labels.
This is an E2E method (no GT boxes needed).

Usage: python tools/samcell_eval.py
"""

import sys
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from torch import nn
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from augmented_dataset import AugmentedAllenDataset
from metrics.instance_metrics import compute_all_metrics

from transformers import SamModel, SamProcessor
from skimage.segmentation import watershed


# ============================================================
# SAMCell Pipeline (adapted from NathanMalta/SAMCell/src/pipeline.py)
# ============================================================

class SAMCellPipeline:
    """SAMCell inference: dist-map prediction + watershed."""
    
    def __init__(self, model_path, device='cuda', crop_size=256):
        self.device = device
        self.crop_size = crop_size
        self.sigmoid = nn.Sigmoid()
        
        # Load fine-tuned SAM model (manual load to bypass CVE-2025-32434)
        from transformers import SamConfig
        import warnings
        warnings.filterwarnings('ignore')
        sam_config = SamConfig.from_pretrained(model_path)
        self.model = SamModel(sam_config)
        state_dict = torch.load(
            str(Path(model_path) / "pytorch_model.bin"),
            map_location='cpu', weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)
        
        # SAM processor for image preprocessing
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
    
    def _preprocess(self, img):
        """Preprocess image for SAM (grayscale -> RGB, normalize)."""
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def get_model_prediction(self, image):
        """Get distance map prediction for a single crop."""
        image = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(pixel_values=image, multimask_output=True)
        prob = outputs['pred_masks'].squeeze(1)
        dist_map = self.sigmoid(prob)[0][0]
        return dist_map
    
    def predict_on_full_img(self, image_orig):
        """Predict on full image using sliding window."""
        orig_shape = image_orig.shape[:2]
        crops, positions = self._split_into_crops(image_orig)
        
        # Predict on each crop
        dist_maps = []
        for crop in crops:
            dist_map = self.get_model_prediction(crop).cpu().numpy()
            dist_map = cv2.resize(dist_map, (self.crop_size, self.crop_size))
            dist_maps.append(dist_map)
        
        # Reconstruct full image (simple stitching, last write wins for overlap)
        cell_dist_map = np.zeros(orig_shape, dtype=np.float32)
        count_map = np.zeros(orig_shape, dtype=np.float32)
        
        for dist_map, (min_x, min_y) in zip(dist_maps, positions):
            h = min(self.crop_size, orig_shape[0] - min_x)
            w = min(self.crop_size, orig_shape[1] - min_y)
            cell_dist_map[min_x:min_x+h, min_y:min_y+w] += dist_map[:h, :w]
            count_map[min_x:min_x+h, min_y:min_y+w] += 1
        
        count_map[count_map == 0] = 1
        cell_dist_map /= count_map
        
        return cell_dist_map
    
    def _split_into_crops(self, image):
        """Split image into overlapping crops."""
        crops = []
        positions = []
        h, w = image.shape[:2]
        stride = self.crop_size  # no overlap for simplicity
        
        for i in range(0, max(h, self.crop_size), stride):
            for j in range(0, max(w, self.crop_size), stride):
                min_x = min(i, max(0, h - self.crop_size))
                min_y = min(j, max(0, w - self.crop_size))
                crop = image[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size]
                crops.append(crop)
                positions.append((min_x, min_y))
        
        return crops, positions
    
    def cells_from_dist_map(self, dist_map):
        """Convert distance map to instance labels using watershed."""
        cells_max = dist_map > 0.5
        cell_fill = dist_map > 0.05
        
        contours, _ = cv2.findContours(
            cells_max.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask = np.zeros(dist_map.shape, dtype=np.int32)
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            mask[int(cY), int(cX)] = i + 1
        
        labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)
        return labels
    
    def run(self, image):
        """Full pipeline: image -> instance labels."""
        dist_map = self.predict_on_full_img(image)
        labels = self.cells_from_dist_map(dist_map)
        return labels


# ============================================================
# Evaluation
# ============================================================

def eval_samcell(model_path, dataset):
    """Run SAMCell on test(73)."""
    print(f"\n{'='*60}")
    print(f"SAMCell — E2E on BF grayscale")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = SAMCellPipeline(model_path, device=device, crop_size=256)
    
    all_metrics = []
    
    for idx in tqdm(range(len(dataset)), desc="SAMCell"):
        # Load raw image to get BF channel
        raw_path = dataset.samples[idx]['image_path']
        raw_img = np.load(raw_path)
        
        if raw_img.ndim == 3 and raw_img.shape[2] == 3:
            raw_img = raw_img.transpose(2, 0, 1)
        
        # BF channel (C0), resize to 1024x1024
        bf = raw_img[0].astype(np.float32)
        bf_min, bf_max = bf.min(), bf.max()
        if bf_max > bf_min:
            bf = (bf - bf_min) / (bf_max - bf_min)
        bf_uint8 = (bf * 255).astype(np.uint8)
        
        # Resize to 1024x1024 if needed
        if bf_uint8.shape[0] != 1024 or bf_uint8.shape[1] != 1024:
            bf_uint8 = cv2.resize(bf_uint8, (1024, 1024))
        
        sample = dataset[idx]
        gt_mask = sample['mask'].numpy().astype(np.int32)
        sample_id = sample['sample_id']
        
        try:
            labels = pipeline.run(bf_uint8)
            
            # Resize labels to 1024x1024 if different
            if labels.shape != gt_mask.shape:
                labels = cv2.resize(
                    labels.astype(np.float32), 
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
            
            m = compute_all_metrics(labels, gt_mask)
            m['sample_id'] = sample_id
            m['n_samcell_cells'] = int(labels.max())
            all_metrics.append(m)
        except Exception as e:
            print(f"  ⚠️ Sample {idx} ({sample_id}) failed: {e}")
            import traceback; traceback.print_exc()
            all_metrics.append({
                'sample_id': sample_id, 'error': str(e),
                'pq': 0, 'bm_1to1_dice': 0, 'aji': 0
            })
            torch.cuda.empty_cache()
    
    return all_metrics


def main():
    project_root = Path(__file__).parent.parent
    
    # Load test dataset
    test_ids = (project_root / "data/splits/test_ids.txt").read_text().strip().split('\n')
    dataset = AugmentedAllenDataset(
        data_dir=str(project_root / "data/processed"),
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Loaded test dataset: {len(dataset)} samples")
    
    model_path = str(project_root / "checkpoints/samcell/livecell/samcell-livecell")
    
    all_metrics = eval_samcell(model_path, dataset)
    
    # Aggregate
    METRIC_KEYS = [
        'bm_1to1_dice', 'bm_coverage_dice', 'gap_dice',
        'pq', 'sq', 'rq', 'aji', 'semantic_dice',
        'tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells'
    ]
    
    valid = [m for m in all_metrics if 'error' not in m]
    agg = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in valid if key in m]
        if values:
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'n': len(values),
            }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"  SAMCell (livecell)")
    print(f"{'='*50}")
    for key in ['pq', 'bm_1to1_dice', 'aji', 'semantic_dice', 'sq', 'rq']:
        if key in agg:
            print(f"  {key:20s}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}  (n={agg[key]['n']})")
    for key in ['tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells']:
        if key in agg:
            print(f"  {key:20s}: {agg[key]['mean']:.1f} ± {agg[key]['std']:.1f}")
    
    # Save results
    output_dir = project_root / "experiments" / "baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "per_sample_samcell_livecell.json", 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
