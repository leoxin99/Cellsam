"""
边界精度分析脚本

分析为什么修复后 PQ@0.3=0.18 但 PQ@0.5=0.01
大部分预测 IoU 在 0.3-0.5 之间，存在系统性边界偏差

分析内容：
1. 每个预测的 IoU 分布
2. 边界偏差方向（过大/过小）
3. 不同类型细胞的边界精度差异
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent / "cellSAM_source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset


def compute_instance_iou(pred_mask, gt_mask):
    """Compute best IoU match for each GT instance."""
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels > 0]
    
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels > 0]
    
    results = []
    
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        gt_area = gt_region.sum()
        
        best_iou = 0
        best_pred_label = -1
        is_overseg = False
        is_underseg = False
        
        for pred_label in pred_labels:
            pred_region = (pred_mask == pred_label)
            intersection = (gt_region & pred_region).sum()
            union = (gt_region | pred_region).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_label = pred_label
                    pred_area = pred_region.sum()
                    is_overseg = pred_area > gt_area * 1.1
                    is_underseg = pred_area < gt_area * 0.9
        
        # Get GT bbox
        gt_props = measure.regionprops(gt_region.astype(int))
        if gt_props:
            y1, x1, y2, x2 = gt_props[0].bbox
            gt_bbox_area = (x2 - x1) * (y2 - y1)
        else:
            gt_bbox_area = gt_area
        
        results.append({
            'gt_label': int(gt_label),
            'best_pred_label': int(best_pred_label),
            'iou': float(best_iou),
            'gt_area': int(gt_area),
            'is_overseg': is_overseg,
            'is_underseg': is_underseg,
            'matched': best_iou >= 0.5,
            'partial_match': 0.3 <= best_iou < 0.5,
            'no_match': best_iou < 0.3,
        })
    
    return results


def analyze_boundary_precision():
    """Main analysis function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = get_model()
    checkpoint = torch.load("checkpoints/bf_baseline_full_best.pt", map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_ids = Path("data/splits/test_ids.txt").read_text().strip().split('\n')[:30]
    dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        is_training=False,
        sample_ids=test_ids
    )
    print(f"Analyzing {len(dataset)} samples...")
    
    all_ious = []
    all_results = []
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        
        image = sample['image'].numpy()
        gt_mask = sample['mask'].numpy().astype(np.int32)
        boxes = sample['boxes'].numpy()
        num_boxes = sample['num_boxes']
        
        # Prepare BF-only input
        img_bf = np.stack([image[0], image[0], image[0]], axis=0)
        img_tensor = torch.from_numpy(img_bf).float().unsqueeze(0).to(device)
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_preprocessed = model.sam_preprocess(img_tensor)
        
        with torch.no_grad():
            image_embedding = model.model.image_encoder(img_preprocessed)
        
        # Segment with box clipping
        pred_mask = np.zeros((1024, 1024), dtype=np.int32)
        
        for i in range(num_boxes):
            box = boxes[i]
            if box.sum() == 0:
                continue
            
            box_tensor = torch.tensor([box.tolist()], dtype=torch.float32).unsqueeze(0).to(device)
            
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
            expand = 0.1  # Reduced from 0.2
            x1_clip = max(0, int(x1 - bw * expand))
            y1_clip = max(0, int(y1 - bh * expand))
            x2_clip = min(w, int(x2 + bw * expand))
            y2_clip = min(h, int(y2 + bh * expand))
            
            mask_clipped = np.zeros_like(mask)
            mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]
            
            pred_mask[mask_clipped > 0] = i + 1
        
        # Analyze IoU distribution
        instance_results = compute_instance_iou(pred_mask, gt_mask)
        all_results.extend(instance_results)
        all_ious.extend([r['iou'] for r in instance_results])
    
    # Analysis
    print("\n" + "="*70)
    print("边界精度分析结果")
    print("="*70)
    
    ious = np.array(all_ious)
    print(f"\n【IoU 分布统计】")
    print(f"  总实例数: {len(ious)}")
    print(f"  IoU 均值: {np.mean(ious):.4f}")
    print(f"  IoU 中位数: {np.median(ious):.4f}")
    print(f"  IoU 标准差: {np.std(ious):.4f}")
    
    # IoU bands
    bands = [
        (0.0, 0.1, "无匹配"),
        (0.1, 0.3, "差匹配"),
        (0.3, 0.5, "部分匹配"),
        (0.5, 0.7, "良好匹配"),
        (0.7, 1.0, "优秀匹配"),
    ]
    
    print(f"\n【IoU 分段分布】")
    for lo, hi, label in bands:
        count = np.sum((ious >= lo) & (ious < hi))
        pct = count / len(ious) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:8} [{lo:.1f}-{hi:.1f}): {count:3d} ({pct:5.1f}%) {bar}")
    
    # Over/under segmentation
    overseg = sum(1 for r in all_results if r['is_overseg'])
    underseg = sum(1 for r in all_results if r['is_underseg'])
    print(f"\n【分割偏差分析】")
    print(f"  过分割 (pred > GT*1.1): {overseg} ({overseg/len(all_results)*100:.1f}%)")
    print(f"  欠分割 (pred < GT*0.9): {underseg} ({underseg/len(all_results)*100:.1f}%)")
    print(f"  正常范围: {len(all_results) - overseg - underseg} ({(len(all_results)-overseg-underseg)/len(all_results)*100:.1f}%)")
    
    # Match statistics
    matched = sum(1 for r in all_results if r['matched'])
    partial = sum(1 for r in all_results if r['partial_match'])
    no_match = sum(1 for r in all_results if r['no_match'])
    print(f"\n【匹配统计】")
    print(f"  IoU ≥ 0.5 (良好): {matched} ({matched/len(all_results)*100:.1f}%)")
    print(f"  0.3 ≤ IoU < 0.5 (部分): {partial} ({partial/len(all_results)*100:.1f}%)")
    print(f"  IoU < 0.3 (失败): {no_match} ({no_match/len(all_results)*100:.1f}%)")
    
    # Key insight
    print(f"\n【关键发现】")
    if partial > matched:
        print(f"  ⚠️ 大部分预测 (部分{partial} > 良好{matched}) 处于 IoU 0.3-0.5")
        print(f"  → 边界存在系统性偏差，约 10-30% 面积误差")
        print(f"  → 可能原因: (1) BF边界模糊 (2) 训练数据边界噪声 (3) Box扩展过大")
    
    # Save results
    output_dir = Path("experiments/boundary_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "iou_distribution.json", 'w') as f:
        json.dump({
            'iou_stats': {
                'mean': float(np.mean(ious)),
                'median': float(np.median(ious)),
                'std': float(np.std(ious)),
            },
            'distribution': {
                f'{lo:.1f}-{hi:.1f}': int(np.sum((ious >= lo) & (ious < hi)))
                for lo, hi, _ in bands
            },
            'segmentation_bias': {
                'overseg': overseg,
                'underseg': underseg,
                'normal': len(all_results) - overseg - underseg,
            },
            'match_stats': {
                'matched_0.5': matched,
                'partial_0.3-0.5': partial,
                'no_match': no_match,
            }
        }, f, indent=2)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.3, color='orange', linestyle='--', label='IoU=0.3')
    plt.axvline(x=0.5, color='red', linestyle='--', label='IoU=0.5')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title('Instance IoU Distribution')
    plt.legend()
    plt.savefig(output_dir / "iou_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n结果保存至: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    analyze_boundary_precision()
