"""
Box Clipping 影响实验

验证假设: 使用 GT 框时，box clipping 是冗余的
因为 SAM 模型设计上只在 box 区域内预测 mask

实验方法:
1. 加载模型和数据
2. 对每个 box 预测 mask
3. 比较 clipped 和 unclipped 预测
4. 统计 box 外的预测值分布
"""

import sys
sys.path.insert(0, 'cellSAM_source')
sys.path.insert(0, 'src')

import torch
import numpy as np
from torch.utils.data import DataLoader
from cellSAM import get_model
from augmented_dataset import AugmentedAllenDataset, collate_fn, load_split_ids


def run_clipping_experiment():
    print("=" * 60)
    print("Box Clipping 影响实验")
    print("=" * 60)
    
    # 加载数据
    val_ids = load_split_ids("val", "data/splits")[:5]  # 5张图像
    val_dataset = AugmentedAllenDataset(
        data_dir="data/processed",
        target_size=(1024, 1024),
        is_training=False,
        max_boxes_per_image=30,
        sample_ids=val_ids,
        use_bf_only=True,
        use_semantic_mapping=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 加载模型
    print("\n加载模型...")
    model = get_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 统计变量
    total_boxes = 0
    total_outside_pixels = 0
    total_outside_nonzero = 0
    max_outside_value = 0
    outside_values = []
    dice_with_clip = []
    dice_without_clip = []
    
    print("\n开始实验...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['boxes']
            cell_ids = batch.get('cell_ids', None)
            
            # SAM preprocessing
            img_preprocessed = model.sam_preprocess(images)
            image_embedding = model.model.image_encoder(img_preprocessed)
            
            for i in range(images.shape[0]):
                sample_boxes = boxes[i]
                sample_mask = masks[i]
                sample_cell_ids = cell_ids[i] if cell_ids is not None else None
                
                for j, box in enumerate(sample_boxes):
                    if box.sum() == 0:
                        continue
                    
                    total_boxes += 1
                    
                    # Get cell ID
                    if sample_cell_ids is not None and j < len(sample_cell_ids):
                        cell_id = sample_cell_ids[j].item() if hasattr(sample_cell_ids[j], 'item') else sample_cell_ids[j]
                    else:
                        continue
                    
                    if cell_id is None or cell_id == 0:
                        continue
                    
                    # Predict
                    box_tensor = box.unsqueeze(0).to(device)
                    sparse_embeddings, dense_embeddings = model.model.prompt_encoder(
                        points=None, boxes=box_tensor, masks=None
                    )
                    
                    low_res_masks, iou_pred = model.model.mask_decoder(
                        image_embeddings=image_embedding[i:i+1],
                        image_pe=model.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    
                    # Upscale and sigmoid
                    upscaled_masks = model.model.postprocess_masks(
                        low_res_masks,
                        input_size=(1024, 1024),
                        original_size=(1024, 1024)
                    )
                    pred_sigmoid = torch.sigmoid(upscaled_masks[0, 0])
                    
                    # Box coordinates
                    x1, y1, x2, y2 = [int(c.item()) for c in box]
                    h, w = pred_sigmoid.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # 分析 box 外的预测值
                    box_mask = torch.zeros_like(pred_sigmoid, dtype=torch.bool)
                    box_mask[y1:y2, x1:x2] = True
                    outside_mask = ~box_mask
                    
                    outside_pred = pred_sigmoid[outside_mask]
                    total_outside_pixels += outside_pred.numel()
                    
                    # 统计 box 外非零值
                    nonzero_outside = (outside_pred > 0.01).sum().item()
                    above_threshold = (outside_pred > 0.5).sum().item()
                    max_val = outside_pred.max().item() if outside_pred.numel() > 0 else 0
                    
                    total_outside_nonzero += nonzero_outside
                    max_outside_value = max(max_outside_value, max_val)
                    
                    if max_val > 0.01:
                        outside_values.append(max_val)
                    
                    # 计算 Dice (with and without clipping)
                    target = (sample_mask == cell_id).float()
                    
                    # With clipping
                    pred_clipped = torch.zeros_like(pred_sigmoid)
                    pred_clipped[y1:y2, x1:x2] = pred_sigmoid[y1:y2, x1:x2]
                    pred_bin_clipped = (pred_clipped > 0.5).float()
                    
                    # Without clipping
                    pred_bin_full = (pred_sigmoid > 0.5).float()
                    
                    # Dice
                    def dice(pred, target):
                        intersection = (pred * target).sum()
                        return (2 * intersection / (pred.sum() + target.sum() + 1e-8)).item()
                    
                    d_clip = dice(pred_bin_clipped, target)
                    d_full = dice(pred_bin_full, target)
                    
                    dice_with_clip.append(d_clip)
                    dice_without_clip.append(d_full)
                    
                    if abs(d_clip - d_full) > 0.01:
                        print(f"  样本 {batch_idx}, Box {j}: Dice差异 = {d_full - d_clip:.4f}")
            
            print(f"处理完成: 图像 {batch_idx + 1}/{len(val_loader)}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n总 box 数量: {total_boxes}")
    print(f"总 box 外像素: {total_outside_pixels:,}")
    print(f"Box 外 > 0.01 的像素数: {total_outside_nonzero:,}")
    print(f"Box 外最大预测值: {max_outside_value:.4f}")
    
    print(f"\n平均 Dice (有 clipping): {np.mean(dice_with_clip):.4f}")
    print(f"平均 Dice (无 clipping): {np.mean(dice_without_clip):.4f}")
    print(f"Dice 差异: {np.mean(dice_without_clip) - np.mean(dice_with_clip):.6f}")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    if max_outside_value < 0.5 and abs(np.mean(dice_without_clip) - np.mean(dice_with_clip)) < 0.001:
        print("✅ Box clipping 在 GT 框场景下确实是冗余的")
        print("   - Box 外预测值均低于 0.5 阈值")
        print("   - Dice 计算结果几乎无差异")
        print("   - 可以安全移除训练时的 box clipping")
    else:
        print("⚠️ Box clipping 有一定影响")
        print(f"   - Box 外最大值: {max_outside_value:.4f}")
        print(f"   - Dice 差异: {np.mean(dice_without_clip) - np.mean(dice_with_clip):.6f}")
        print("   - 建议保留 box clipping")
    
    # 保存结果
    results = {
        'total_boxes': total_boxes,
        'max_outside_value': max_outside_value,
        'dice_with_clip': np.mean(dice_with_clip),
        'dice_without_clip': np.mean(dice_without_clip),
        'outside_values': outside_values
    }
    
    return results


if __name__ == "__main__":
    results = run_clipping_experiment()
