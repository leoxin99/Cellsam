# CellSAM 推理标准文档

## 概述

本文档定义了 CellSAM 项目的标准推理流程和评估指标计算方法。所有推理测试都应遵循本文档的标准。

---

## 评估指标标准

### Instance Dice (Best-Match 方法) ⭐

**这是项目的官方 Instance Dice 计算方法。**

```python
def compute_best_match_instance_dice(pred_mask, gt_mask):
    """
    每个 GT 细胞找最佳匹配的预测细胞
    """
    for gt_region in gt_regions:
        gt_cell = (gt_mask == gt_region.label)
        best_dice = 0
        
        for pred_region in pred_regions:
            pred_cell = (pred_mask == pred_region.label)
            intersection = np.sum(gt_cell & pred_cell)
            union = np.sum(gt_cell) + np.sum(pred_cell)
            dice = 2 * intersection / union
            best_dice = max(best_dice, dice)  # 取最高匹配
        
        dices.append(best_dice)
    
    return np.mean(dices)
```

**为什么使用 Best-Match:**
- 与论文标准一致
- 对多预测覆盖情况更公平
- 不惩罚边界过分割

### PQ@0.5 (Panoptic Quality)

```
PQ = SQ × RQ

SQ (Segmentation Quality) = mean(IoU of matched pairs)
RQ (Recognition Quality) = TP / (TP + 0.5*FP + 0.5*FN)

匹配条件: IoU ≥ 0.5
```

---

## 训练 vs 推理的 Dice 差异

| 阶段 | 方法 | 说明 |
|------|------|------|
| **训练** | Direct-Match | 每个 box 对应固定的 cell_id |
| **推理** | Best-Match | 每个 GT 找最佳预测匹配 |

训练使用 Direct-Match 因为有明确的 box→cell_id 对应关系。
推理使用 Best-Match 因为需要评估整体分割质量。

---

## 标准推理流程

```
1. 加载模型
   - Baseline: get_model() (预训练权重)
   - 微调: get_model() + load_state_dict(checkpoint)

2. 加载数据
   - 验证集: load_split_ids(split='val')
   - 使用 AugmentedAllenDataset
   - target_size=(1024, 1024)
   - use_bf_only=True

3. 推理
   - 使用 segment_cellular_image() API
   - normalize=False (数据已预处理)
   - 传入 GT boxes

4. 评估
   - Instance Dice: Best-Match 方法
   - PQ@0.5: 标准计算
```

---

## 标准推理脚本

使用 `tools/standardized_inference.py`:

```bash
# 对比 Baseline vs E29
python tools/standardized_inference.py --samples 71

# 只测试 Baseline
python tools/standardized_inference.py --model baseline --samples 71

# 测试自定义 checkpoint
python tools/standardized_inference.py --model checkpoints/E30_adapter_best.pt --samples 71
```

---

## 历史结果记录

| 日期 | 模型 | Instance Dice | PQ@0.5 | 样本数 | 备注 |
|------|------|---------------|--------|--------|------|
| 2026-02-07 | Baseline (预训练) | **0.589 ± 0.237** | **0.337 ± 0.140** | 71 | Best-Match |
| 2026-02-07 | E29 (BF P1) | 待确认 | 待确认 | 71 | - |

---

## 注意事项

1. **不要使用 overlap-based dice** - 这会低估分割质量
2. **normalize 参数**: 如果数据已预处理，设置 `normalize=False`
3. **GT boxes**: 标准测试使用 GT boxes，DAPI 检测是另一个测试维度
4. **样本数**: 标准测试使用全部 71 个验证样本

---

## 更新日志

- 2026-02-07: 创建文档，确立 Best-Match 为标准方法
