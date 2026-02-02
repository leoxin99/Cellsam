# 检测问题诊断报告

> **日期**: 2026-02-02
> **状态**: 根本原因已定位

---

## 问题总结

在全面评估中发现：
- **n_pred = 4.3** (但 GT 有 10 个细胞)
- **PQ@0.5 = 0.000** (完全失败)
- **Dice = 0.715** (看起来还行)

## 根本原因

### ⚠️ SAM 预测的 Mask 远超 Bounding Box

测试数据显示：

| Box ID | Box 面积 | Mask 面积 | 比例 |
|--------|----------|-----------|------|
| 0 | 53,960 | 809,897 | **15.0x** |
| 1 | 152,766 | 629,394 | 4.1x |
| 2 | 126,324 | 786,086 | 6.2x |
| ... | ... | ... | ... |
| **平均** | **126,462** | **639,211** | **5.1x** |

**关键数据**：
- 平均每个 mask 覆盖 **61%** 的图像 (639K / 1M)
- Box 平均只有 **12%** 图像面积
- Mask 比 Box 大 **2.3x ~ 15x**

### 问题连锁效应

```
SAM 预测超大 mask
      ↓
11 个 mask 互相覆盖
      ↓
后来的覆盖前面的 (pred_mask[mask] = i+1)
      ↓
最终只剩 4.3 个不同标签
      ↓
PQ 实例匹配失败 (IoU < 0.5)
      ↓
PQ = 0
```

## 问题可视化

```
+----------------------------------+
|  Image (1024x1024)               |
|                                  |
|   +--------+                     |
|   | Box 1  |   SAM 预测...       |
|   | (12%)  |                     |
|   +--------+                     |
|         \                        |
|          \                       |
|  +------------------------+      |
|  |                        |      |
|  |    SAM Mask (61%)      |      |
|  |    覆盖其他细胞        |      |
|  |                        |      |
|  +------------------------+      |
+----------------------------------+
```

---

## 根本原因定位 ✅

### 训练-推理不一致

**训练时** (`src/losses/combined.py` L227-238):
```python
if box is not None:
    expand = 0.2  # 20% 扩展
    pred_box = pred[..., y1:y2, x1:x2]  # 只裁剪到 box 区域
    target_box = target[..., y1:y2, x1:x2]
    # Loss 只在 box 范围内计算！
```

**推理时** (`tools/comprehensive_eval.py` L125-126):
```python
mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()
pred_mask[mask] = i + 1  # 直接使用全图 mask，无裁剪！
```

### 问题原理

| 阶段 | Mask 范围 | box 外处理 |
|------|-----------|-----------|
| **训练** | 全图 | 被 Loss 忽略 (不惩罚) |
| **推理** | 全图 | 无处理 → 覆盖其他细胞 |

**模型学会了**: "box 外随便预测都不扣分"
**推理结果**: 巨大 mask 覆盖全图

---

## 解决方案

### 方案 A: 后处理裁剪 (快速)

```python
# 将预测 mask 裁剪到 box 内
pred_mask_in_box = pred_mask.copy()
pred_mask_in_box[:y1, :] = 0
pred_mask_in_box[y2:, :] = 0
pred_mask_in_box[:, :x1] = 0
pred_mask_in_box[:, x2:] = 0
```

### 方案 B: 使用 multimask_output=True

获取多个 mask 并选择最佳的：

```python
low_res_masks, iou_preds = model.mask_decoder(
    ...,
    multimask_output=True,  # 返回 3 个 mask
)
# 选择 IoU 最高的
best_idx = iou_preds.argmax()
best_mask = low_res_masks[0, best_idx]
```

### 方案 C: 调试 CellSAM 推理流程

检查 `cellSAM_source/cellSAM/model.py` 的推理设置。

---

## 下一步行动

1. **验证方案 A (Box 裁剪)**：快速实现并测试
2. **验证方案 B (multimask)**：选择最佳 mask
3. **检查 CellSAM 推理代码**：确认正确使用方式

---

## 附录：实验数据

详细结果保存在：
- `experiments/comprehensive_eval/results.json`
- `experiments/detection_analysis/diagnosis.json`
