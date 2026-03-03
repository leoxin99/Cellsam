# 边界精度分析报告

> **日期**: 2026-02-02
> **分析样本**: 30 测试图像, 310 个细胞实例
> **模型**: BF Baseline Full (334样本训练)

---

## 核心发现

### IoU 分布

| IoU 范围 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| **0.0-0.1** | 3 | 1.0% | 无匹配 |
| **0.1-0.3** | 176 | **56.8%** | ⚠️ 差匹配 |
| **0.3-0.5** | 125 | 40.3% | 部分匹配 |
| **0.5-0.7** | 6 | 1.9% | 良好匹配 |
| **0.7-1.0** | 0 | 0.0% | 优秀匹配 |

**关键指标**:
- IoU 均值: **0.283** (远低于 0.5 阈值)
- IoU 中位数: **0.282**
- 只有 **1.9%** 实例达到 IoU ≥ 0.5

---

## 根本原因: 系统性过分割

### 分割偏差统计

| 偏差类型 | 数量 | 占比 |
|----------|------|------|
| **过分割** (pred > GT×1.1) | 208 | **67.1%** |
| 欠分割 (pred < GT×0.9) | 60 | 19.4% |
| 正常范围 | 42 | 13.5% |

**结论**: 模型系统性地**预测比 GT 更大的区域**

---

## 原因分析

### 1. Box 扩展累积 (主因)

训练和推理都使用 20% box 扩展:
```python
expand = 0.2  # 20% 扩展
x1_clip = x1 - bw * expand
```

问题：
- GT box 基于 GT mask 的最小外接矩形
- 预测 mask 可以扩展到 box 边界
- 20% 扩展 → 面积可增加 ~44%

### 2. BF 边界模糊

- 心肌细胞边界在 BF 图像中不清晰
- SAM 倾向于过度预测模糊区域
- Actn2/DAPI 未被有效利用

### 3. 训练监督不足

当前 Loss 设计：
- Dice Loss: 只关心重叠比例，对边界不敏感
- Boundary Loss: 权重 0.3，可能不够
- 没有直接惩罚"超出 GT 边界"的预测

---

## 解决方案建议

### 方案 A: 减小 Box 扩展 (快速验证)

```python
expand = 0.1  # 从 20% 减小到 10%
```

预期: 限制预测区域，减少过分割

### 方案 B: 增强边界 Loss

在 `CombinedLoss` 中增加边界惩罚:

```python
# 惩罚超出 GT 边界的预测
outside_penalty = (pred * (1 - target)).sum() / pred.sum()
loss += 0.3 * outside_penalty
```

### 方案 C: 使用更严格的后处理

推理时使用形态学操作缩小预测:

```python
from scipy.ndimage import binary_erosion
mask = binary_erosion(mask, iterations=5)
```

### 方案 D: 多通道输入改进

- 使用 DAPI 核区域约束细胞范围
- 使用 Actn2 边界信息精化边界

---

## 可视化

![IoU分布直方图](file:///d:/AI/paper/CellSam/experiments/boundary_analysis/iou_histogram.png)

---

## 下一步行动

| 优先级 | 方案 | 预期效果 | 复杂度 |
|--------|------|----------|--------|
| **P0** | 减小 expand 到 0.1 | +5-10% IoU | ⭐ |
| **P1** | 增加 boundary 惩罚 | +5% IoU | ⭐⭐ |
| **P2** | 多通道边界融合 | 需要新模型 | ⭐⭐⭐ |

---

## 数据文件

- IoU 分布: `experiments/boundary_analysis/iou_distribution.json`
- 分析脚本: `tools/analyze_boundary_precision.py`
