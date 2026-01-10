# CellSAM 结果汇总 (Results Summary)

> **用途**: 论文 Results 部分数据快速索引
> **最后更新**: 2026-01-11

---

## 核心结果表

### 表1: 检测方法对比

| Method | Precision | Recall | F1 | Status |
|--------|-----------|--------|-----|--------|
| CellFinder (原始) | 0.009 | 0.016 | **0.012** | ❌ 失败 |
| DAPI-based (本文) | 0.708 | 0.797 | **0.750** | ✅ 采用 |
| + Watershed | 0.277 | 0.461 | **0.344** | ❌ 放弃 |

**提升**: DAPI 方案较 CellFinder 提升 **62×**

---

### 表2: 分割方法对比 (10 测试样本)

| Method | Overall Dice | Cell Dice | Instance-aware |
|--------|--------------|-----------|----------------|
| 像素级合并 | 0.5757 | 0.7623 | ❌ |
| 实例级分割 | **0.7066** | - | ✅ |

**提升**: 实例级方案 Overall Dice 提升 **+0.13**

---

### 表3: 逐样本实例级分割结果

| # | Sample ID | GT Cells | Pred Cells | Dice |
|---|-----------|----------|------------|------|
| 1 | cf4fb0e8... | 10 | 10 | 0.6354 |
| 2 | 3a3cf60a... | 10 | 6 | 0.8538 |
| 3 | 27e55ff3... | 13 | 21 | 0.5942 |
| 4 | ec4c125c... | 6 | 8 | 0.4302 |
| 5 | 60f3d143... | 8 | 8 | 0.7611 |
| 6 | 5c2b8632... | 11 | 11 | 0.7325 |
| 7 | 570acc96... | 10 | 12 | 0.7964 |
| 8 | 43283e18... | 18 | 16 | 0.8553 |
| 9 | ebfc8c4d... | 8 | 8 | 0.6255 |
| 10 | 39531263... | 12 | 7 | 0.7816 |
| **Mean** | - | **10.6** | **10.7** | **0.7066** |

**数据来源**: `experiments/exp_20260109_204227/experiment_log.md`

---

### 表4: 消融实验 - 分水岭核分离

| Variant | min_distance | F1 | Δ vs Baseline |
|---------|--------------|-----|---------------|
| Baseline (无 watershed) | - | 0.750 | - |
| + Watershed | 20 | 0.304 | **-0.446** |
| + Watershed | 40 | 0.344 | **-0.406** |

**结论**: Watershed 导致过度分割，不适用于心肌细胞核

---

## 关键数值摘要 (论文用)

```
Detection F1 improvement:    0.012 → 0.750 (62× increase)
Segmentation Dice:           0.82 (boundary-tuned model)
Per-cell Dice:               0.76 (matched cells only)
Training samples:            50 images, 350 cell instances
Test samples:                10 unseen images
Model:                       CellSAM (fine-tuned SAM + Boundary Loss)
```

---

## 边界损失微调结果 (E12) ⭐

| 指标 | 旧模型 | 新模型 | 变化 |
|------|--------|--------|------|
| **PQ@0.5** | 0.024 | **0.087** | ↑ **+265%** |
| **PQ@0.3** | 0.176 | **0.249** | ↑ +42% |
| **AJI** | 0.251 | **0.314** | ↑ +25% |
| **Dice** | 0.758 | **0.822** | ↑ +8% |
| **RI** | 0.815 | **0.829** | ↑ +2% |
| **Max_IoU** | 0.489 | **0.548** | ↑ +12% |

**最佳模型**: `checkpoints/boundary_20260111_012636/best_model.pt`

---

## 指标实现状态

| Metric | 状态 | 当前值 |
|--------|------|--------|
| PQ (Panoptic Quality) | ✅ 已实现 | 0.087 |
| AJI (Aggregated Jaccard) | ✅ 已实现 | 0.314 |
| Rand Index | ✅ 已实现 | 0.829 |
| Boundary IoU | ✅ 已实现 | 0.425 |
| SarcGraph OOP | ⏳ 待集成 | - |
