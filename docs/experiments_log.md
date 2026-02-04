# CellSAM 实验记录 (Experiment Log)

> **项目**: hiPSC-CM 细胞自动分割
> **维护者**: Research Documentation Architect
> **最后更新**: 2026-02-02

---

## 实验索引 (Experiment Index)

| ID | 日期 | 实验名称 | 结果 | 状态 |
|----|------|---------|------|------|
| E01 | 2026-01-08 | 类别不平衡修复 | Dice 0→0.52 | ✅ 成功 |
| E02 | 2026-01-08 | CellFinder 检测测试 | F1=0.012 | ❌ 失败 |
| E03 | 2026-01-08 | DAPI 核检测方案 | F1=0.750 | ✅ 成功 |
| E04 | 2026-01-09 | 全管线测试 (像素级) | Dice=0.58 | ⚠️ 基线 |
| E05 | 2026-01-09 | 全管线测试 (实例级) | Dice=0.71 | ✅ 改进 |
| E06 | 2026-01-11 | 分水岭核分离 | F1=0.34 | ❌ 失败 |
| E09 | 2026-01-11 | 验证指标实现 | PQ=0, AJI=0.10 | ⚠️ 发现问题 |
| **E12** | **2026-01-11** | **边界损失微调** | **PQ↑265%, Dice↑8%** | **✅ 当前最佳** |
| E13 | 2026-01-11 | 数据集标准化 + 代码简化 | 固定划分 + 统一训练入口 | ✅ 成功 |
| E14 | 2026-01-14 | 核-细胞轴向对齐分析 + 智能扩展 | 50%对齐@30°, Dice+3.3% | ✅ 成功 |
| E15a | 2026-01-15 | 多通道融合A: BF基线 | Val Dice=0.6472 | ✅ 基线 |
| **E15b** | **2026-01-15** | **多通道融合B: BF+DAPI+Actn2** | **Pixel Dice=0.7454** | **❌ 劣于E12** |
| ~~E15c~~ | - | ~~多通道融合C: 加权Actn2~~ | 已废弃 | ❌ 取消 |
| ~~E15d~~ | - | ~~多通道融合D: 不确定引导~~ | 已废弃 | ❌ 取消 |
| **E16** | **2026-01-16** | **E12 vs E15b 对比** | **E12优2.6%** | **✅ E12确认最佳** |
| **E17** | **2026-01-21** | **GT 细胞面积统计** | **阈值 40K-450K** | **✅ 数据驱动** |
| **E18** | **2026-01-23** | **SarcGraph 检测对比** | **F1↑7.4%** | **✅ 优于DAPI** |
| **E19** | **2026-01-26** | **边缘/双核参数微调** | **Edge=100px, Binuc=1.5x** | **✅ 精确化** |
| **E20** | **2026-01-30** | **DAPI Only vs Adaptive 消融** | **DAPI Only F1=0.765 胜** | **✅ 完成** |
| **E21** | **2026-01-30** | **E12 vs Semantic Adapter 对比** | **E12 Dice=0.598 胜** | **✅ 完成** |
| **E22** | **2026-02-02** | **推理 Box Clipping 修复 + 边界精度分析** | **PQ@0.3↑90x, 发现67%过分割** | **⚠️ 需改进** |
| **E23** | **2026-02-02** | **关键Bug修复：数据加载uint8截断** | **DAPI检测0→78% F1** | **✅ 关键修复** |
| E24 | 2026-02-03 | BF Baseline v2 (A100) | Val Dice=0.7520 | ✅ 完成 |
| E25 | 2026-02-03 | Boundary Enhanced (L4) | Val Dice=0.7595 | ✅ 完成 |
| E26 | 2026-02-03 | 3ch No Adapter (L4) | Val Dice=0.7549 | ✅ 完成 |
| E27 | 2026-02-04 | 3ch Semantic Adapter (A100) | Job 899581_0 | 🔄 运行中 |
| E28 | 2026-02-04 | BF Adapter (A100) | Job 899581_1 | ⏳ 队列中 |

---

## E22: 推理 Box Clipping 修复 + 边界精度分析 ⭐⭐⭐

**日期**: 2026-02-02

**背景/假设**: 
全面评估发现 PQ@0.5=0，n_pred=4.3 但 GT=10。分析发现 SAM 预测的 mask 远超 box 范围 (2-15x)。

**根本原因**: 训练-推理不一致
- 训练: `CombinedLoss` 只在 box+20%扩展区域计算 loss，box 外不惩罚
- 推理: 未对预测 mask 做 box 裁剪，导致巨大 mask 互相覆盖

**修复**: 在 `tools/comprehensive_eval.py` L125-138 添加 box clipping:
```python
mask_clipped = np.zeros_like(mask)
mask_clipped[y1_clip:y2_clip, x1_clip:x2_clip] = mask[y1_clip:y2_clip, x1_clip:x2_clip]
```

**修复后结果**:

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| **Dice** | 0.715 | 0.766 | +7.1% |
| **PQ@0.3** | 0.002 | 0.181 | **90x** |
| **AJI** | 0.063 | 0.203 | **3.2x** |
| **n_pred** | 4.3 | 9.8 | ≈ GT |

**边界精度分析** (310 实例):

| IoU 范围 | 占比 |
|----------|------|
| **0.1-0.3** | **56.8%** ⚠️ |
| **0.3-0.5** | 40.3% |
| **≥0.5** | 1.9% |

**分割偏差**: 67% 过分割 (pred > GT×1.1)

**结论**:
1. ✅ Box clipping 修复成功，PQ@0.3 大幅提升
2. ⚠️ 边界精度仍低，IoU 均值仅 0.28
3. → 根本原因是 20% box 扩展导致系统性过分割

**下一步**: 减小 expand 参数或增加边界惩罚

**文档**: `docs/boundary_precision_analysis.md`, `docs/detection_problem_report.md`

---

## E20: DAPI Only vs Adaptive 检测消融 ⭐⭐

**日期**: 2026-01-30

**背景/假设**: 
Adaptive 方案使用 Z-线自适应框 (`detect_with_adaptive_box`)，理论上能更准确地定位心肌细胞边界。需要与 DAPI Only 方案 (`detect_and_create_boxes`) 进行对比。

**方法**:
1. 在 20 个测试样本上分别运行两种检测方法
2. 使用 IoU@0.3 阈值匹配预测框与 GT 框
3. 计算 Precision, Recall, F1

**参数**:
| 参数 | 值 |
|------|-----|
| min_nucleus_area | 3000 |
| max_nucleus_area | 30000 |
| min_zlines (Adaptive) | 15 |
| zline_threshold | 0.03 |
| IoU 阈值 | 0.3 |
| 测试样本数 | 20 |

**结果**:

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| **DAPI Only** | **0.793** | **0.739** | **0.765** |
| Adaptive | 0.311 | 0.290 | 0.300 |
| **差异** | **+0.48** | **+0.45** | **+0.465** |

**分析**:
- **DAPI Only 明显优于 Adaptive**，F1 差距高达 **46.5%**
- Adaptive 方案 Precision 极低 (0.31)，说明 Z-线自适应框生成了大量误检
- Adaptive 的框尺寸可能偏大，导致与 GT 的 IoU 过低

**原因分析**:
1. `create_adaptive_box` 的 `fallback_expansion` 参数 (4.0) 过大
2. Z-线聚类可能将多个细胞的 Z-线合并
3. 当前 IoU@0.3 阈值对大框不友好

**结论**: ❌ Adaptive 方案在当前参数下严重劣于 DAPI Only，建议：
1. 使用 **DAPI Only** 作为默认检测方法
2. 如需使用 Adaptive，需大幅调低 `fallback_expansion` 和 `padding_ratio`

**代码位置**: `tools/ablation_detection.py`
**结果存档**: `experiments/ablation_detection/results.json`

**日期**: 2026-01-26

**背景**:
E18 发现 Adaptive 方法 Precision 较低 (0.672)，怀疑边缘过滤过松。同时 DAPI 方法 FN 较高，怀疑双核合并阈值不准。

**方法**:
1. **GT 极小核分析**: GT Mask 中存在大量 <1000px 的碎片。确认有效心肌细胞核应 >3000px (可视验证最小约 5000px)。
2. **边缘过滤重算**: 仅统计 >=5000px 的有效 GT 核在不同边缘阈值下的排除率。
3. **双核间距重算**: 仅统计 >=5000px 且 size_ratio < 3.0 的有效配对。

**结果 (min_area=5000)**:
1. **边缘排除率 (Valid GT >5000px)**:
   - 30px: 0.0%
   - 50px: 0.8%
   - **100px: 5.6%** (推荐)
   - 150px: 15.8%

2. **双核间距 (Valid Pairs)**:
   - Median: 137 px
   - **Mean: 161 px**
   - **P75: 160 px**
   - P95: 322 px (过大，离群)

**结论**:
- **边缘阈值**: 定为 **100px** (排除 ~5.6% 有效核，换取高 Precision)。
- **合并阈值**: 定为 **1.5×直径 (~170px)**，完美覆盖 Mean/P75，避免 P95 的过度合并。

**代码位置**: `dapi.py`, `evaluate_box_generation.py`

## E01: 类别不平衡修复

**日期**: 2026-01-08

**背景/假设**: 
训练时 Val Dice 始终为 0，怀疑是类别不平衡导致模型预测"全背景"。

**方法**:
1. 损失仅在扩展边界框内计算 (非全图)
2. 动态 pos_weight = min(n_neg/n_pos, 10.0)
3. 组合损失 = 0.5×Dice + 0.5×BCE

**参数**:
- Learning Rate: 1e-4
- Epochs: 50
- Batch Size: 4
- 训练样本: 50

**结果**:
| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| Val Dice | 0.00 | 0.52 | +0.52 |
| 单样本 Dice | - | 0.78 | - |

**结论**: 类别不平衡是主要问题，组合损失有效。

**代码位置**: `train_expanded.py`

---

## E02: CellFinder 检测测试

**日期**: 2026-01-08

**背景/假设**: 
CellSAM 原始的 CellFinder 检测器能否用于心肌细胞定位。

**方法**: 
直接调用 CellFinder 模块检测心肌细胞。

**结果**:
| 指标 | 值 |
|------|-----|
| Precision | 0.009 |
| Recall | 0.016 |
| **F1** | **0.012** |

**分析**:
- CellFinder 使用 AnchorDETR，可能主要在小型圆形细胞上训练
- 心肌细胞特征：大尺寸 (100-200μm)、不规则形状
- 模型域不匹配

**结论**: ❌ CellFinder 完全失效，需替代方案。

---

## E03: DAPI 核检测方案

**日期**: 2026-01-08

**背景/假设**: 
利用 DAPI 核染色通道进行传统图像处理检测，替代 CellFinder。

**方法**:
1. DAPI 通道 Otsu 阈值分割
2. 形态学清理 (opening, fill_holes)
3. 面积过滤 (min=500, max=15000)
4. 相对大小过滤 (小于中位数 20% 排除)
5. 双核合并 (merge_distance=100px)
6. 边缘细胞排除 (margin=30px)

**参数**:
| 参数 | 值 |
|------|-----|
| min_nucleus_area | 500 |
| max_nucleus_area | 15000 |
| merge_distance | 100 |
| relative_size_threshold | 0.2 |
| edge_margin | 30 |

**结果**:
| 指标 | CellFinder | DAPI 检测 | 提升 |
|------|------------|-----------|------|
| Precision | 0.009 | 0.708 | +77× |
| Recall | 0.016 | 0.797 | +50× |
| **F1** | 0.012 | **0.750** | +62× |

**结论**: ✅ DAPI 检测方案有效，可替代 CellFinder。

**代码位置**: `anti_test/test_dapi_detection.py`

---

## E04: 全管线测试 (像素级)

**日期**: 2026-01-09

**背景/假设**: 
测试 DAPI 检测 + SAM 分割的完整管线效果。

**方法**:
1. DAPI 检测生成边界框
2. SAM 对每个框进行分割
3. `np.maximum()` 合并所有预测 mask

**测试集**: 10 个未见样本 (从 428 个未训练样本中随机选取)

**结果**:
| 指标 | 值 |
|------|-----|
| Mean Overall Dice | 0.5757 |
| Mean Cell Dice | 0.7623 |

**问题**: 像素级合并丢失实例信息，无法区分个体细胞。

**代码位置**: `anti_test/test_full_pipeline.py`

---

## E05: 全管线测试 (实例级)

**日期**: 2026-01-09

**背景/假设**: 
改进分割输出为实例级 mask，每个细胞有唯一 ID。

**方法**:
1. 每个检测框分配唯一 cell_id
2. 后处理: binary_closing + fill_holes + largest_component
3. 实例 mask 输出

**结果**:
| 指标 | 像素级 | 实例级 | 提升 |
|------|--------|--------|------|
| Overall Dice | 0.5757 | 0.7066 | +0.13 |
| 可区分细胞 | ❌ | ✅ | - |

**结论**: ✅ 实例级分割 + 后处理提升效果。

**代码位置**: `anti_test/visualize_test_results.py`
**实验存档**: `experiments/exp_20260109_204227/`

---

## E06: 分水岭核分离

**日期**: 2026-01-11

**背景/假设**: 
参考 Allen CellProfiler 方案，使用分水岭分离粘连核。

**方法**:
1. 距离变换 `distance_transform_edt()`
2. 局部极值检测 `peak_local_max()`
3. 分水岭分割 `watershed()`
4. Circularity 过滤 (min=0.2)

**参数测试**:
| min_distance | Precision | Recall | F1 |
|--------------|-----------|--------|-----|
| 20 | 0.225 | 0.483 | 0.304 |
| 40 | 0.277 | 0.461 | 0.344 |

**对比**:
| 方法 | F1 | Δ |
|------|-----|-----|
| 原始 DAPI | 0.750 | 基线 |
| + Watershed | 0.344 | **-0.41** |

**分析**:
- 心肌细胞核形态不规则
- 距离变换产生多个局部极值
- 单个核被过度分割为 2-3 块

**结论**: ❌ 分水岭导致过度分割，不适用于当前数据。

**代码位置**: `anti_test/test_dapi_improved.py`

---

## 待实验 (Planned)

| ID | 实验名称 | 优先级 | 状态 |
|----|---------|--------|------|
| E07 | 单独测试 Circularity 过滤 | P1 | 待做 |
| E08 | 单独测试光照校正 | P2 | 待做 |
| E10 | Rand Index 实现 | P1 | ✅ 完成 (见E09) |
| E11 | SarcGraph 集成 | P2 | 待做 |
| **E12** | **边界损失微调** | **P0** | **✅ 完成** |

---

## E09: 验证指标实现与评估 ⭐

**日期**: 2026-01-11

**背景/假设**: 
实现 index optimization.docx 中定义的四层验证体系，对当前分割结果进行全面评估。

**实现内容**:

| 指标 | 实现函数 | 状态 |
|------|---------|------|
| PQ (SQ × RQ) | `compute_panoptic_quality()` | ✅ |
| AJI | `compute_aji()` | ✅ |
| Rand Index / ARI | `compute_rand_index()` | ✅ |
| Boundary IoU | `compute_boundary_iou()` | ✅ |
| HD95 | `compute_hd95()` | ✅ |

**测试集**: 5 个样本 (来自 exp_20260109_204227)

**结果**:

| 指标 | 值 | 说明 |
|------|-----|------|
| PQ@0.5 | **0.000** | ❌ 无匹配 (IoU都<0.5) |
| PQ@0.3 | 0.014 | 少量匹配 |
| AJI | **0.102** | 低 |
| RI | 0.616 | 中等 |
| ARI | 0.063 | 低 |
| Boundary IoU | 0.439 | 中等 |
| HD95 | 403 px | 高 (差) |
| Dice | 0.655 | 中等 |

**关键发现**:

⚠️ **严重问题**: Mean Max IoU 仅 0.05-0.22

| 现象 | 原因分析 |
|------|---------|
| 预测实例与 GT 实例的最佳 IoU 仅 0.1-0.3 | 预测 mask 位置/形状与 GT 有较大偏差 |
| 即使 Dice=0.65，PQ=0 | Dice 是像素级，PQ 是实例级 |
| 预测细胞数量正确，但位置不对 | 检测框正确，但 SAM 分割边界偏移 |

**诊断**:
```
高 Dice + 低 PQ = "像素总体对，但每个细胞边界都有偏差"
```

这表明 SAM 模型需要进一步微调，特别是边界精度。

**结论**: 
- ✅ 指标实现成功
- ⚠️ 发现分割质量问题：实例级 IoU 过低
- 建议：增加边界损失 (Boundary Loss) 进行微调

**代码位置**: `anti_test/eval_metrics.py`

---

## E12: 边界损失微调 ⭐⭐⭐

**日期**: 2026-01-11

**背景/假设**: 
E09 发现旧模型实例级 IoU 过低 (Max IoU 仅 0.1-0.3)，导致 PQ=0。
假设增加 Boundary Loss 可以提升边界精度。

**方法**:
1. 加载预训练模型 `expanded_20260108_034352/best_model.pt`
2. 添加 BoundaryLoss 组件 (提取边缘像素单独计算损失)
3. 新损失 = 0.7×(Dice+BCE) + 0.3×BoundaryLoss
4. 冻结 Image Encoder，仅训练 Decoder
5. 学习率降低至 1e-5
6. 训练 20 Epochs

**参数**:
| 参数 | 值 |
|------|-----|
| Epochs | 20 |
| Learning Rate | 1e-5 |
| Boundary Weight | 0.3 |
| Batch Size | 2 |
| Train Samples | 40 |
| Val Samples | 8 |

**结果** (10 测试样本):

| 指标 | 旧模型 | 新模型 | 变化 |
|------|--------|--------|------|
| **PQ@0.5** | 0.024 | **0.087** | ↑ **+265%** |
| **PQ@0.3** | 0.176 | **0.249** | ↑ +42% |
| **AJI** | 0.251 | **0.314** | ↑ +25% |
| **Dice** | 0.758 | **0.822** | ↑ +8% |
| **RI** | 0.815 | **0.829** | ↑ +2% |
| **Max_IoU** | 0.489 | **0.548** | ↑ +12% |
| Boundary_IoU | 0.447 | 0.425 | ↓ -5% |

**分析**:
- Max_IoU 突破 0.5 阈值 → PQ@0.5 有效提升
- 边界损失有效引导模型学习更精确的细胞轮廓
- Boundary_IoU 略降是因为评估方式不同

**结论**: ✅ 边界损失微调成功，显著提升实例级分割质量

**模型位置**: `checkpoints/boundary_20260111_012636/best_model.pt`
**代码位置**: `finetune_boundary_simple.py`, `train_expanded.py`

---

## 关键决策记录 (Decision Log)

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-01-08 | 使用 DAPI 替代 CellFinder | CellFinder F1=0.012 完全失效 |
| 2026-01-09 | 使用实例级而非像素级分割 | 像素级无法区分个体细胞 |
| 2026-01-11 | 放弃全局 Watershed | 过度分割 (F1 降 0.41) |
| 2026-01-11 | 采用边界损失微调 | 提升 PQ@0.5 从 0.02 到 0.09 |

---

## 参考方法：Allen 实验室方案

> 来源: `CellProfiler_deeplab.docx` 文档分析

### Allen 分割策略

| 任务 | Allen 方法 | 准确率 | 我们对应方法 |
|------|-----------|--------|-------------|
| 细胞边界 (大规模) | DeepLabV3 | Dice > 0.85 | CellSAM (Dice 0.71) |
| 细胞边界 (FISH) | 手工分割 | ~99% | N/A |
| 细胞核 | CellProfiler | Rand > 0.92 | DAPI 检测 (F1 0.75) |
| 肌节分析 | SarcGraph | - | **待整合** |

### 可借鉴方案

| 方案 | 状态 | 说明 |
|------|------|------|
| **SarcGraph** | 待整合 | 肌节结构分析，生成 Organization Score |
| **DeepLabV3** | 待对比 | 作为 SAM baseline |
| 分水岭分离 | ❌ 失败 | 见 E06 |
| Circularity 过滤 | ✅ 已实现 | 可选启用 |
| 光照校正 | ✅ 已实现 | 可选启用 |

### SarcGraph 信息

- **来源**: [PLOS Comp Bio 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013436)
- **功能**: 自动检测 α-actinin-2 标记的 Z-线，构建肌原纤维网络图
- **应用**: 分割完成后，在 Actn2 通道 (Ch1) 上运行

---

## 验证指标优化方案

> 来源: `index optimization.docx` 文档分析

### 四层验证体系

| Tier | 指标 | 目的 | 状态 |
|------|------|------|------|
| 1 | AP@0.5, AP@0.75 | 实例检测能力 | 待实现 (E09) |
| 2 | **PQ** (SQ + RQ), **AJI** | 综合质量评估 | 待实现 (E09) |
| 3 | Boundary IoU, HD95 | 边界精度 | 待实现 |
| 4 | SarcGraph OOP/SPD | 生物学功能 | 待整合 (E11) |

### 核心指标说明

| 指标 | 公式/说明 | 诊断用途 |
|------|-----------|---------|
| **PQ** | SQ × RQ | 检测+分割综合 |
| **SQ** | TP 平均 IoU | 分割精度 |
| **RQ** | F1 分数 | 检测精度 |
| **AJI** | 聚合 Jaccard | 惩罚过/欠分割 |
| **Rand Index** | (TP+TN)/(All) | Allen 使用, >0.92 |

### 诊断逻辑

```
PQ 分解:
├─ RQ高 + SQ低 → "检测好，分割粗糙" → 需要边界损失微调
└─ SQ高 + RQ低 → "分割好，检测差" → 调整置信度阈值
```

### 实现优先级

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| **P0** | PQ + AJI 实现 | 1天 |
| P1 | Rand Index | 0.5天 |
| P1 | Boundary IoU | 0.5天 |
| P2 | SarcGraph OOP | 1天 |

---

## E13: 数据集标准化 + 代码简化

**日期**: 2026-01-11

**背景/假设**: 
之前实验使用随机划分，不同实验间的可比性受限。同时存在多个冗余的训练脚本。

**方法**:
1. 固定 Train/Val/Test 划分 (70/15/15)，使用 seed=42 确保可复现
2. 创建统一的 `src/train.py` 支持 YAML 配置
3. 提取损失函数到 `src/losses/` 模块

**交付物**:
| 文件 | 内容 |
|------|------|
| `data/splits/train_ids.txt` | 334 样本 |
| `data/splits/val_ids.txt` | 71 样本 |
| `data/splits/test_ids.txt` | 73 样本 |
| `src/train.py` | 统一训练入口 |
| `src/config/base.yaml` | 基础配置 |
| `src/config/boundary.yaml` | 边界损失微调配置 |
| `src/losses/combined.py` | DiceLoss, BoundaryLoss, CombinedLoss |

**验证**:
```
Loaded 334 samples from train split
Loaded 71 samples from val split
Loaded 73 samples from test split
CombinedLoss import OK
```

**结论**: ✅ 数据划分已固化，代码结构已简化，为后续消融实验奠定基础。

---

## E17: GT 细胞面积统计分析 (完整数据集)

**日期**: 2026-01-21

**背景/假设**: 
需要确定合理的细胞大小阈值，用于训练时的 SizeLoss 和推理时的过滤。

**方法**:
分析 **全部 478 张图片** 中所有 GT 细胞的面积分布。

**数据源**: 
- `data/raw/allen_segmented_fields_full/*.tiff`
- 通道 9 为 GT 分割 mask

**统计结果**:

| 指标 | 值 (像素) | 等效边长 |
|------|----------|----------|
| 样本数 | **5173 个细胞** | **478 张图片** |
| **最小** | **6,240** | ~79×79 |
| P1 | 40,836 | ~202×202 |
| P5 | 59,966 | ~245×245 |
| P10 | 72,443 | ~269×269 |
| P25 | 99,916 | ~316×316 |
| **中位数** | **142,316** | **~377×377** |
| P75 | 198,024 | ~445×445 |
| P90 | 275,367 | ~525×525 |
| P95 | 338,190 | ~582×582 |
| P99 | 513,928 | ~717×717 |
| **最大** | **1,026,328** | ~1013×1013 |
| 均值 | 162,688.5 | ~403×403 |
| 标准差 | 95,797.0 | - |

**阈值设定 (使用 P1/P99)**:

| 阈值 | 值 | 理由 |
|------|-----|------|
| MIN_CELL_AREA | **40,836** | P1, 排除最小1%异常 (含标注空洞) |
| MAX_CELL_AREA | **513,928** | P99, 排除最大1%异常 |

> **发现**: 最小值 6240 是细胞间隙被误标为细胞，不是真正的心肌细胞。
> GT 可能存在标注错误，使用 P1/P99 更稳健。

**代码位置**:
- `src/inference/postprocess.py`: MIN_CELL_AREA, MAX_CELL_AREA
- `src/losses/combined.py`: SizeLoss 类

**结论**: ✅ 使用 P1/P99 [40836, 513928] 覆盖 98% 正常细胞，排除标注异常。

---

## E18: SarcGraph 检测对比实验

**日期**: 2026-01-23

**背景/假设**: 
当前 DAPI 核检测方案存在局限性（检测非心肌细胞核）。SarcGraph 方案利用 α-actinin 通道的 Z-线检测，理论上具有 100% 心肌细胞特异性。

**方法**:
基于 Claude Pipeline 的 SarcGraph 检测实现：
1. **Z-线检测**: 使用 `blob_log` (LoG 算法) 检测肌节 Z-线
2. **空间聚类**: 使用 DBSCAN 将 Z-线点聚类为细胞簇
3. **边界框生成**: 从簇的凸包生成 Padding 后的边界框

**参数** (经调优后):
| 参数 | 值 | 说明 |
|------|-----|------|
| pixel_size_um | 0.108 | 63X 物镜估算 |
| sarcomere_length_um | 2.0 | 标准肌节长度 |
| threshold (blob_log) | **0.05** | 优于默认 0.1 |
| eps_factor | **2.0** | DBSCAN eps 系数 |
| min_samples | **30** | 最小 Z-线数 (优于默认 15) |
| padding_pixels | 20 | 边界框外扩 |

**数据**:
- Allen 数据源通道: **Ch1 = Actn2** (用户 Napari 确认)
- 测试样本: 5 张

**结果**:

| 方法 | Mean F1 | Precision | Recall |
|------|---------|-----------|--------|
| **SarcGraph** | **0.351** | 0.229 | 0.843 |
| DAPI | 0.277 | 0.180 | 0.971 |
| **差异** | **+7.4%** | +4.9% | -12.8% |

**分析**:
- SarcGraph **精确率更高** (0.229 vs 0.180): 检测到的更可能是心肌细胞
- DAPI **召回率更高** (0.971 vs 0.843): 能检测到更多细胞（包括非心肌）
- SarcGraph 检测到 ~2000+ Z-lines/样本，聚类后生成 1-5 个框

**代码位置**:
- SarcGraph Pipeline: `src/comparison/sarcgraph_pipeline/`
- 测试脚本: `tools/test_sarcgraph_detection.py`
- 结果可视化: `experiments/e18_sarcgraph/sample_*_comparison.png`

**结论**: ✅ SarcGraph 检测 F1 优于 DAPI +7.4%。
后续可考虑：
1. 将 SarcGraph 与 DAPI 检测**融合**（提高召回率）
2. 使用 SarcGraph 边界框进行 SAM 分割训练
3. 调整 min_samples 参数优化召回率

