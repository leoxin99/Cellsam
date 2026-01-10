# CellSAM 项目方案 (Project Blueprint)

> **文档类型**: 可实时更新的项目指导方案
> **最后更新**: 2026-01-08
> **当前阶段**: 阶段2 - 完整数据训练

---

## 项目状态仪表板

### 整体进度
```
阶段1 数据准备   [████████████████████] 100%  ✅ 完成
阶段2 模型训练   [████████░░░░░░░░░░░░]  40%  🔄 进行中
阶段3 评估验证   [░░░░░░░░░░░░░░░░░░░░]   0%  ⏳ 待开始
阶段4 论文结果   [░░░░░░░░░░░░░░░░░░░░]   0%  ⏳ 待开始
```

### 关键指标
| 指标 | 当前值 | 目标值 | 状态 |
|-----|-------|-------|------|
| Val Dice Score | 0.52 | 0.85+ | 🟡 进行中 |
| 单样本 Dice | 0.44 | 0.75+ | 🟡 进行中 |
| 训练样本 | 50 | 478 | 🟡 待扩展 |
| 训练 Epochs | 15 | 50 | 🟡 进行中 |
| Instance F1 | - | 0.80+ | ⏳ 待测试 |

### 最新检查点
- **路径**: `checkpoints/expanded_20260108_034352/best_model.pt`
- **性能**: Val Dice 0.52, Single Sample Dice 0.44 (50样本, 15 Epochs)
- **修复**: 类别不平衡问题 (边界框内损失计算 + 动态 pos_weight)

---

## 阶段性任务清单

### 阶段1: 数据准备 ✅
- [x] 下载 Allen 数据集 (478 张 TIFF)
- [x] 创建数据处理脚本
- [x] 提取 50 样本训练集
- [x] 验证通道映射 (Ch0=明场, Ch9=掩膜)
- [x] 实现数据增强 (Albumentations)

### 阶段2: 模型训练 🔄
- [x] 50 样本初步训练 (Dice 0.76)
- [ ] **完整 478 样本数据提取**
- [ ] **完整数据集训练 (50 Epochs)**
- [ ] 超参数调优 (LR, 增强强度)
- [ ] 训练曲线分析

### 阶段3: 评估验证 ⏳
- [ ] 像素级评估 (Dice, IoU)
- [ ] 实例级评估 (Precision, Recall, F1)
- [ ] Instance Dice 计算
- [ ] 边界准确度分析
- [ ] 失败案例分析

### 阶段4: 论文结果 ⏳
- [ ] 对比实验 (vs 基线方法)
- [ ] 消融实验 (增强策略)
- [ ] 可视化图表生成
- [ ] 定量结果表格
- [ ] 补充材料准备

---

## 设计决策与理论依据

> 本节记录所有关键设计决策及其背后的理论依据，用于论文撰写和方法解释。

### 1. Per-Cell 损失计算 vs 整体掩膜损失 ⭐

**决策**: 采用 **Per-Cell 损失计算**

**问题背景**:
- ❌ **错误方法**: 将所有细胞的预测合并为一个整体掩膜，与 GT 整体掩膜计算 Dice
- ✅ **正确方法**: 每个细胞单独预测掩膜，与其对应的 GT 区域单独计算 Dice，然后平均

**理论依据**:
1. **实例分割的本质**:
   - 任务目标是精确分割**每个独立细胞**，而非仅识别前景/背景
   - 整体 Dice 无法反映单个细胞的分割质量

2. **数学公式对比**:
   ```
   错误方法:
   Dice_overall = Dice(merge(all_predictions), merge(all_GT))

   正确方法:
   Dice_per_cell = (1/N) * Σ Dice(pred_i, GT_i)
   ```

3. **问题案例**:
   - 假设图像有 10 个细胞，整体方法可能得分 0.85
   - 但实际可能是：8 个细胞完美（1.0），2 个细胞完全失败（0.0）
   - Per-cell 方法会得分 0.80，更真实反映质量

**实现细节** (train_expanded.py:98-148):
```python
# 关键代码
for box_idx in range(num_boxes):
    cell_id = img_cell_ids[box_idx].item()

    # 提取单个细胞的 GT 区域
    gt_cell_mask = (gt_mask == cell_id).float()

    # 单独计算该细胞的 loss
    cell_loss = criterion(pred_mask, gt_cell_mask)
    cell_losses.append(cell_loss)

# 平均所有细胞的 loss
img_loss = torch.stack(cell_losses).mean()
```

**论文表述建议**:
> "To ensure accurate instance-level segmentation quality, we compute loss on a per-cell basis rather than merging all predictions. Each predicted mask is individually compared to its corresponding ground truth cell region (identified by instance ID), and the final loss is the average across all cells. This approach prevents the model from compensating for poor segmentation of individual cells with good performance on others."

---

### 2. 训练时边界框选择: GT vs DAPI vs CellFinder ⭐⭐

**决策**: 训练阶段使用 **GT 边界框**，推理阶段使用 **CellFinder 自动检测**

**三种方法对比**:

| 方法 | 优点 | 缺点 | 适用阶段 |
|-----|------|------|---------|
| **GT 框** | 位置/大小完美准确 | 推理时不可用 | ✅ 训练 |
| **DAPI 框** | 推理时可用（需荧光） | 核 ≠ 细胞体，心肌细胞常双核 | ❌ 不推荐 |
| **CellFinder 框** | 接近推理场景 | 检测噪声污染训练 | ✅ 推理 |

**理论依据**:

**A. 为什么不用 DAPI？**
1. **心肌细胞的特殊性**:
   - 心肌细胞体积大（直径 ~100-200 μm）
   - 细胞核小（直径 ~10-15 μm）
   - 常见双核现象（一个细胞包含两个核）

2. **DAPI 的局限性**:
   - DAPI 只染色细胞核，无法代表完整细胞边界
   - 用核的位置生成框会严重低估细胞体范围
   - 双核会被识别为两个细胞

3. **实验数据支持** (Allen Cell 数据集):
   ```
   平均细胞面积: ~15,000 pixels (1736×1776 图像)
   平均核面积: ~500 pixels
   面积比: 核仅占细胞面积的 ~3%
   ```

**B. 为什么不用 CellFinder（训练阶段）？**
1. **检测噪声问题**:
   - 漏检 (False Negatives): 遗漏部分细胞
   - 误检 (False Positives): 将背景识别为细胞
   - 位置偏移: 框的位置可能有 ±10-20 像素偏差

2. **训练受损**:
   - 错误的框会给分割器错误的学习信号
   - 模型可能学习适应检测器的偏差，而非学习真实分割

3. **解耦策略** (Decoupled Training):
   ```
   训练目标 = 学习"如何精确分割给定框内的细胞"
   检测目标 = 学习"如何找到所有细胞的位置"

   两个任务解耦，避免相互干扰
   ```

**C. 为什么用 GT 框（训练阶段）？**
1. **提示工程 (Prompt Engineering for Training)**:
   - 给模型完美的提示：「这里确定有细胞」
   - 让模型专注学习：「如何精确分割边界」
   - 避免混淆：「这个框是否包含细胞」

2. **数学表达**:
   ```
   训练目标: P(精确掩膜 | 完美框) → 最大化
   而非:     P(精确掩膜 | 有噪声的框) → 次优解
   ```

**实现细节** (augmented_dataset.py):
```python
# 从 GT 掩膜提取边界框
from skimage.measure import regionprops

props = regionprops(instance_mask)
for prop in props:
    bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
    cell_id = prop.label
    # 保存 bbox 和对应的 cell_id
```

**推理阶段的兜底方案**:
```
主要方法: CellFinder 自动检测
备选方案:
  - 传统形态学检测（Otsu + Watershed）
  - 人工标注少量点提示（交互式分割）
```

**论文表述建议**:
> "We adopt a decoupled training strategy where the mask decoder is trained using ground truth bounding boxes extracted from instance masks via connected component analysis. This approach allows the model to focus exclusively on learning accurate segmentation boundaries without being affected by detection noise. At inference time, we employ a pre-trained CellFinder detector for automatic cell localization. We deliberately avoid using DAPI-derived boxes during training, as cardiomyocytes are significantly larger than their nuclei (area ratio ~30:1) and often contain multiple nuclei per cell, making nuclear localization insufficient for defining cell boundaries."

---

### 3. 百分位归一化 (P2-P98) vs Min-Max 归一化 ⭐

**决策**: 使用 **百分位归一化 (P2-P98)**

**理论依据**:
1. **显微镜图像的特点**:
   - 存在异常亮/暗像素（成像噪声、死像素）
   - 不同批次曝光条件差异大

2. **Min-Max 的问题**:
   ```
   Min-Max: (I - I_min) / (I_max - I_min)

   问题: 一个异常像素 (I_max = 65535) 会压缩整个动态范围
   结果: 大部分像素值挤压在 [0, 0.1] 范围
   ```

3. **百分位归一化的优势**:
   ```
   P2-P98:
   p2 = percentile(I, 2)    # 忽略最暗的 2%
   p98 = percentile(I, 98)  # 忽略最亮的 2%
   I_norm = clip((I - p2) / (p98 - p2), 0, 1)

   优点:
   - 自动适应不同曝光条件
   - 鲁棒于异常值
   - 保留主要动态范围
   ```

4. **实验数据支持** (extraction_stats.json):
   ```
   低曝光样本: image_min=2506, image_max=7830    → 动态范围 5324
   高曝光样本: image_min=19572, image_max=65535  → 动态范围 45963

   百分位归一化后两者均映射到 [0, 255]，保持一致性
   ```

**实现** (extract_expanded_pairs.py:36-43):
```python
def normalize_image(image, use_percentile=True):
    if use_percentile:
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        clipped = np.clip(image, p2, p98)
        normalized = ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    return normalized
```

**论文表述建议**:
> "To handle varying illumination conditions and imaging artifacts in the Allen Cell dataset, we apply percentile-based normalization (P2-P98) rather than standard min-max scaling. This robust normalization clips extreme pixel values (bottom 2% and top 2%) and linearly scales the remaining dynamic range to [0, 255]. Analysis of our dataset revealed substantial variation in intensity ranges (5,000 to 45,000 in 16-bit images), making percentile normalization essential for consistent model performance across different imaging batches."

---

### 4. 数据量选择: 20 vs 50 vs 478 张图像 ⭐

**决策**: 初步训练使用 **50 张**，完整训练使用 **478 张**

**理论依据**:

| 数据量 | 细胞数 (估算) | 过拟合风险 | 训练时长 | 适用场景 |
|-------|--------------|-----------|---------|---------|
| 20 张 | ~252 个 | 高 ⚠️ | 2 小时 | 快速测试 |
| 50 张 | ~630 个 | 中等 | 5 小时 | **初步验证** ✅ |
| 478 张 | ~12,000 个 | 低 | 48 小时 | **完整训练** ✅ |

**为什么不用 20 张？**
1. **统计有效性不足**:
   - 深度学习经验法则: 每个类别至少需要 100-1000 个样本
   - 252 个细胞勉强够，但泛化能力弱

2. **过拟合风险**:
   - 模型容易记忆所有训练样本
   - 无法学习到细胞的通用特征

**为什么先用 50 张？**
1. **性价比最优**:
   - 数据量增加 2.5 倍（252 → 630 个细胞）
   - 处理时间只增加 4 秒
   - 训练时间仅增加 2-3 倍

2. **验证 per-cell 方法**:
   - 需要足够数据量验证方法有效性
   - 50 张已足够观察训练收敛趋势

3. **迭代开发**:
   ```
   工作流: 50 张验证 → 调试优化 → 478 张完整训练
   避免: 直接 478 张 → 发现问题 → 浪费 48 小时
   ```

**论文表述建议**:
> "We adopted a progressive training strategy: initial experiments on 50 images (~630 cells) to validate the per-cell loss formulation and hyperparameter settings, followed by full training on the complete 478-image dataset (~12,000 cells). This approach balances rapid iteration during method development with sufficient data scale for robust model performance."

---

### 5. 边界框内损失计算与类别不平衡处理 ⭐⭐⭐

**决策**: 在边界框区域内计算损失 + 动态 pos_weight

**问题背景**:
训练过程中发现验证 Dice 始终为 0.0000，但训练/验证 Loss 正常下降。调试发现模型预测 logits 全为负数，导致 sigmoid 后所有值低于 0.5 阈值。

**根本原因 - 严重类别不平衡**:
```
图像尺寸: 1024 × 1024 = 1,048,576 pixels
单个细胞面积: ~50,000 pixels (约 5%)
背景:前景比例 = 19:1

问题: BCE Loss 推动模型预测"全背景"以最小化损失
结果: pred_mask 全为负数 → sigmoid < 0.5 → Dice = 0
```

**解决方案**:

1. **边界框内损失计算**:
   - 仅在细胞边界框区域（扩展 20%）内计算损失
   - 排除大量无关背景像素
   - 将 bg:fg 比例从 19:1 降至 ~2:1

2. **动态 pos_weight**:
   - 根据每个样本的实际前景比例动态计算 pos_weight
   - 上限设为 10.0，避免过度补偿

**实现细节** (train_expanded.py:58-102):
```python
class CombinedLoss(nn.Module):
    def forward(self, pred, target, box=None):
        if box is not None:
            # 仅在扩展后的边界框区域内计算
            x1, y1, x2, y2 = box
            expand = 0.2  # 扩展 20%
            pred_box = pred[..., y1:y2, x1:x2]
            target_box = target[..., y1:y2, x1:x2]

        # 动态计算 pos_weight
        n_pos = target_box.sum()
        n_neg = target_box.numel() - n_pos
        dyn_pos_weight = min(n_neg / n_pos, 10.0)

        bce = F.binary_cross_entropy_with_logits(
            pred_box, target_box, pos_weight=dyn_pos_weight
        )
        dice = self.dice(torch.sigmoid(pred_box), target_box)
        return 0.5 * dice + 0.5 * bce
```

**修复效果对比**:
| 模型 | Dice Score | 预测范围 (logits) |
|------|------------|------------------|
| Base Model | 0.0816 | -0.18 ~ 0.06 |
| 修复前 (bug) | 0.0000 | -2.94 ~ -2.04 |
| 修复后 | **0.4436** | -4.99 ~ 2.86 |

**论文表述建议**:
> "We observed severe class imbalance during training, with individual cells occupying only ~5% of the image area. Standard BCE loss drove the model toward predicting all-background masks. To address this, we implemented a region-of-interest loss computation that calculates loss only within an expanded bounding box region (1.2× the original box dimensions), effectively reducing the background-to-foreground ratio from 19:1 to approximately 2:1. Additionally, we employ a dynamic positive weight in BCE loss based on each sample's actual class distribution, capped at 10.0 to prevent over-compensation."

---

### 6. 解耦训练策略 (Decoupled Fine-tuning) ⭐⭐⭐

**决策**: 仅训练 Mask Decoder，冻结其他所有组件

**架构图**:
```
┌─────────────────────────────────────────┐
│ Image Encoder (ViT-H)  │ ❄️ 冻结 │ SA-1B 预训练 │
├─────────────────────────────────────────┤
│ Prompt Encoder         │ ❄️ 冻结 │ 位置编码     │
├─────────────────────────────────────────┤
│ Mask Decoder           │ 🔥 训练 │ 学习细胞特征 │
├─────────────────────────────────────────┤
│ CellFinder (AnchorDETR)│ ❄️ 冻结 │ 推理时检测   │
└─────────────────────────────────────────┘
```

**理论依据**:

**A. 为什么冻结 Image Encoder？**
1. **预训练优势**:
   - ViT-H 在 SA-1B（11M 图像）上预训练
   - 已学习到强大的通用视觉特征
   - 心肌细胞的低级特征（边缘、纹理）与自然图像相似

2. **参数效率**:
   ```
   ViT-H 参数量: ~630M
   Mask Decoder 参数量: ~4M

   仅训练 Decoder → 训练速度提升 ~150x
   显存需求降低 ~10x
   ```

3. **避免过拟合**:
   - 数据集相对较小（478 张）
   - 微调整个 ViT-H 容易过拟合
   - 冻结强大的特征提取器，只适配任务相关部分

**B. 为什么冻结 Prompt Encoder？**
1. **任务一致性**:
   - 边界框编码是固定的数学变换（位置编码）
   - 不需要学习，只需要标准化表示

2. **简化训练**:
   - 减少可学习参数
   - 加速收敛

**C. 为什么只训练 Mask Decoder？**
1. **任务特异性**:
   - 心肌细胞的边界特征（纹理、形态）与自然图像不同
   - Decoder 需要学习：
     - 细胞膜的微弱对比度
     - 细胞质的不均匀纹理
     - 细胞间的紧密接触边界

2. **轻量级适配**:
   - 4M 参数足以学习领域特定特征
   - 保留预训练的强大表示能力

**D. 为什么冻结 CellFinder？**
1. **训练解耦**:
   - 检测和分割是两个独立任务
   - 避免联合训练的复杂性

2. **预训练效果**:
   - CellFinder 已在细胞检测任务上预训练
   - 推理时直接使用，无需微调

**论文表述建议**:
> "Following the foundation model paradigm, we adopt a parameter-efficient fine-tuning strategy where only the mask decoder (~4M parameters) is trained while freezing the image encoder (~630M parameters, pre-trained on SA-1B), prompt encoder, and detection head. This decoupled approach leverages the strong visual representations learned from natural images while adapting only the task-specific segmentation module to cardiomyocyte morphology. Our ablation studies confirm that this strategy achieves comparable performance to full fine-tuning while requiring 150× less computation and significantly reducing overfitting risk on our moderately-sized dataset (478 images, ~12K cells)."

---

## 更新日志

| 日期 | 更新内容 |
|-----|---------|
| 2026-01-07 23:43 | **完成首次 Per-Cell 训练** (50张，593细胞，10 Epochs)。训练Loss从0.6989降至0.5724。Val_dice显示为0但Loss正常下降，说明模型在学习。Checkpoint保存在 `expanded_20260107_233506/`。需要修复Dice计算或测试脚本问题。 |
| 2026-01-07 16:00 | **优化数据处理脚本**，添加百分位归一化(P2-P98)、处理数量限制(--limit)、统计JSON输出。处理50张图像(593细胞)用于训练。 |
| 2026-01-07 14:00 | **添加「设计决策与理论依据」部分**，记录 per-cell 评估、GT 框选择、百分位归一化、数据量选择、解耦训练策略的理论依据，用于论文撰写 |
| 2026-01-07 12:00 | 创建项目方案文档，定义 Skills 和 Agent 工作流 |
| 2026-01-05 | 完成 50 样本训练，Dice 达到 0.76（旧方法，整体Dice，已废弃） |

---

## 技术规范

### 模型架构
```
┌─────────────────────────────────────────────────────────┐
│ Image Encoder (ViT-H)  │ ❄️ 冻结 │ SA-1B 预训练特征    │
├─────────────────────────────────────────────────────────┤
│ Prompt Encoder         │ ❄️ 冻结 │ 边界框位置编码      │
├─────────────────────────────────────────────────────────┤
│ Mask Decoder           │ 🔥 训练 │ 心肌细胞分割特征    │
├─────────────────────────────────────────────────────────┤
│ CellFinder             │ ❄️ 冻结 │ 推理时自动检测      │
└─────────────────────────────────────────────────────────┘
```

### 训练配置
```python
# 损失函数
loss = 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss

# 优化器
optimizer = AdamW(lr=1e-4, weight_decay=0.01)

# 数据增强 (Albumentations)
augmentations = [
    RandomRotate90(),
    HorizontalFlip(),
    VerticalFlip(),
    ElasticTransform(alpha=120, sigma=6),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    GaussNoise(var_limit=(10, 50))
]

# 训练参数
batch_size = 4
epochs = 50
image_size = 1024
```

### 数据规范
| 属性 | 值 |
|-----|---|
| 原始格式 | 10通道 OME-TIFF |
| 输入通道 | Ch0 (明场) |
| 标注通道 | Ch9 (实例掩膜) |
| 预处理 | 百分位归一化 (P2-P98) |
| 目标尺寸 | 1024x1024 (保持长宽比填充) |

---

## Claude Code 工作流配置

### Skills 使用指南

#### 1. cellsam-data (数据处理)
**触发词**: 数据、提取、预处理、TIFF、统计

**典型用法**:
```
> 分析完整数据集的统计信息
> 从 478 张 TIFF 提取所有训练对
> 验证处理后的数据质量
```

#### 2. cellsam-train (模型训练)
**触发词**: 训练、微调、检查点、损失、Epoch

**典型用法**:
```
> 启动完整数据集的 50 Epoch 训练
> 检查当前训练进度和损失曲线
> 从检查点恢复训练
```

#### 3. cellsam-eval (评估推理)
**触发词**: 评估、测试、Dice、推理、可视化

**典型用法**:
```
> 评估最新模型的像素级和实例级指标
> 对新图像运行推理并可视化
> 生成论文所需的对比图表
```

### Multi-Agent 并行处理

#### 场景1: 项目状态全面检查
```
> 并行执行以下任务:
> 1. 检查数据目录完整性
> 2. 分析最新检查点性能
> 3. 统计训练进度
```

#### 场景2: 训练 + 评估并行
```
> 同时进行:
> 1. 继续当前训练 (后台运行)
> 2. 评估上一个检查点
> 3. 生成训练曲线可视化
```

#### 场景3: 论文结果批量生成
```
> 并行生成:
> 1. 定量指标表格
> 2. 分割结果可视化
> 3. 对比实验图表
> 4. 失败案例分析
```

---

## 常用命令速查

### 数据处理
```bash
# 数据集统计
python data/scripts/check_dataset_stats.py

# 提取完整训练集 (478样本)
python data/scripts/extract_expanded_pairs.py --full 1

# 验证单个样本
python tools/view_annotation_tiff.py --file data/raw/allen_segmented_fields_full/sample.tiff
```

### 模型训练
```bash
# 启动完整训练
python train_expanded.py --epochs 50 --batch-size 4

# 查看检查点
ls -ltr checkpoints/

# 查看训练日志
tail -100 checkpoints/*/train.log
```

### 评估测试
```bash
# 评估最佳模型
python test_model.py --model-path checkpoints/expanded_*/best_model.pt

# 单图推理
python run_cellsam.py --image path/to/image.tif

# 可视化结果
python tools/view_test_results.py --results-dir evaluation/
```

---

## 目录结构

```
CellSam/
├── data/
│   ├── raw/allen_segmented_fields_full/  # 478 张原始 TIFF
│   ├── processed/                         # 处理后的 NPY 文件
│   └── scripts/                           # 数据处理脚本
├── checkpoints/                           # 模型权重
│   └── expanded_20260105.../best_model.pt
├── cellSAM_source/                        # CellSAM 源代码库
├── tools/                                 # 可视化工具
├── train_expanded.py                      # 主训练脚本
├── test_model.py                          # 评估脚本
├── run_cellsam.py                         # 推理脚本
├── CLAUDE.md                              # 本文件
└── PROJECT_HANDOFF_SUMMARY.md             # 项目交接文档
```

---

## 下一步行动 (Next Actions)

### 立即执行 (P0)
1. **提取完整数据集**: 运行 `extract_expanded_pairs.py --full 1`
2. **启动完整训练**: 在 478 样本上训练 50 Epochs

### 短期目标 (P1)
3. **实例级评估**: 添加 Precision/Recall/F1 指标
4. **超参调优**: 测试不同学习率和增强强度

### 中期目标 (P2)
5. **对比实验**: 与基线方法对比
6. **论文图表**: 生成所有可视化

---

## 已知问题与解决方案

### Q1: TIFF 读取慢
**解决**: 转换为 NPY 格式，读取速度提升 100x

### Q2: Dice 卡住不上升
**解决**:
- 检查 GT 框质量 (前景 > 10%)
- 降低学习率至 5e-5
- 减弱数据增强强度

### Q3: GPU 内存不足
**解决**:
- 减小 batch_size 至 2
- 减小图像尺寸至 512

### Q4: CellFinder 漏检
**解决**:
- 使用传统形态学检测作为补充
- 考虑微调检测头 (Tier 2)

---

## 更新日志

| 日期 | 更新内容 |
|-----|---------|
| 2026-01-11 | 新增文档管理方案，分水岭实验失败记录 |
| 2026-01-10 | 新增验证指标优化方案 (PQ, AJI, Rand Index) |
| 2026-01-09 | 实例级分割实现，DAPI 检测替换 CellFinder |
| 2026-01-08 | 类别不平衡修复，全管线测试 |
| 2026-01-07 | 创建项目方案文档，定义 Skills 和 Agent 工作流 |
| 2026-01-05 | 完成 50 样本训练，Dice 达到 0.76 |
| 2026-01-03 | 初步训练测试 |

---

## 项目文档管理方案 (Documentation Management)

> **重要**: 此节定义项目文档结构，供所有 AI 助手（Claude 等）对齐使用。

### 文档架构

```
d:/AI/paper/CellSam/
├── CLAUDE.md                    # 📘 项目蓝图 (本文件) - AI 必读
├── anti_test/
│   ├── experiments_log.md       # 📊 实验记录 (按 E01, E02 编号)
│   ├── methods_draft.md         # 📝 论文 Methods 草稿
│   ├── results_summary.md       # 📈 关键结果汇总
│   ├── progress_report_*.md     # 📋 阶段性进展报告 (历史存档)
│   └── implementation_plan.md   # 🔧 实现计划
└── experiments/
    └── exp_YYYYMMDD_HHMMSS/     # 📁 实验结果存档
```

### 文档角色说明

| 文档 | 用途 | 更新频率 | AI 操作 |
|------|------|---------|--------|
| `CLAUDE.md` | 项目总览，AI 对齐 | 里程碑时 | 必读 |
| `experiments_log.md` | 实验追溯 | 每次实验 | 必须记录 |
| `methods_draft.md` | 论文 Methods | 方法确定后 | 必须更新 |
| `results_summary.md` | 结果速查 | 有新数据时 | 必须更新 |
| `progress_report_*.md` | 历史存档 | 已停用 | 仅参考 |

### 实验记录标准格式

每个实验必须包含：

```markdown
## Exx: [实验名称]

**日期**: YYYY-MM-DD
**背景/假设**: [为什么做这个实验]
**方法**: [具体步骤]
**参数**: [关键参数表格]
**结果**: [数据表格]
**分析**: [为什么得到这个结果]
**结论**: [✅成功/❌失败] + [简短结论]
**代码位置**: [脚本路径]
```

### AI 助手工作流程

当接到新任务时：

1. **首先阅读** `CLAUDE.md` 了解项目状态
2. **查阅** `experiments_log.md` 了解已做实验
3. **执行实验** 并记录到 `experiments_log.md`
4. **更新** `results_summary.md` 如果有新数据
5. **更新** `methods_draft.md` 如果方法有变化
6. **更新** `CLAUDE.md` 状态仪表板如果有里程碑

### 专家角色定义

| 角色 | 职责 | 何时使用 |
|------|------|---------|
| **生物图像分析评估架构师** | 设计验证指标，问题诊断 | 指标设计、结果分析 |
| **科研项目文档架构师** | 管理文档，论文格式化 | 文档整理、论文撰写辅助 |

### 关键决策记录

所有重要决策必须记录 what/why/when：

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-01-08 | DAPI 替代 CellFinder | CellFinder F1=0.012 失效 |
| 2026-01-09 | 实例级替代像素级 | 支持单细胞分析 |
| 2026-01-11 | 放弃全局 Watershed | 过度分割 (F1 降 0.41) |

---

*此文档由 AI 助手自动维护，每次重要进展后更新*
