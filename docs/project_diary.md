# 项目梳理日记 (Project Diary)

> **项目**: CellSAM - hiPSC-CM 心肌细胞自动分割
> **创建日期**: 2026-01-12
> **最后更新**: 2026-01-12

---

## 📌 快速参考

| 项目 | 当前值 | 目标 |
|------|--------|------|
| Detection F1 | 0.750 | >0.85 |
| Segmentation Dice | 0.822 | >0.85 |
| PQ@0.5 | 0.087 | >0.4 |
| RI (Rand Index) | 0.829 | >0.92 |

---

## 1. 评估指标详解

### 1.1 实例匹配指标

| 指标 | 英文 | 计算方式 | 含义 |
|------|------|---------|------|
| **IoU** | Intersection over Union | `交集 / 并集` | 两个 Mask 的重叠程度 (0~1) |
| **TP** | True Positive | IoU ≥ 阈值的匹配对数 | 正确检测+正确分割 |
| **FP** | False Positive | 未匹配的预测 | 假阳性 (预测了不存在的细胞) |
| **FN** | False Negative | 未匹配的真实 | 漏检 (真实存在但没预测到) |

### 1.2 核心评估指标

| 指标 | 公式 | 含义 | 目标值 |
|------|------|------|--------|
| **PQ** | SQ × RQ | 检测正确率 × 分割质量 (综合评价) | >0.4 |
| **SQ** | mean(TP 的 IoU) | 正确匹配的边界有多准 | >0.7 |
| **RQ** | TP / (TP + 0.5FP + 0.5FN) | F1 检测分数 | >0.6 |
| **AJI** | Σ(交集) / Σ(并集+FP) | 聚合 Jaccard，惩罚过/欠分割 | >0.5 |
| **RI** | (正确配对) / (总配对) | 像素级聚类一致性 (Allen 标准) | **>0.92** |
| **Dice** | 2×交集 / (A+B) | 像素级重叠率 | >0.85 |

### 1.3 指标诊断逻辑

```
如果 PQ 低:
├─ SQ 高 + RQ 低 → 检测差 → 改进核检测/Box 提示
└─ SQ 低 + RQ 高 → 边界差 → 增加 Boundary Loss 权重
```

---

## 2. 实例匹配算法 (Hungarian)

**目的**: 找到预测细胞与真实细胞的**最优一对一匹配**

```
步骤 1: 计算 IoU 矩阵
        ┌───────────────────┐
        │ Pred\GT  G1   G2  │
        │   P1    0.8  0.1  │
        │   P2    0.2  0.6  │
        └───────────────────┘

步骤 2: Hungarian 算法 (最小化 1-IoU)
        P1 ↔ G1 (IoU=0.8) ✓ TP
        P2 ↔ G2 (IoU=0.6) ✓ TP

步骤 3: 阈值过滤 (IoU ≥ 0.5 才算 TP)
```

**代码实现**: `anti_test/eval_metrics.py` → `match_instances()`

---

## 3. 边界损失函数 (Boundary Loss)

### 3.1 问题背景

传统 Dice Loss 只关注整体像素分布，导致：
- 像素级 Dice 高 (0.82)
- 实例级 PQ@0.5 接近 0 (边界不准)

### 3.2 解决方案

```python
def get_boundary_mask(mask):
    eroded = binary_erosion(mask, disk(3))  # 形态学腐蚀
    boundary = mask - eroded                 # 边界 = 原始 - 腐蚀
    return boundary  # 只保留边缘 3 像素
```

**损失组合**:
```
Total = 0.7 × (Dice + BCE) + 0.3 × BoundaryLoss
```

### 3.3 效果 (E12 实验)

| 指标 | 无边界损失 | 有边界损失 | 变化 |
|------|-----------|-----------|------|
| PQ@0.5 | 0.024 | 0.087 | **+265%** |
| Max_IoU | 0.489 | 0.548 | **+12%** |

---

## 4. CellFinder 替代方案

### 4.1 方案对比

| 方案 | 检测 F1 | 状态 |
|------|---------|------|
| CellFinder | 0.012 | ❌ 失败 (针对神经元设计) |
| **DAPI 核检测** | **0.750** | ✅ 采用 |

### 4.2 DAPI 核检测流程

```python
def detect_nuclei_dapi(dapi_channel):
    # 1. Otsu 阈值
    threshold = threshold_otsu(dapi_channel)
    binary = dapi_channel > threshold * 0.8
    
    # 2. 形态学清理
    cleaned = remove_small_objects(binary, min_size=200)
    
    # 3. 合并同一细胞的多个核 (距离 < 100px)
    merged = merge_close_nuclei(labeled, distance=100)
    
    # 4. 扩展为细胞 Box (6x)
    boxes = create_bounding_boxes(merged, expansion=6)
    return boxes
```

---

## 5. 训练流程

### 5.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 准备阶段                                                     │
│    python data/scripts/generate_splits.py                       │
│    → train_ids.txt, val_ids.txt, test_ids.txt                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌───────────────────────────────┴─────────────────────────────────┐
│ 2. 基础训练                                                     │
│    python src/train.py --config src/config/base.yaml           │
│    → checkpoints/base_xxx/best_model.pt                        │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌───────────────────────────────┴─────────────────────────────────┐
│ 3. 边界微调 (可选)                                              │
│    python src/train.py --config src/config/boundary.yaml       │
│    → 加载 base 模型 + Boundary Loss 微调                        │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌───────────────────────────────┴─────────────────────────────────┐
│ 4. 评估                                                         │
│    python anti_test/visualize_test_results.py                  │
│    → PQ, AJI, RI 指标                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 微调的 "Base 模型" 是什么

```yaml
# src/config/boundary.yaml
model:
  checkpoint: "checkpoints/expanded_xxx/best_model.pt"  # 这就是 base 模型
```

- **Base 模型**: 第一次训练产生的权重 (Dice + BCE)
- **微调**: 加载 Base 权重，用 Boundary Loss **继续训练**

---

## 6. 数据集结构

### 6.1 固定划分

| Split | 样本数 | 比例 |
|-------|--------|------|
| Train | 334 | 70% |
| Val | 71 | 15% |
| Test | 73 | 15% |

### 6.2 短 ID 映射

```
train_001 → 570acc96_5500000013_63X_20190807_S1_P14_B3_annotations_corrected
train_002 → d3328698_5500000013_63X_20190807_S1_P7_B4_annotations_corrected
...
```

映射文件: `data/splits/{split}_mapping.csv`

---

## 7. 代码结构

```
src/
├── train.py              # 统一训练入口
├── augmented_dataset.py  # 数据加载 + load_split_ids()
├── config/
│   ├── base.yaml         # 基础配置
│   └── boundary.yaml     # 边界微调配置
└── losses/
    └── combined.py       # DiceLoss, BoundaryLoss, CombinedLoss
```

---

## 附录: 实验记录索引

| ID | 日期 | 实验 | 结果 |
|----|------|------|------|
| E01 | 01-08 | 类别不平衡修复 | Dice 0→0.52 ✅ |
| E02 | 01-08 | CellFinder 测试 | F1=0.01 ❌ |
| E03 | 01-08 | DAPI 核检测 | F1=0.75 ✅ |
| E12 | 01-11 | 边界损失微调 | PQ↑265% ✅ |
| E13 | 01-11 | 数据集标准化 | ✅ |

详细记录见: `anti_test/experiments_log.md`

---

## 8. 问答记录 (Q&A Log)

### 2026-01-13: 检测与训练策略分析

#### Q1: 框尺寸阈值在不同数据集能泛化吗？

**导师建议**: 根据 GT 框的最大/最小尺寸设定阈值

**回答**: ⚠️ **绝对尺寸不泛化**

- 不同数据集放大倍数不同 (20X vs 63X vs 100X)
- 图像分辨率不同

**解决方案**: 使用**相对面积比例**
```python
min_area_ratio = 0.001  # 占图像面积 0.1%
max_area_ratio = 0.1    # 占图像面积 10%
```

---

#### Q2: Actn2 通道在检测阶段有价值吗？

**回答**: ✅ 有价值

**用途**: 过滤非心肌细胞核 (如成纤维细胞)

```python
# Actn2 阳性区域 = 心肌细胞区域
valid_nuclei = nuclei_mask & (actn2_channel > threshold)
```

---

#### Q3: 训练时 GT Box 精确，推理时检测 Box 偏大，继续训练有意义吗？

**回答**: ⭐ 核心问题

| 阶段 | Box 来源 | 特点 |
|------|---------|------|
| 训练 | GT Mask 边界 | 紧贴细胞，+5px padding |
| 推理 | DAPI×6 扩展 | 通常大于细胞 |

**SAM 对 Box 尺寸有一定容忍度**，但需要：

**建议**: 训练时加入 **Box 扰动增强**
```python
def augment_box(box, noise_ratio=0.2):
    # 随机扩大 Box，让模型学习适应不精确的 Box
```

---

#### Q4: GT Box 从哪来？是单独的 Box 通道吗？

**回答**: **不是**。GT Box 是动态计算的。

```python
# 从 Mask (Ch9) 的连通区域计算
for region in measure.regionprops(mask):
    box = region.bbox  # 自动提取边界框
```

Allen 数据没有专门的 Box 通道。

---

#### Q5: 分割结果是边界+填充，还是只填充？

**GT 格式**: 填充式 (整个细胞区域有 label)
```
GT Mask: 背景=0, 细胞1=1(所有像素), 细胞2=2(所有像素)
```

**SAM 输出**: 概率图 (0~1)
- 边界处概率 ~0.5
- 内部概率 ~0.9

**这是正常的**，阈值化后变成填充结果。

---

#### 核心结论

**当前瓶颈在检测 (RQ=0.16) 而非分割 (SQ=0.55)**

| 优先级 | 行动 |
|--------|------|
| P0 | 训练时加入 Box 扰动增强 |
| P0 | 使用相对尺寸阈值 |
| P1 | 利用 Actn2 过滤非心肌核 |

---

### 2026-01-14: 数据统计与 Actn2 实现

#### Q1: 相对尺寸阈值是如何得到的？

**统计数据** (50 个已处理样本):
| 指标 | 值 |
|------|-----|
| 图像尺寸 | **1736×1776** (非 1024×1024) |
| 细胞总数 | 593 个 |
| P1 面积 | 40,464 px² (1.3%) |
| P50 面积 | 135,508 px² (4.4%) |
| P99 面积 | 425,464 px² (13.8%) |

**建议阈值**: `min_ratio=0.01`, `max_ratio=0.15`

---

#### Q2: GT 框从 Mask 动态计算是什么意思？

GT 框是用 `regionprops(mask).bbox` 从 Mask 连通区域自动计算的**最小外接矩形**，不是独立存储的。Allen 数据没有 Box 通道。

---

#### Q3: 训练框是固定核大小 6 倍吗？

| 阶段 | 框来源 | 扩展 |
|------|--------|------|
| **训练** | GT Mask 边界 | +5px padding |
| **推理** | DAPI 核检测 | **6x 核大小** (hardcoded) |

6x 是经验值，来源于 `create_bounding_boxes(expansion_factor=6.0)`。

---

#### Q4: 动态 Box 指什么？

**Box 扰动增强** (`_augment_box()`) 作用于**训练阶段**：
- 对 GT 框随机扰动 ±30%
- 帮助模型适应推理时的不精确框

**不影响**推理时的 DAPI 检测。

---

#### Q5: CellFinder 如何生成框？

CellFinder 使用 3D 卷积网络直接**预测细胞中心点**，然后根据预设尺寸生成框。不适合心肌细胞（设计目标是神经元）。

---

#### Q6: 当前 DAPI 检测有 Actn2 过滤吗？

**已实现**: `filter_by_actn2()` 函数

```python
# 过滤逻辑:
# 1. 检测 Actn2 阳性区域 (>10% 信号强度)
# 2. 每个核周围 50% 区域检查 Actn2 覆盖
# 3. 覆盖率 >= 30% 才保留

cardiomyocyte_nuclei = filter_by_actn2(all_nuclei, actn2_channel)
```

**效果**: 过滤非心肌细胞核（成纤维细胞等）

---

### 2026-01-14: 核轴向与扩展策略

#### Q1: 核的长短轴与心肌细胞的长短轴有关联吗？

**有关联**，但不是完全一致。

| 指标 | 关联程度 |
|------|---------|
| 核长轴方向 ≈ 细胞长轴方向 | ~80% 相关 |
| 核短轴方向 ≈ 细胞短轴方向 | ~80% 相关 |

**生物学依据**:
- 心肌细胞在收缩方向排列（沿肌节方向）
- 细胞核通常与细胞长轴大致对齐
- 但存在多核细胞和不规则排列的例外

**相关文献**: 暂无直接专门论文，但心肌细胞形态学研究普遍认可这一假设。

---

#### Q2: 为什么移除 30% 缩小逻辑？

**问题**: GT 框已经是 mask 的最小外接矩形，缩小后 mask 会超出框。

```
GT状态:
┌─────────────┐
│ ██████████  │ <- mask 恰好填满框
└─────────────┘

缩小后:
  ┌─────────┐
  │█████████│█ <- mask 超出框！
  └─────────┘
```

**SAM 无法分割框外区域**，所以训练时只能扩展，不能缩小。

**已修复**: `_augment_box()` 现在只扩展 0-30%，不再缩小。

---

#### Q3: 各向异性扩展实现

**设计**:
| 方向 | 扩展倍数 |
|------|---------|
| 长轴 (major axis) | **5.0x** |
| 短轴 (minor axis) | **3.0x** |

**使用核的 `orientation` 和 `major_axis_length` 判断方向**:
```python
if abs(orientation) < π/4:  # 核倾向水平
    new_w = box_w * 5.0  # 宽度扩展更多
    new_h = box_h * 3.0
else:  # 核倾向垂直
    new_h = box_h * 5.0  # 高度扩展更多
    new_w = box_w * 3.0
```

---

### 2026-01-14: 核-细胞轴向对齐统计

#### 统计结果 (50样本, 593细胞)

| 指标 | 值 |
|------|-----|
| 平均角度误差 | 35.2° |
| 中位角度误差 | 30.0° |
| P75 角度误差 | 55.0° |
| P90 角度误差 | 76.1° |

#### 对齐率

| 阈值 | 对齐率 |
|------|--------|
| ≤15° | **30.9%** |
| ≤30° | **50.1%** |
| ≤45° | **66.4%** |

**结论**: 之前估计的 80% 对齐率是高估的，实际只有 ~50% 在 30° 以内。

---

#### 智能扩展策略 (已实现)

**策略**: 根据核形状动态选择扩展方式

```python
if is_binucleated:
    # 双核: 使用合并框的长宽比判断方向
    if combined_aspect < 1.3:
        # 合并后形状圆 → 等比例
        new_h, new_w = box_h * 4.0, box_w * 4.0
    else:
        # 合并后有方向 → 各向异性
        ...
else:
    # 单核
    if nuc_aspect < 1.3:
        # 圆核 → 等比例 (方向不可靠)
        new_h, new_w = box_h * 4.0, box_w * 4.0
    else:
        # 椭圆核 → 各向异性
        expansion_long = 5.0
        expansion_short = 3.0
```

**参数**:
| 参数 | 值 | 用途 |
|------|-----|------|
| `expansion_long` | 5.0 | 长轴扩展 |
| `expansion_short` | 3.0 | 短轴扩展 |
| `expansion_isotropic` | 4.0 | 圆核等比例扩展 |
| `round_threshold` | 1.3 | 圆核判定阈值 |

---

### 2026-01-15: 统一多通道数据管道

#### 问题分析

之前的数据流存在不一致：
- DAPI 检测：使用原始 TIFF (1736×1776)
- SAM 训练：使用 processed NPY (仅 BF, 未 resize)

#### 新数据管道设计

**输出格式**:
```
data/processed/
├── images/*.npy   → shape (3, 1024, 1024)  # [BF, DAPI, Actn2]
└── masks/*.npy    → shape (1024, 1024)     # Instance mask
```

**通道分配**:
| 通道 | 内容 | 用途 |
|------|------|------|
| Ch0 | Brightfield | SAM 主输入 |
| Ch1 | DAPI | 核定位参考 |
| Ch2 | Actn2 | 肌节辅助 |

**Mask**: 单独作为训练 target，不占输入通道

#### DAPI 检测更新

移除 `binary_opening` 步骤，更新尺寸过滤参数：

| 参数 | 旧值 | 新值 | 依据 |
|------|------|------|------|
| min_nucleus_area | 500 | **240** | P5 统计 |
| max_nucleus_area | 15000 | **21000** | P95 统计 |
| relative_threshold | 0.2 | **移除** | 固定尺寸数据集无需 |
| binary_opening | Yes | **移除** | DAPI 信号清晰 |

#### 变更文件

- `data/scripts/extract_expanded_pairs.py` - 重写为多通道输出
- `anti_test/visualize_test_results.py` - 更新 `detect_nuclei_dapi()`
- 删除 `data/fixed_50/` 文件夹

---

### 2026-01-15: 用户 Napari 观察与导师反馈

#### 用户观察结论

| 观察 | 结论 |
|------|------|
| Actn2 亮的连续肌节信号 | 部分细胞可作为边界参考 |
| 不是所有细胞都有清晰边界 | Actn2 不能作为唯一依据 |
| 相邻心肌细胞 | Actn2 可辅助区分边界 |

#### 导师反馈

> **重要**: GT 分割存在漏标情况，预测多出的个别细胞可能是真实结果。

**评估建议**:
- Pred > GT 少量时，检查是否为真实细胞
- 添加人工复核环节

#### 分割策略建议

| 信号 | 用途 | 优先级 |
|------|------|--------|
| DAPI | 定位核位置 | 主要 |
| Brightfield | 学习细胞形态 | 主要 |
| Actn2 | 辅助边界判定 | 辅助 |

#### 实验计划

| 实验 | 输入 | 优先级 |
|------|------|--------|
| A | BF × 3 (基线) | P1 |
| **B** | **BF + DAPI + Actn2** | **P0** |
| C | 加权 Actn2 融合 | P2 |

---

### 2026-01-15: 用户 Napari 观察反馈

#### DAPI 检测参数最终值

| 参数 | 值 | 说明 |
|------|-----|------|
| min_nucleus_area | **500** | 过滤极小噪点 |
| max_nucleus_area | **30000** | 包含大核 |
| binary_opening | **disk(3)** | 去除 <28 px² 噪点 |

#### 用户观察结论

| 观察 | 处理方式 |
|------|---------|
| 边缘不完整核 | ✅ `exclude_edges=True` |
| 非 Actn2 区域核 | ✅ `filter_by_actn2()` |
| 靠近的核 → 一个细胞 | ✅ `merge_close_nuclei()` |
| GT 边缘截断细胞未标注 | 评估时允许 Pred > GT |
| GT 存在漏检 | 评估时允许 Pred > GT |

#### 完整 DAPI 检测管线

```python
def dapi_detect_cells(dapi_channel, image_shape, actn2_channel=None):
    # 1. 核检测
    regions = detect_nuclei_dapi(dapi_channel)  # min=500, max=30000
    
    # 2. Actn2 过滤 (可选)
    if actn2_channel:
        regions = filter_by_actn2(regions, actn2_channel)
    
    # 3. 靠近核合并
    cell_groups = merge_close_nuclei(regions)
    
    # 4. 创建框 (排除边缘)
    boxes = create_bounding_boxes(cell_groups, image_shape)  # exclude_edges=True
    
    return boxes
```

---

### 2026-01-15: Q&A 记录与待办任务

#### Q1: Mean Dice 是像素级还是实例级？

**澄清**：当前 `visualize_test_results.py` 输出的 Mean Dice 是**像素级指标**。

| 指标类型 | 说明 | 实现状态 |
|----------|------|----------|
| **像素级 Dice** | 整张图预测与 GT 的像素重叠 | ✅ 已实现 |
| 实例级 Dice | 每个细胞实例单独计算 Dice，再平均 | ⚠️ 未集成到推理 |
| 实例级 PQ/AJI | 考虑检测 + 分割质量 | ⚠️ 未集成到推理 |

**结论**：需要更新 `visualize_test_results.py` 以输出真正的实例级指标。

---

#### Q2: regionprops 是什么？

`regionprops` 是 scikit-image 的函数，用于从标签图提取区域属性：

```python
from skimage import measure

# 输入: GT mask (每个值代表一个细胞实例)
# 输出: 每个区域的属性列表
for region in measure.regionprops(gt_mask):
    y1, x1, y2, x2 = region.bbox  # 边界框
    area = region.area            # 面积
    centroid = region.centroid    # 中心点
```

**训练时**：从 GT Mask 用 regionprops 提取每个细胞的边界框作为 Prompt。
**推理时**：从 DAPI 检测结果生成 Prompt（不依赖 GT）。

---

#### Q3: 推理使用的模型

**当前 `visualize_test_results.py` 使用的默认模型**:

```python
MODEL_PATH = "checkpoints/boundary_20260111_012636/best_model.pt"  # E12 边界损失模型
```

**E12 边界模型**：
- 来源：实验 E12 边界损失微调
- 指标：PQ↑265%, Dice↑8%
- 日期：2026-01-11

**E15b 多通道模型**：
- 位置：`checkpoints/base_20260115_021255/best_model.pt`
- 指标：Val Dice 0.6588
- **尚未集成到推理脚本**

---

#### 今日已完成任务 ✅

| 任务 | 说明 |
|------|------|
| E15a 基线训练 | BF×3, Val Dice 0.6472 |
| E15b 多通道训练 | BF+DAPI+Actn2, Val Dice 0.6588 |
| DAPI 检测参数更新 | min=500, max=30000 |
| RGB 顺序实验设计 | 记录到 implementation_plan |
| 通道注意力方案设计 | 方案 B/C 详细代码 |

---

#### 待办任务 (Pending Tasks)

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 添加早停到 train.py | 从 train_expanded.py 移植 |
| **P0** | 集成 E15b 模型到推理 | 替换默认 MODEL_PATH |
| **P0** | 集成实例级指标到推理 | 使用 eval_metrics.py |
| P1 | 通道权重学习 (方案B) | 实现并验证 |
| P1 | LR 调参实验 | 1e-5, 5e-5, 1e-4 |
| P2 | 通道消融 A2, A3 | BF+DAPI, BF+Actn2 |
| P2 | RGB 顺序验证 O1-O3 | [BF,DAPI,Actn2] vs 其他顺序 |
| P3 | SE 通道注意力 (方案C) | 更复杂的通道注意力 |

---

#### 关键结论

1. **Mean Dice 0.7468** 是像素级，不是实例级
2. **推理使用 E12 模型**，未使用今天训练的 E15b
3. **regionprops** 用于从 GT Mask 提取边界框
4. 所有待办任务已记录

