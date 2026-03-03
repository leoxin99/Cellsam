# CellSAM 三项优化实施方案

> **创建日期**: 2026-01-30
> **状态**: 待执行
> **预计总耗时**: 3-4 天 (含 10 小时训练时间)

---

## 摘要

本文档详细设计了 CellSAM 项目的三项关键优化：
1. **4-way 消融实验** - 验证各组件的真实贡献
2. **多指标评估体系** - 全面评估分割质量
3. **边界数据增强** - 提升边界分割精度

---

## 一、消融实验设计 (Ablation Study)

### 1.1 实验矩阵

| 实验 | 名称 | Semantic | Adapter | 预期效果 |
|:----:|------|:--------:|:-------:|---------|
| **Exp1** | BF Baseline | ❌ | ❌ | 基线 (Dice=0.7718) |
| **Exp2** | BF + Adapter | ❌ | ✅ | Adapter 单独效果 |
| **Exp3** | Semantic Only | ✅ | ❌ | 语义映射效果 |
| **Exp4** | Full Pipeline | ✅ | ✅ | 完整方案 |

### 1.2 配置文件

需创建 4 个配置文件在 `src/config/ablation/`:

#### `exp1_bf_baseline.yaml`
```yaml
data:
  use_bf_only: true
  use_semantic_mapping: false
model:
  use_adapter: false
training:
  seed: 42
output:
  experiment_name: "exp1_bf_baseline"
```

#### `exp2_bf_adapter.yaml`
```yaml
data:
  use_bf_only: true
  use_semantic_mapping: false
model:
  use_adapter: true
  adapter:
    type: "independent"
    kernel_size: 3
training:
  seed: 42
output:
  experiment_name: "exp2_bf_adapter"
```

#### `exp3_semantic_only.yaml`
```yaml
data:
  use_bf_only: false
  use_semantic_mapping: true
model:
  use_adapter: false
training:
  seed: 42
output:
  experiment_name: "exp3_semantic_only"
```

#### `exp4_full.yaml`
```yaml
data:
  use_bf_only: false
  use_semantic_mapping: true
model:
  use_adapter: true
  adapter:
    type: "independent"
    kernel_size: 3
training:
  seed: 42
output:
  experiment_name: "exp4_full"
```

### 1.3 批量训练脚本

**文件**: `scripts/run_ablation.py`

核心功能:
- 顺序运行 4 组实验
- 记录训练日志和最终指标
- 生成 Markdown 报告和 JSON 结果
- 支持 `--skip` 参数跳过已完成实验

### 1.4 结果分析脚本

**文件**: `scripts/analyze_ablation.py`

核心功能:
- 计算各组件效应 (Main Effect, Interaction Effect)
- 生成对比柱状图和 2×2 热力图
- 输出 LaTeX 表格

### 1.5 train.py 修改

添加随机种子设置:
```python
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
```

### 1.6 预计耗时

| 阶段 | 时间 |
|------|------|
| 配置文件创建 | 0.5 小时 |
| 脚本开发 | 2 小时 |
| 实验运行 | **10 小时** (可挂机) |
| 结果分析 | 1 小时 |

---

## 二、多指标评估体系

### 2.1 指标定义

| 层级 | 指标 | 定义 | 现状 |
|------|------|------|:----:|
| **Tier 1** | Detection F1 | TP/(TP+0.5FP+0.5FN) | ✅ |
| **Tier 2** | PQ@0.5 | SQ × RQ | ✅ |
| **Tier 2** | AJI | Σ∩/Σ∪ | ✅ |
| **Tier 2** | Instance Dice | 匹配实例平均 Dice | 🆕 |
| **Tier 3** | Boundary IoU | 边界区域 IoU | ✅ |
| **Tier 3** | Boundary F1 | 边界像素 F1 | 🆕 |
| **Tier 3** | HD95 | 95% Hausdorff 距离 | ✅ |
| **Tier 4** | Pixel Dice | 全局像素 Dice | ✅ |

### 2.2 代码结构

```
src/evaluation/
├── __init__.py
├── metrics.py          # InstanceSegmentationMetrics 类
└── visualization.py    # 图表生成
```

### 2.3 核心类设计

```python
class InstanceSegmentationMetrics:
    def __init__(self, iou_thresholds=[0.3, 0.5, 0.7]):
        ...

    # Tier 1
    def compute_detection_metrics(self, pred, gt, thresh) -> dict

    # Tier 2
    def compute_panoptic_quality(self, pred, gt, thresh) -> dict
    def compute_aji(self, pred, gt) -> float
    def compute_instance_dice(self, pred, gt, thresh) -> float

    # Tier 3
    def compute_boundary_f1(self, pred, gt, tolerance=2) -> float
    def compute_boundary_iou(self, pred, gt, dilation=2) -> float
    def compute_hd95(self, pred, gt) -> float

    # Tier 4
    def compute_pixel_dice(self, pred, gt) -> float

    # 统一接口
    def evaluate(self, pred, gt) -> dict
```

### 2.4 评估脚本

**文件**: `scripts/evaluate_model.py`

用法:
```bash
# 单模型评估
python scripts/evaluate_model.py --checkpoint best.pt --split test

# 多模型对比
python scripts/evaluate_model.py --compare exp1.pt exp2.pt exp3.pt exp4.pt
```

### 2.5 可视化输出

- 多模型指标对比柱状图
- 雷达图对比
- LaTeX 表格 (论文用)

---

## 三、边界数据增强

### 3.1 增强方法

| 方法 | 目的 | 参数 | 对图像 | 对 Mask |
|------|------|------|:------:|:-------:|
| **Elastic Deformation** | 模拟细胞变形 | α=120, σ=12 | ✅ | ✅ |
| **Boundary Perturbation** | 模拟标注不确定性 | shift=5px | ❌ | ✅ |
| **Morphological Noise** | 边界局部变化 | kernel=3 | ❌ | ✅ |

### 3.2 代码结构

```
src/augmentations/
├── __init__.py
└── boundary_augment.py
    ├── ElasticDeformation
    ├── BoundaryPerturbation
    ├── MorphologicalNoise
    └── BoundaryAugmentationPipeline
```

### 3.3 核心实现

#### ElasticDeformation
```python
class ElasticDeformation:
    def __init__(self, alpha=120, sigma=12, p=0.3):
        ...

    def __call__(self, image, mask):
        # 生成随机位移场
        dx = gaussian_filter(random, sigma) * alpha
        dy = gaussian_filter(random, sigma) * alpha
        # 同步变形图像和掩膜
        return deformed_image, deformed_mask
```

#### BoundaryPerturbation
```python
class BoundaryPerturbation:
    def __init__(self, max_shift=5, p=0.3):
        ...

    def __call__(self, mask):
        # 随机膨胀/腐蚀每个实例
        for inst_id in unique_ids:
            operation = random.choice(['dilate', 'erode', 'both'])
            ...
        return perturbed_mask
```

### 3.4 配置整合

更新 `semantic_adapter.yaml`:
```yaml
data:
  boundary_augmentation:
    enabled: true
    elastic:
      alpha: 120
      sigma: 12
      p: 0.3
    perturbation:
      max_shift: 5
      p: 0.3
    morphological:
      kernel_size: 3
      p: 0.2
```

### 3.5 验证脚本

**文件**: `scripts/verify_boundary_augmentation.py`

生成增强前后对比图，验证参数设置合理性。

---

## 四、实施计划

### 4.1 优先级排序

| 优先级 | 任务 | 依赖 | 时间 |
|:------:|------|------|------|
| **P0** | 多指标评估体系 | 无 | 1 天 |
| **P1** | 消融实验框架 | 评估体系 | 0.5 天 |
| **P2** | 运行消融实验 | 框架 | 10 小时 |
| **P3** | 边界数据增强 | 无 | 1 天 |
| **P4** | 增强效果验证 | 增强 | 0.5 天 |

### 4.2 依赖关系

```
多指标评估体系 (P0)
        │
        ▼
消融实验框架 (P1)
        │
        ▼
运行消融实验 (P2) ───► 结果分析与报告

边界数据增强 (P3) ───► 增强效果验证 (P4)
                              │
                              ▼
                       集成到训练 Pipeline
```

### 4.3 时间线

| 天数 | 任务 |
|------|------|
| **Day 1** | 创建 `src/evaluation/metrics.py`<br>创建 `scripts/evaluate_model.py`<br>验证评估指标正确性 |
| **Day 2** | 创建消融实验配置文件<br>创建 `scripts/run_ablation.py`<br>修改 `train.py` 添加随机种子 |
| **Day 2-3** | 运行 4 组消融实验 (挂机) |
| **Day 3** | 创建 `src/augmentations/boundary_augment.py`<br>集成到 `augmented_dataset.py`<br>验证增强效果 |
| **Day 4** | 分析消融实验结果<br>生成论文表格和图表<br>更新 `CLAUDE.md` |

### 4.4 验收标准

| 任务 | 验收标准 |
|------|---------|
| **评估体系** | `evaluate_model.py` 正常运行，输出所有指标 |
| **消融实验** | 4 组实验完成，生成对比表格和效应分析 |
| **边界增强** | 可视化验证效果合理，训练收敛正常 |
| **文档** | CLAUDE.md 更新，实验日志完整 |

---

## 五、关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/train.py` | 修改 | 添加随机种子、边界增强配置 |
| `src/augmented_dataset.py` | 修改 | 集成 BoundaryAugmentationPipeline |
| `src/evaluation/metrics.py` | 新建 | 统一评估指标 |
| `src/augmentations/boundary_augment.py` | 新建 | 边界增强 |
| `scripts/run_ablation.py` | 新建 | 消融实验批量运行 |
| `scripts/evaluate_model.py` | 新建 | 模型评估脚本 |
| `scripts/analyze_ablation.py` | 新建 | 消融结果分析 |
| `src/config/ablation/*.yaml` | 新建 | 4 个实验配置 |

---

## 六、预期成果

### 消融实验报告示例

| 实验 | 配置 | Dice | 较基线 |
|------|------|------|--------|
| Exp1 | BF Baseline | 0.7718 | - |
| Exp2 | BF + Adapter | 0.78xx | +0.01 |
| Exp3 | Semantic Only | 0.79xx | +0.02 |
| Exp4 | Full | 0.81xx | +0.04 |

### 效应分析示例

- **Semantic 主效应**: +0.025 (主要贡献)
- **Adapter 主效应**: +0.015
- **交互效应**: +0.005 (正向协同)

### 论文表格示例

| Model | PQ@0.5 | AJI | Instance Dice | Boundary F1 |
|-------|--------|-----|---------------|-------------|
| BF Baseline | 0.52 | 0.58 | 0.72 | 0.65 |
| Full Pipeline | **0.61** | **0.67** | **0.79** | **0.74** |

---

**等待审批后执行。**
