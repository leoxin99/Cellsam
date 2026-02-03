# CellSAM 项目完整指南

> **目的**: 帮助理解整个项目流程
> **更新**: 2026-02-03
> **位置**: docs/project_guide.md

---

## 一、项目概述

### 1.1 问题定义

**任务**: 自动分割 hiPSC-CM (人诱导多能干细胞衍生心肌细胞) 的细胞边界

**输入**: 显微镜图像 (3 通道)
- **BF (Brightfield)**: 明场图像，显示细胞形态
- **DAPI**: 细胞核染色，定位核心位置
- **Actn2**: α-肌动蛋白染色，显示 Z-线结构

**输出**: 实例分割 mask (每个细胞一个唯一 ID)

### 1.2 技术方案

```
┌─────────────────────────────────────────────────────────────┐
│                    CellSAM Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [输入图像] → [核检测] → [框生成] → [SAM分割] → [后处理]   │
│     3通道      DAPI       Box        Mask      边界平滑     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、数据流程

### 2.1 数据结构

```
data/
├── raw/allen_segmented_fields_full/   # 原始 478 张 TIFF
├── processed/                          # 预处理后的 .npy 文件
│   ├── SAMPLE_ID_image.npy            # (H, W, 3) 图像
│   └── SAMPLE_ID_mask.npy             # (H, W) GT mask
└── splits/                             # 固定数据划分
    ├── train_ids.txt                   # 400 样本
    ├── val_ids.txt                     # 验证
    └── test_ids.txt                    # 78 样本
```

### 2.2 通道映射

| 索引 | 通道 | 用途 |
|------|------|------|
| 0 | BF | 细胞形态 → SAM 主输入 |
| 1 | DAPI | 核检测 → 定位 |
| 2 | Actn2 | Z-线检测 → 框优化 |

### 2.3 数据加载代码

```python
# src/augmented_dataset.py
from augmented_dataset import AugmentedAllenDataset

dataset = AugmentedAllenDataset(
    data_dir="data/processed",
    is_training=True,
    sample_ids=train_ids,
    use_bf_only=True,         # 只用 BF 通道
    use_semantic_mapping=False # 不用语义映射
)

sample = dataset[0]
# sample['image']  → (3, 1024, 1024) 图像
# sample['mask']   → (1024, 1024) GT mask
# sample['boxes']  → (N, 4) 边界框
```

---

## 三、核检测模块

### 3.1 核心检测函数

```python
# src/detection/dapi.py
from detection.dapi import detect_nuclei

boxes, (labeled, nuclei_props) = detect_nuclei(
    dapi_image,           # (H, W) DAPI 通道
    min_area=3000,        # 最小核面积 (像素)
    edge_distance=100     # 距边缘距离阈值
)
# boxes: List[(x1, y1, x2, y2)]
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_area` | 3000 | 过滤碎屑 |
| `edge_distance` | 100 | 排除边缘不完整细胞 |
| `merge_threshold` | 1.2× 直径 | 双核合并距离 |

---

## 四、训练流程

### 4.1 训练入口

```bash
# 激活环境
conda activate cellsam

# 本地训练
python src/train.py --config src/config/bf_baseline_v2.yaml

# ALICE 集群训练
sbatch scripts/train_ablation_v2.sh
```

### 4.2 配置文件结构

```yaml
# src/config/bf_baseline_v2.yaml
data:
  splits_dir: "data/splits"
  processed_data_dir: "data/processed"
  target_size: [1024, 1024]
  use_bf_only: true

model:
  freeze_encoder: true      # 冻结 ViT
  freeze_decoder: false     # 训练 Decoder

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001

loss:
  boundary_weight: 0.3
  box_expand: 0.1

output:
  experiment_name: "bf_baseline_v2"
```

### 4.3 训练前验证 (必须执行!)

```bash
python tools/verify_training_config.py
```

---

## 五、评估方法

### 5.1 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| **Dice** | 像素重叠度 | >0.75 |
| **PQ@0.5** | 实例匹配质量 | >0.5 |
| **AJI** | 聚合 Jaccard | >0.4 |

### 5.2 评估命令

```bash
python tools/comprehensive_eval.py \
    --model checkpoints/xxx_best.pt \
    --config src/config/bf_baseline_v2.yaml
```

---

## 六、关键代码文件

### 核心模块

| 文件 | 路径 | 功能 |
|------|------|------|
| 数据加载 | `src/augmented_dataset.py` | `AugmentedAllenDataset` |
| 训练入口 | `src/train.py` | `main()` |
| 损失函数 | `src/losses/combined.py` | `CombinedLoss` |
| DAPI 检测 | `src/detection/dapi.py` | `detect_nuclei()` |

### 评估工具

| 文件 | 功能 |
|------|------|
| `tools/comprehensive_eval.py` | 全面指标评估 |
| `tools/evaluate_e2e.py` | 端到端评估 |
| `tools/verify_training_config.py` | 训练前验证 |

---

## 七、常用命令速查

```bash
# 环境激活
conda activate cellsam

# 本地训练
python src/train.py --config src/config/xxx.yaml

# ALICE 训练
ssh s3890074@login.alice.universiteitleiden.nl
sbatch scripts/train_ablation_v2.sh
squeue -u s3890074

# 评估
python tools/comprehensive_eval.py --model checkpoints/xxx.pt
```

---

## 八、关键文档

| 文档 | 内容 |
|------|------|
| `CLAUDE.md` | 项目总览入口 |
| `docs/error_log_and_checklist.md` | 错误归纳 + 检查清单 |
| `docs/experiments_log.md` | 完整实验历史 |
| `docs/alice_quick_reference.md` | ALICE 集群参考 |
