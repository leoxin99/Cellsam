# CellSAM 项目完整指南

> **目的**: 帮助理解整个项目流程
> **更新**: 2026-02-24
> **SSOT 引用**: 参数/评估口径以各 SSOT 文档为准，此文档为导航指南

---

## 一、项目概述

**任务**: 自动分割 hiPSC-CM (人诱导多能干细胞衍生心肌细胞) 的细胞边界

**输入**: 显微镜图像 (3 通道)
- **BF (Brightfield)**: 明场图像，显示细胞形态
- **DAPI**: 细胞核蓝色荧光染色，定位核心位置
- **Actn2**: α-肌动蛋白绿色荧光染色，显示 Z-线结构

**输出**: 实例分割 mask (每个细胞一个唯一 ID)

```
┌────────────────────────────────────────────────────┐
│                 CellSAM Pipeline                    │
├────────────────────────────────────────────────────┤
│ [输入图像] → [核检测] → [框生成] → [SAM分割] → [后处理] │
│   3通道       DAPI      Box       Mask     边界平滑  │
└────────────────────────────────────────────────────┘
```

---

## 二、数据

### 数据结构
```
data/
├── raw/allen_segmented_fields_full/   # 原始 478 张 TIFF
├── processed/                          # 预处理后 .npy 文件
│   ├── SAMPLE_ID_image.npy            # (H, W, 3) 图像
│   └── SAMPLE_ID_mask.npy             # (H, W) GT mask
└── splits/                             # 固定数据划分
    ├── train_ids.txt                   # 334 样本
    ├── val_ids.txt                     # 71 样本
    └── test_ids.txt                    # 73 样本
```

### 通道映射 (SemanticChannelMapper)

| 输入索引 | 通道 | 伪 RGB 映射 | 预处理 |
|:--------:|------|:-----------:|--------|
| 0 | BF | **R** | CLAHE 增强 (clip=2.0) |
| 1 | DAPI | **B** (蓝色荧光) | 高斯平滑 (σ=1.5) |
| 2 | Actn2 | **G** (绿色荧光) | P1-P99 百分位截断 |

> 2ch 模式 (`use_2ch: true`): B 通道为 BF 副本 (无 DAPI)

---

## 三、核检测

检测参数已锁定 (E34/T3b)，SSOT: `src/detection/profiles.py`

| 参数 | DAPI | Adaptive |
|------|:----:|:--------:|
| `min_nucleus_area` | 1500 | 1500 |
| `max_nucleus_area` | 20000 | 20000 |
| `edge_margin` | 20 | 20 |
| `merge_coeff` | 1.4 | 1.4 |
| `search_radius` | — | 160 |

**test73 封板**: DAPI F1=**0.803**, Adaptive F1=0.750

---

## 四、训练

### 当前最优配置 (Best Config)
```yaml
# src/config/best_config.yaml
loss:
  pos_weight: 10.0       # Ab-5: +4.1pp PQ
  use_contour: false     # Ab-2: +2.3pp PQ
  boundary_weight: 1.5
  use_boundary: true
  use_aji: true
  aji_weight: 0.2
training:
  epochs: 80
  learning_rate: 0.0001
  early_stop_patience: 15
  use_pq_early_stop: true
model:
  freeze_encoder: true
  freeze_decoder: false
```

### 训练命令
```bash
conda activate cellsam
# 本地
python src/train.py --config src/config/best_config.yaml --seed 42
# ALICE
sbatch scripts/train_best_config.sh
```

### 三通道训练 (T18)
```bash
python src/train.py --config src/config/t18b_3ch.yaml --seed 42
```
配置: `t18a_2ch.yaml` | `t18b_3ch.yaml` | `t18c_3ch_no_adapter.yaml`

---

## 五、评估

### 指标

| 指标 | Best Config (val) | Phase1 (test73) | 目标 |
|------|:-----------------:|:---------------:|:----:|
| **PQ@0.5** | 0.484 | 0.464 | ≥0.48 ✅ |
| **BM-Dice** | 0.720 | 0.695 | ≥0.70 ✅ |
| **AJI** | 0.570 | 0.519 | ≥0.50 ✅ |

### 评估命令
```bash
# Oracle 评估 (GT box)
# [已归档] comprehensive_eval.py  tools/archive/, 被 eval_ablation.py 取代\npython tools/eval_ablation.py --exp-dir checkpoints/xxx/ --output experiments/ablation_eval/
# E2E 评估
python tools/evaluate_e2e.py --model checkpoints/xxx/best_model.pt
```

---

## 六、核心代码

| 模块 | 文件 | 关键类/函数 |
|------|------|------------|
| 数据加载 | `src/augmented_dataset.py` | `AugmentedAllenDataset`, `SemanticChannelMapper` |
| 训练入口 | `src/train.py` | `main()` |
| 损失函数 | `src/losses/combined.py` | `CombinedLoss` |
| 通道适配器 | `src/adapters/channel_adapter.py` | `IndependentChannelAdapter` |
| DAPI 检测 | `src/detection/dapi.py` | `detect_nuclei()` |
| 检测参数 | `src/detection/profiles.py` | `DETECTION_PROFILES` |
| 推理核心 | `src/inference/core.py` | `segment_with_boxes()` |

---

## 七、关键文档

| 文档 | 内容 |
|------|------|
| `CLAUDE.md` | **项目总览入口** (AI 必读) |
| `docs/error_log_and_checklist.md` | ⚠️ **训练前必读** — 错误归纳 + 检查清单 |
| `docs/experiments_log.md` | 完整实验历史 |
| `docs/inference_standard.md` | 推理与评估标准 SSOT |
| `docs/dataset_parameters.md` | 数据集统计参数 |
| `docs/alice_quick_reference.md` | ALICE 集群参考 |

---

*此文档由 AI 助手维护，引用 SSOT 文件获取最新参数*
