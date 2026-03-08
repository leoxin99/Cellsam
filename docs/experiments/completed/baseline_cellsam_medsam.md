# Baseline Evaluation: CellSAM / MedSAM (Pre-training, No Fine-tuning)

## 1. Metadata
- ID: `Baseline`
- Status: `Completed`
- Owner: `A2`
- Priority: `P1`
- Related result files:
  - `experiments/baseline_comparison/results.json`
  - `experiments/baseline_comparison/results_combined.json`
  - `experiments/t34_official_path_ablation/results_val.json` (Arm A = CellSAM unified baseline)
  - `experiments/t27a_eval/results.json` (T27a GT-box eval as upper bound)

## 2. Background

CellSAM 和 MedSAM 使用预训练权重（无针对心肌细胞数据微调）在 test73 / val71 上的基线性能。
所有评估使用 GT boxes (oracle detection)，IoU threshold = 0.5。

## 3. Results

### 3.1 MedSAM (test73, GT boxes)

| Metric | Value |
|--------|:-----:|
| PQ | **0.576** |
| SQ | 0.685 |
| RQ | 0.840 |
| F1 (micro) | **0.834** |
| Precision | 0.834 |
| Recall | 0.834 |
| BM-1to1 Dice | **0.771** |
| AJI | 0.634 |
| Semantic Dice | 0.862 |
| TP / FP / FN | 609 / 121 / 121 |

> MedSAM: ViT-B backbone, 预训练在 100 万+ 医学图像对上。Oracle detection (GT boxes)。

### 3.2 CellSAM Unified Path (val71, GT boxes) — via T34 Arm A

| Metric | Value |
|--------|:-----:|
| PQ | **0.491** |
| SQ | 0.606 |
| RQ | 0.811 |
| F1 (micro) | **0.798** |
| Precision | 0.798 |
| Recall | 0.798 |
| BM-1to1 Dice | **0.723** |
| AJI | 0.570 |
| TP / FP / FN | 595 / 151 / 151 |

### 3.3 CellSAM Official Path (val71, GT boxes) — via T34 Arm C

| Metric | Value |
|--------|:-----:|
| PQ | **0.630** |
| SQ | 0.674 |
| RQ | 0.934 |
| F1 (micro) | **0.932** |
| Precision | 0.933 |
| Recall | 0.930 |
| BM-1to1 Dice | **0.783** |
| AJI | 0.638 |
| TP / FP / FN | 694 / 50 / 52 |

### 3.4 Baseline Summary Table

| Method | Split | PQ | F1 | BM-Dice | AJI |
|--------|:-----:|:---:|:---:|:------:|:---:|
| MedSAM (pretrained) | test73 | 0.576 | 0.834 | 0.771 | 0.634 |
| CellSAM unified | val71 | 0.491 | 0.798 | 0.723 | 0.570 |
| CellSAM official | val71 | 0.630 | 0.932 | 0.783 | 0.638 |

> [!IMPORTANT]
> MedSAM (test73) vs CellSAM (val71) 是不同 split，不可直接比较。CellSAM official path 显著优于 unified path，差异来源见 T34 实验文档。

## 4. IoU Threshold

所有评估使用 `compute_all_metrics(pred, gt, iou_threshold=0.5)`:
- PQ/SQ/RQ: 基于 Hungarian matching, IoU ≥ 0.5 的 matched pairs
- F1/P/R: 同上 matching 产生的 TP/FP/FN
- BM-Dice/AJI: IoU-independent (基于最佳匹配或交并比)
