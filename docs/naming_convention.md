# CellSAM 项目命名规范

## 概述

> **状态**: 🟢 Active — 命名规范参考文档
> **最后更新**: 2026-02-13

本文档定义了 CellSAM 项目中模型、实验和方案的标准命名规范，避免混淆。

---

## 模型命名

### 基础模型层级

| 缩写 | 全称 | 说明 |
|------|------|------|
| **SAM** | Segment Anything Model | Meta 原始通用分割模型 |
| **CellSAM** | Cell Segment Anything Model | 基于 SAM 的细胞分割模型 (已用生物数据微调) |
| **CM-CellSAM** | Cardiomyocyte-CellSAM | 我们用心肌细胞(CM)数据微调的 CellSAM |

### 模型变体

| 模型名 | 说明 |
|--------|------|
| **CM-CellSAM-BF** | 只使用明场通道的心肌细胞微调模型 |
| **CM-CellSAM-Adapter** | 使用 Channel Adapter 的多通道模型 |
| **CM-CellSAM-Semantic** | 使用语义通道映射的模型 |

---

## 检测方案命名

### 框生成方案

| 方案名 | 缩写 | 说明 |
|--------|------|------|
| **GT-Box** | GTB | 使用 Ground Truth 框 (评测标准) |
| **DAPI-Box** | DPB | 基于 DAPI 核检测生成框 |
| **ZLine-Box** | ZLB | 基于 Z 线 (Actn2) 检测生成框 |
| **CellFinder-Box** | CFB | 使用 CellFinder 自动检测生成框 |

### 检测方案对比表

| 方案 | 输入通道 | 检测目标 | 特点 |
|------|---------|---------|------|
| **GTB** | GT Mask | 细胞边界 | 用于纯分割性能评估 |
| **DPB** | DAPI (Ch4) | 细胞核 | 初始方案，核心定位 |
| **ZLB** | Actn2 (Ch1) | Z 线纹理 | 替代方案，肌节定位 |
| **CFB** | BF/多通道 | 自动检测 | 端到端方案 |

---

## 实验命名规范

### 格式

```
E{编号}_{模型类型}_{阶段}_{日期}
```

### 示例

| 实验 ID | 含义 |
|---------|------|
| **E29_BF_P1** | 第29号实验，BF 通道，Phase 1 (基础训练) |
| **E30_Adapter_P1** | 第30号实验，Adapter 方案，Phase 1 |
| **E31_BF_P2** | 第31号实验，BF 通道，Phase 2 (高级损失) |
| **E32_Adapter_P2** | 第32号实验，Adapter 方案，Phase 2 |

### 阶段定义

| 阶段 | 说明 | 主要特征 |
|------|------|---------|
| **P1** | Phase 1 - 基础训练 | Dice + BCE + Boundary + AJI |
| **P2-A** | Phase 2-A - 邻居约束 | P1 + L_neighbor(0.3) + L_overlap(0.1) |
| **P2-B** | Phase 2-B - 边界精度 | P2-A + DiffContour + DiffTopology |
| **P2-C** | Phase 2-C - 参数优化 | P2-A + lr=5e-5, epochs=80 |

---

## 推理方案命名

### 格式

```
{模型}_{检测方案}
```

### 示例

| 方案名 | 说明 |
|--------|------|
| **Baseline-GTB** | CellSAM 预训练 + GT 框 |
| **E29-GTB** | E29 微调模型 + GT 框 |
| **E29-DPB** | E29 微调模型 + DAPI 检测框 |
| **E30-GTB** | E30 Adapter 模型 + GT 框 |

---

## 通道命名

| 通道索引 | 名称 | 缩写 | 用途 |
|---------|------|------|------|
| Ch0 | Brightfield | BF | 细胞边界可视化 |
| Ch1 | Actn2/Z-Line | ACT/ZL | 肌节结构标记 |
| Ch4 | DAPI | DAPI | 细胞核染色 |
| Ch9 | GT Mask | GT | 标注掩码 |

---

## 数据集命名

| 名称 | 说明 |
|------|------|
| **Allen-CM-405** | Allen Institute 心肌细胞数据集 (405 样本) |
| **Train-334** | 训练集 (334 样本, 82.5%) |
| **Val-71** | 验证集 (71 样本, 17.5%) |

---

## 文件命名约定

### Checkpoint

```
{实验ID}_{模型类型}_best.pt
{实验ID}_{模型类型}_epoch{N}.pt
```

例如: `E29_bf_instance_best.pt`

### 配置文件

```
{模型类型}_{阶段}_{日期}.yaml
```

例如: `bf_instance_p2_20260205.yaml`

---

## 快速参考

```
┌─────────────────────────────────────────────────────────────┐
│ SAM (Meta) → CellSAM (生物微调) → CM-CellSAM (心肌细胞微调) │
└─────────────────────────────────────────────────────────────┘

检测方案:
├── GTB (GT-Box): Ground Truth 框 - 评测标准
├── DPB (DAPI-Box): DAPI 核检测 - 初始方案
├── ZLB (ZLine-Box): Z 线检测 - 替代方案
└── CFB (CellFinder-Box): 自动检测 - 端到端

实验阶段:
├── P1: 基础损失 (Dice + BCE + Boundary + AJI)
└── P2: 高级损失 (P1 + Topology + Size + Contour)
```
