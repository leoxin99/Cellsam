# 论文引用指南 (Reference Guide)

> **用途**: 列出硕士论文中需要引用的所有论文, 说明引用理由和位置
> **最后更新**: 2026-02-27
> **BibTeX 文件**: `references.bib`

---

## ✅ 必须引用 (核心方法依赖)

### 1. SAM — Segment Anything Model
- **引用**: Kirillov et al., ICCV 2023
- **BibKey**: `kirillov2023sam`
- **简介**: Meta 提出的视觉基础模型, 用 SA-1B (11M 图) 预训练。ViT-B encoder + prompt encoder + mask decoder 架构。在自然图像上实现零样本分割。
- **引用理由**: **你的整个项目基于 SAM 架构**。CellSAM 是 SAM 的生物领域微调版本。
- **出现位置**: Ch.1 Introduction, Ch.2 §2.1-2.2, Ch.3 §3.1

### 2. CellSAM — A Foundation Model for Cell Segmentation
- **引用**: Israel et al., bioRxiv 2024
- **BibKey**: `israel2024cellsam`
- **简介**: 在 SAM 基础上用大规模细胞数据集 (Stage 1 + Stage 2) 微调, 专门用于细胞分割。我们使用的是其 ViT-B 版本的公开权重。
- **引用理由**: **你的基座模型**, 所有实验从 CellSAM 权重出发微调。
- **出现位置**: Ch.1, Ch.2 §2.2, Ch.3 §3.1, Ch.4 §4.1, Ch.5 §5.1

### 3. MedSAM — Segment Anything in Medical Images
- **引用**: Ma et al., Nature Communications 2024
- **BibKey**: `ma2024medsam`
- **简介**: 用 1.5M 医学图像 (11 种模态) **全参数微调** SAM ViT-B。checkpoint 名 `medsam_vit_b.pth`, 下载自 Zenodo。
- **引用理由**: **你的 baseline 上界**, PQ=0.576。体现了大规模医学预训练的优势, 也是你做 LoRA encoder 微调的动机 (缩小与 MedSAM 的差距)。
- **出现位置**: Ch.2 §2.1, Ch.4 §4.3, Ch.5 §5.1 Tab.1, Ch.6 Discussion

### 4. LoRA — Low-Rank Adaptation
- **引用**: Hu et al., ICLR 2022
- **BibKey**: `hu2022lora`
- **简介**: 提出低秩分解旁路用于大模型微调。核心思想: 冻结原始权重, 加可训练的 A(d×r) × B(r×d) 旁路。实验发现 Q+V 投影是最优目标。
- **引用理由**: **LoRA 原理的原始论文**, T11 实验的理论基础。
- **出现位置**: Ch.2 §2.3, Ch.3 §3.6

### 5. SAMed — Customized SAM for Medical Segmentation
- **引用**: Zhang & Liu, arXiv:2304.13785, 2023
- **BibKey**: `zhang2023samed`
- **简介**: 首次将 LoRA 应用到 SAM image encoder 的 Q/V 投影, rank=4。在 Synapse 多器官 CT 数据集上验证。用 AdamW + warmup, 2×RTX 3090, 224→512 分辨率。
- **引用理由**: **T11 LoRA 策略的直接来源** — 你的 Q/V rank=4/8 方案参考 SAMed。
- **出现位置**: Ch.2 §2.3, Ch.3 §3.6, Ch.5 §5.7
- **⚠️ 注意**: 不是 ICLR 2024 (那是另一篇 *Convolution Meets LoRA*), 正确为 arXiv 2023

### 6. Cellpose
- **引用**: Stringer et al., Nature Methods 2021
- **BibKey**: `stringer2021cellpose`
- **简介**: 基于梯度场预测的通用细胞分割算法。无需边界框, 直接从图像预测实例掩码。
- **引用理由**: **你的 baseline 之一** (E2E 评估), PQ=0.391。
- **出现位置**: Ch.2 §2.1, Ch.4 §4.3, Ch.5 §5.1 Tab.1

### 7. StarDist
- **引用**: Schmidt et al., MICCAI 2018
- **BibKey**: `schmidt2018stardist`
- **简介**: 用星凸多边形预测细胞形状, 擅长圆形/椭圆形细胞。
- **引用理由**: **你的 baseline 之一**, PQ=0.307。在心肌细胞 (长条形) 上表现差, 说明通用方法不适用。
- **出现位置**: Ch.2 §2.1, Ch.4 §4.3, Ch.5 §5.1 Tab.1

### 8. Dice Loss (V-Net)
- **引用**: Milletari et al., 3DV 2016
- **BibKey**: `milletari2016vnet`
- **简介**: 提出在医学图像分割中直接优化 Dice 系数作为损失函数。
- **引用理由**: 你的 CombinedLoss 核心组件 — Dice loss。
- **出现位置**: Ch.3 §3.4

### 9. AJI (Aggregated Jaccard Index)
- **引用**: Kumar et al., IEEE TMI 2017
- **BibKey**: `kumar2017aji`
- **简介**: 提出 AJI 指标用于评估核分割质量, 考虑实例级匹配。
- **引用理由**: **你的评估指标之一 + Loss 组件** (L_aji)。
- **出现位置**: Ch.2 §2.5, Ch.3 §3.4, Ch.5 全部结果表

### 10. PQ (Panoptic Quality)
- **引用**: Kirillov et al., CVPR 2019
- **BibKey**: `kirillov2019pq`
- **简介**: 定义 PQ = SQ × RQ, 统一评估分割质量和检测精度。
- **引用理由**: **你的主评估指标和早停指标**。
- **出现位置**: Ch.2 §2.5, Ch.4 §4.2, Ch.5 全部

---

## 🟡 可选引用 (Related Work 完整性)

### 11. Med-SA — Medical SAM Adapter
- **引用**: Wu et al., arXiv:2304.12620, 2023
- **BibKey**: `wu2023medsa`
- **简介**: 在 SAM Transformer block 之间插入 learnable adapter 层 (非 LoRA)。后来代码加了 LoRA 选项。
- **引用理由**: Ch.2 Related Work 中提及 PEFT 方法的多样性, 一句话带过。
- **出现位置**: Ch.2 §2.3 (一句话)

### 12. SAM-Med2D
- **引用**: Cheng et al., arXiv:2308.16184, 2023
- **BibKey**: `cheng2023sammed2d`
- **简介**: 用 4.6M 医学图像 + adapter 微调 SAM, 256×256 输入。
- **引用理由**: Ch.2 Related Work 完整性, 与 MedSAM 的区别 (adapter vs 全参)。
- **出现位置**: Ch.2 §2.3 (一句话)

### 13. Gradient Checkpointing
- **引用**: Chen et al., arXiv:1604.06174, 2016
- **BibKey**: `chen2016gradientcheckpoint`
- **简介**: 用计算换内存 — 不存中间激活, backward 时重算。
- **引用理由**: T11 OOM 修复中使用了此技术。如果 §3.6 或 §5.7 提到 gradient checkpointing 实现细节, 需引用。
- **出现位置**: Ch.3 §3.6 (可选)

### 14. Allen Institute hiPSC Dataset
- **引用**: Allen Institute for Cell Science
- **BibKey**: `rafelski2024allen`
- **简介**: 你使用的数据集来源。
- **引用理由**: Ch.3 §3.2 数据集说明。
- **出现位置**: Ch.3 §3.2, Ch.4 §4.1

---

## 论文中引用分布总览

| 章节 | 需引用 |
|------|--------|
| Ch.1 Introduction | SAM, CellSAM, MedSAM |
| Ch.2 Background | SAM, CellSAM, MedSAM, LoRA, SAMed, Cellpose, StarDist, Med-SA*, SAM-Med2D*, PQ, AJI, Dice |
| Ch.3 Methodology | CellSAM, SAMed, LoRA, Dice, AJI, Allen Dataset, Gradient Ckpt* |
| Ch.4 Setup | CellSAM, MedSAM, Cellpose, StarDist, Allen Dataset |
| Ch.5 Results | (所有 baseline 论文在 Tab.1 已标注) |
| Ch.6 Discussion | MedSAM, SAMed, LoRA |

`*` = 可选
