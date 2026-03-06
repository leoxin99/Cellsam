# CellSAM Methods (Paper-Ready 1-Page Table)

> Author: A1 (Codex)  
> Source scope: Nature Methods paper (`docs/Cellsam-nature.pdf`) + public repo loss evidence for CellFinder (`cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py`).  
> Note: Stage-2 exact loss formula/weights are **not explicitly published** as a reproducible script in the public CellSAM repo.

| 阶段 | 训练目标 | 训练模块 | 冻结模块 | Loss（可证据化口径） | 关键超参 | 主要指标 | 证据页码/代码 |
|---|---|---|---|---|---|---|---|
| Stage 1 (Detection) | 学到细胞检测能力（由 GT mask 转 GT box） | `CellFinder` + `SAM image encoder (ViT)` | `SAM mask decoder` | 检测损失：分类 + 框回归 + 几何（公开实现对应 Focal CE + L1 + GIoU） | AdamW；CellFinder lr=`1e-4`；SAM-ViT backbone lr=`1e-5`；wd=`1e-4`；clip norm=`0.1`；step scheduler（1960 epoch 后降 10x）；2800 epochs；batch=4；8x H100 | COCO `mAP`、`AP50`（IoU 0.5:0.95, step 0.05；max detections=10,000） | Paper p3（两阶段与冻结关系），p10（训练超参），p11（COCO 指标）；`anchor_detr.py` 中 `loss_ce/loss_bbox/loss_giou` |
| Stage 2 (Segmentation alignment) | 在 GT boxes + segmentation labels 监督下，重新对齐分割分支 | `model neck`（仅 neck 微调） | `SAM-ViT` + `mask decoder` | 论文写法为“segmentation supervision fine-tuning neck”；**未给可逐行复现的公开 Stage-2 loss 公式与权重** | AdamW；lr=`1e-4`；wd=`1e-4`；不做 gradient clipping；50 epochs + cosine lr schedule | 分割主比较口径是 `F1 error (1-F1)`；并给 Recall/Precision/F1 | Paper p3（neck-only），p10（lr/wd/no-clip），p11（50 epochs + cosine；R/P/F1定义） |
| Benchmark reporting (paper-level) | 统一比较 CellSAM 与 baselines | 检测与分割分开报告 | - | 检测侧看 COCO；分割侧主文强调 `1-F1` | - | `1-F1`（越低越好），以及 Recall/Precision/F1 | Paper p3-4（1-F1 主文对比），p11（R/P/F1 公式与 COCO 说明） |

## Evidence Boundary (for writing)

1. 可以确定写入论文的内容：两阶段训练结构、Stage1/Stage2超参、检测指标与分割指标口径。  
2. 不能写死的内容：Stage2 “Dice+BCE”或具体权重组合（论文与公开仓库都未给出可逐行复现脚本）。  
3. 若论文需要“损失公式”小节，建议表述为：  
   - Stage1: DETR-style detection objective (classification + box regression + IoU geometry terms).  
   - Stage2: neck fine-tuning under segmentation supervision with GT boxes and masks, exact internal weighting not publicly specified.
