# T33: CellFinder Allen-Specific Adaptation (Head-Only)

## 1. Background

CellSAM Stage 1 jointly trains ViT backbone + CellFinder on object detection.
We want to evaluate whether fine-tuning CellFinder's detection head on our Allen cardiomyocyte data
improves detection. This is a **resource-constrained Allen adaptation inspired by Stage 1**,
not a faithful reproduction of the paper's full Stage 1 (which jointly trains backbone + CellFinder
on ~1.2M cells across multiple datasets).

### CellSAM Paper Stage 1 Training

| 参数 | CellSAM Paper | 备注 |
|------|:------------:|------|
| 训练目标 | ViT backbone + CellFinder | 联合 |
| Loss | Focal CE + L1 + GIoU (SetCriterion) | AnchorDETR 标准 |
| Epochs | 2800 | ~1.2M 数据 |
| lr (backbone) | 1e-5 | ViT backbone |
| lr (rest) | 1e-4 | CellFinder decoder head |
| lr schedule | StepLR, decay 10x @ epoch 1960 | |
| Matcher | Hungarian matching | |
| Num queries | 3500 | 最大检测框数 |
| 数据 | ~1.2M cells, 多数据集 | |

### 代码证据

- `SetCriterion`: `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:140-365`
  - `loss_labels`: Focal CE (alpha=0.25)
  - `loss_boxes`: L1 + GIoU
- 数据格式: targets = `[{"labels": tensor, "boxes": tensor(cx,cy,w,h)}]`

## 2. 我们的 Fine-Tuning 方案

### 2.1 训练目标

- 冻结 ViT backbone（`cellfinder.decode_head.backbone`）
- 只训练 CellFinder decoder head（`cellfinder.decode_head` 中非 backbone 部分）
- **理由**: 我们数据量只有 310 张 / ~5000 cells，不足以训练 ViT（过拟合风险）

### 2.2 数据准备

需要将 GT instance masks 转为 bounding boxes:

```python
# 从 GT mask 提取 boxes
def masks_to_boxes(instance_mask):
    """Convert instance mask to [cx, cy, w, h] normalized boxes."""
    boxes = []
    for cell_id in np.unique(instance_mask):
        if cell_id == 0:  # background
            continue
        ys, xs = np.where(instance_mask == cell_id)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2 / W
        cy = (y_min + y_max) / 2 / H
        w = (x_max - x_min) / W
        h = (y_max - y_min) / H
        boxes.append([cx, cy, w, h])
    return boxes
```

### 2.3 输入通道

CellFinder 输入通过 `sam_bbox_preprocessing` 预处理（`cellSAM_source/cellSAM/sam_inference.py:168`）:
- 使用 CellSAM 官方通道编码: `[blank, DAPI, BF]`
- 归一化到 [0,1] 后 resize to target_size

### 2.4 训练超参

| 参数 | 我们的方案 | 理由 |
|------|:--------:|------|
| Epochs | 200 | 数据量小, 不需要 2800 |
| lr (decoder head) | 1e-4 | 与论文一致 |
| lr (backbone) | 0 (冻结) | 数据量不足 |
| batch_size | 4 | GPU 内存限制 |
| early stop | val F1@0.5 patience=20 | 实际实现为 F1 监控 (COCO mAP 未实现) |
| num_queries | 50 | 心肌细胞数据 ~10-30 cells/image, 50 给 2x 余量 (注: 原 CellFinder=3500, 方案文档写的 300 已改为 50) |
| scheduler | CosineAnnealingWarmRestarts | |
| Loss | Focal CE + L1 + GIoU | 与论文一致 |

### 2.5 评估指标 (实际实现)

- F1 @ IoU=0.5 (主指标, 用于早停)
- Precision @ IoU=0.5
- Recall @ IoU=0.5

> [!NOTE]
> 方案文档原设计的 COCO mAP / AP50 / AP75 未实现。当前代码使用 F1 监控。

### 2.6 需要的代码修改

1. **新建训练脚本**: `tools/train_cellfinder.py`
   - 使用 `SetCriterion` + Hungarian matcher
   - 数据加载: 从 processed data 读取, mask→box 转换
   - 冻结 backbone, 训练 decoder head
   - 验证集评估: F1/Precision/Recall @ IoU=0.5

2. **数据格式适配**:
   - AnchorDETR 期望: `targets = [{"labels": int_tensor, "boxes": float_tensor(cx,cy,w,h)}]`
   - 所有 box 坐标归一化到 [0,1]
   - label = 0 (单类: cell)

### 2.7 预期风险

1. 数据量太小（310张 vs 论文的 100K+），即使只训练 head 也可能过拟合
2. 心肌细胞形态与 CellSAM 训练集差异大（大细胞 vs 小圆细胞）
3. num_queries=50 与官方 3500 不一致 — 工程降规假设 (心肌数据 ~10-30 cells/image, 50 给 2x 余量)
4. **与论文 Stage 1 的关键差异**: 论文联训 backbone+CellFinder, 我们冻结 backbone 只训 head。论文 specialist 是按数据子集重训所有模块得到的, 不是简单调阈值。本方案更接近 head-only continuation 而非完整 Stage 1 复现。

## 3. Results (ALICE, 2026-03-07)

| Seed | Best F1 | Early Stop Epoch | Total Epochs |
|:----:|:-------:|:----------------:|:------------:|
| 42 | **0.5550** | Epoch 39 | 39/200 |
| 123 | **0.5573** | Epoch 39 | 39/200 |
| **Mean** | **0.5562** | | |

- **Checkpoint**: `checkpoints/T33_CellFinder_HeadOnly_seed{42,123}_20260307_213750/`
- **监控指标**: F1 @ IoU=0.5 (非 COCO mAP)
- **早停**: patience=20, 两 seed 均在 epoch 39 触发
- **环境**: ALICE L4 GPU, `num_queries=50`, `batch_size=4`

### 3.1 与其他方法对比

| 方法 | F1 | 说明 |
|------|:---:|------|
| T27a (GT boxes) | 0.944 | CellSAM decoder, Oracle detection |
| **T33 CellFinder** | **0.556** | AnchorDETR head-only, 自动检测 |
| DAPI Z 线 + T27a | 0.507 | 传统检测 + CellSAM 分割 |
| DAPI 核检测 + T27a | 0.434 | 传统检测 + CellSAM 分割 |

> T33 F1 是检测 F1 (box matching)，CellFinder > DAPI 传统方法 (+5-12%)。

### 3.2 与 CellSAM 论文 CellFinder 训练方案的差异

| 方面 | CellSAM 论文 | 我们 (T33) | 影响评估 |
|------|:----------:|:---------:|:-------:|
| 训练范围 | ViT backbone + CellFinder 联训 | 冻结 backbone, 仅训 head | ⚠️ 高 |
| num_queries | 3500 | 50 | ⚠️ 中 (数据密度不同) |
| 数据量 | ~1.2M cells, 多数据集 | ~310 images, 单数据集 | ⚠️ 高 |
| Epochs | 2800 | 200 (early stop at 39) | ✅ 合理 |
| LR schedule | StepLR, decay@1960 | CosineAnnealingWarmRestarts | ⚠️ 低 |
| Loss | Focal CE + L1 + GIoU | 同 (SetCriterion) | ✅ 一致 |
| Matcher | Hungarian | 同 | ✅ 一致 |
| 监控指标 | COCO mAP + AP50 | F1 @ IoU=0.5 | ⚠️ 中 |
| backbone lr | 1e-5 | 0 (冻结) | ⚠️ 高 |

> [!WARNING]
> T33 与论文的 3 个最大差异: (1) backbone 冻结, (2) 数据量差 4 个数量级, (3) 监控指标用 F1 而非 COCO mAP。建议未来补充 COCO mAP evaluator 以论文一致口径汇报。
