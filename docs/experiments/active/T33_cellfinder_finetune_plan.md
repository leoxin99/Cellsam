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
| early stop | val mAP patience=20 | |
| num_queries | 300 | 工程假设: 每张最多 ~40 cells (注: 官方 CellFinder=3500, 此处缩减是非论文一致的工程决策) |
| scheduler | CosineAnnealingWarmRestarts | |
| Loss | Focal CE + L1 + GIoU | 与论文一致 |

### 2.5 评估指标

- COCO mAP (IoU 0.5:0.95)
- AP@0.5
- AP@0.75
- per-class precision/recall

### 2.6 需要的代码修改

1. **新建训练脚本**: `tools/train_cellfinder.py`
   - 使用 `SetCriterion` + Hungarian matcher
   - 数据加载: 从 processed data 读取, mask→box 转换
   - 冻结 backbone, 训练 decoder head
   - 验证集评估: COCO mAP

2. **数据格式适配**:
   - AnchorDETR 期望: `targets = [{"labels": int_tensor, "boxes": float_tensor(cx,cy,w,h)}]`
   - 所有 box 坐标归一化到 [0,1]
   - label = 0 (单类: cell)

### 2.7 预期风险

1. 数据量太小（310张 vs 论文的 100K+），即使只训练 head 也可能过拟合
2. 心肌细胞形态与 CellSAM 训练集差异大（大细胞 vs 小圆细胞）
3. num_queries=300 与官方 3500 不一致 — 此为工程降规假设, 需单独论证合理性
4. **与论文 Stage 1 的关键差异**: 论文联训 backbone+CellFinder, 我们冻结 backbone 只训 head。论文 specialist 是按数据子集重训所有模块得到的, 不是简单调阈值。本方案更接近 head-only continuation 而非完整 Stage 1 复现。
