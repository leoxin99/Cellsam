# CellFinder 框检测评估方案

## 目标

在 Allen cardiomyocyte test(73) 上评估 CellFinder 的检测效果:
1. 验证预训练 CellFinder 在我们数据上的"零样本"检测能力
2. 输出 COCO 标准指标 + 可视化

## 推理流程

```python
import torch, sys, numpy as np
sys.path.insert(0, 'cellSAM_source')
from cellSAM.sam_inference import CellSAM

# 1. 加载模型
config = {}
model = CellSAM(config)
model.eval()

# 2. 加载图像 (3ch processed)
img_3ch = np.load('data/processed/images/{id}.npy')  # (3, H, W)

# 3. 预处理 (CellSAM prep_2 方式)
# prep_2 做 resize + normalize, 输出 cellfinder 输入格式
img_tensor = model.prep_2(img_3ch)  # 内部处理

# 4. 生成 bounding boxes
bboxes = model.generate_bounding_boxes(
    img_tensor,
    bbox_threshold=0.4,   # CellSAM 默认
    iou_threshold=0.4     # NMS threshold
)
# bboxes shape: (N, 4) in [x1, y1, x2, y2] format

# 5. 对比 GT boxes (从 mask 提取)
gt_mask = np.load('data/processed/masks/{id}.npy')
```

## 脚本设计

新建: `tools/eval_cellfinder_detection.py`

### 输入
- test(73) 图像 + GT masks
- GT masks 转 bounding boxes

### 输出
1. **检测指标**:
   - Precision / Recall / F1 @ IoU 0.5
   - mAP @ IoU 0.5:0.95 (COCO style)
   - AP@0.5, AP@0.75
   - FP/FN 统计

2. **逐样本结果**: `experiments/cellfinder_detection_test73/per_sample.json`

3. **可视化**: 前 5 张 test 图像的框叠加图 (pred=红, GT=绿)

### 实现要点

1. **GT box 提取**: 从 instance mask → bounding box (regionprops)
2. **IoU 计算**: box-level IoU (不是 mask-level)
3. **匹配**: Hungarian 匹配 (IoU threshold = 0.5)
4. **通道**: 直接用 processed 3ch 数据（CellFinder 自带预处理）

### CLI

```bash
# 运行评估
python tools/eval_cellfinder_detection.py --split test

# 可视化模式（保存前5张叠加图）
python tools/eval_cellfinder_detection.py --split test --visualize
```

## 预计运行时间

- CellFinder 推理: ~2s/image (GPU) → test(73) 约 3 分钟
- 本地 RTX 4090 可直接运行
- 不需要 ALICE

## 预期风险

1. CellFinder 训练数据以小圆细胞为主, 心肌细胞 (~200px) 可能超出检测范围
2. `bbox_threshold=0.4` 可能需要调整 (可先在 val 做敏感性分析)
3. 预处理通道映射需确认: CellFinder 的 `sam_bbox_preprocessing` 是否正确处理 BF 通道
