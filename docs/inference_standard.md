# CellSAM 推理标准文档

> **状态**: 🟢 Active — 推理与评估的唯一口径文档
> **最后更新**: 2026-02-13
> **事实来源**: `src/inference/core.py` (InferenceConfig + segment_with_boxes)
> **规则**: 所有推理/评估脚本必须调用 `core.py` 的函数，不允许硬编码参数

---

## 一、统一推理配置 (InferenceConfig)

所有推理参数由 `InferenceConfig.default()` 定义，**不允许脚本内硬编码**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_threshold` | 0.5 | sigmoid → 二值化阈值 |
| `use_sam_iou_filter` | False | 是否过滤低 IoU 预测 |
| `sam_iou_threshold` | 0.5 | SAM IoU 过滤阈值 |
| `apply_box_clipping` | True | mask 裁剪到 box 区域 |
| `box_expand` | 0.1 | box 扩展比例 |
| `conflict_policy` | `"confidence"` | 重叠像素归属策略 |
| `min_cell_area` | 13884 | 细胞最小面积 (1024px) |
| `max_cell_area` | 174735 | 细胞最大面积 (1024px) |

```python
from src.inference.core import InferenceConfig, segment_with_boxes

# 正确用法:
config = InferenceConfig.default()
result = segment_with_boxes(model, image, boxes, config)

# ❌ 错误: 不要自己写阈值
# mask = (pred > 0.5).float()  # hardcoded!
```

---

## 二、推理核心函数

### `segment_with_boxes()` — 统一入口

```
输入: model, image [C,H,W], boxes [N,4], config
  ↓
逐 box 切片 → SAM predictor → sigmoid
  ↓
resolve_conflicts (confidence/area policy)
  ↓
postprocess (面积过滤)
  ↓
输出: InferenceResult (instance_mask, confidence_map, n_instances, stats)
```

### `resolve_conflicts()` — 重叠裁决

| 策略 | 说明 |
|------|------|
| `confidence` | **默认** — 重叠像素归属置信度最高的实例 |
| `area` | 重叠像素归属面积最小的实例 |

### `postprocess_instance_mask()` — 后处理

1. 移除面积 < `min_cell_area` 的碎片
2. 移除面积 > `max_cell_area` 的异常区域

---

## 三、评估指标标准

### Instance Dice (Best-Match 方法) ⭐

**项目官方 Dice 计算方法**: 每个 GT 细胞找最佳匹配的预测细胞。

```
Training Dice: Direct-Match (box → cell_id 一对一)
Evaluation Dice: Best-Match (GT 找最佳预测匹配)
```

### PQ@0.5 (Panoptic Quality)

```
PQ = SQ × RQ

SQ (Segmentation Quality) = mean(IoU of matched pairs)
RQ (Recognition Quality) = TP / (TP + 0.5*FP + 0.5*FN)

匹配条件: IoU ≥ 0.5
```

---

## 四、评估工具分工

| 工具 | 用途 | Box 来源 | 说明 |
|------|------|----------|------|
| `tools/comprehensive_eval.py` | **Oracle 评估** | GT boxes | 纯分割能力 |
| `tools/evaluate_e2e.py` | **E2E 评估** | DAPI 检测 | 含检测误差 |
| `tools/test_unified_regression.py` | **回归测试** | GT boxes | 防止退化 |
| `tools/smoke_test_e2e.py` | **冒烟测试** | GT boxes | 快速验证 (1 样本) |

### 使用方式

```bash
# Oracle 评估 (标准测试)
python tools/comprehensive_eval.py \
  --checkpoint checkpoints/E_phase1_rebalance_l4/best_model.pt \
  --split val --samples 71

# E2E 评估
python tools/evaluate_e2e.py \
  --checkpoint checkpoints/E_phase1_rebalance_l4/best_model.pt

# 回归测试 (训练前必跑)
python tools/test_unified_regression.py

# 训练前完整验证
python tools/verify_training_config.py --config src/config/CONFIG.yaml
```

---

## 五、Checkpoint 加载标准

使用 `load_cellsam_checkpoint()` 统一加载，支持 adapter：

```python
from src.inference.core import load_cellsam_checkpoint

model, adapter, info = load_cellsam_checkpoint(
    checkpoint_path="checkpoints/xxx/best_model.pt",
    device="cuda"
)
# info 包含: epoch, best_dice, best_pq, config
```

---

## 六、历史结果

| 日期 | 模型 | Oracle Dice | Oracle PQ | 样本数 |
|------|------|-------------|-----------|--------|
| 2026-02-10 | Baseline (预训练) | 0.589 ± 0.237 | 0.337 ± 0.140 | 71 (val) |
| 2026-02-10 | Phase 1 (L4 best) | 0.695 | 0.464 | 71 (val) |
| 2026-02-13 | Phase 2-A | 训练中... | — | — |

---

## 更新日志

- 2026-02-13: 重写文档，以 `core.py` 为 SSOT，移除旧 API 引用
- 2026-02-07: 创建文档，确立 Best-Match 为标准方法
