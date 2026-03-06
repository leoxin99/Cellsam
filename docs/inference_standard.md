# CellSAM 推理标准文档

> **状态**: 🟢 Active — 推理与评估的唯一口径文档
> **最后更新**: 2026-03-06
> **事实来源**: `src/inference/core.py` (InferenceConfig + segment_with_boxes)
> **规则**: 默认推理/评估脚本必须调用 `core.py` 的函数，不允许硬编码参数；若做官方路径对照实验，必须显式标注为非 SSOT 审计路径

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
| `conflict_policy` | `"argmax_prob"` | 重叠像素归属策略 (argmax_prob/first_write/last_write) |
| `apply_postprocess` | True | 是否启用面积过滤后处理 |
| `validate_size` | False | 是否验证细胞面积范围 |
| `min_cell_area` | 13884 | 细胞最小面积 (GT P1, 1024px) |
| `max_cell_area` | 174735 | 细胞最大面积 (GT P99, 1024px) |

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
resolve_conflicts (argmax_prob policy)
  ↓
postprocess (仅当 apply_postprocess=True 时启用)
  ↓
输出: InferenceResult (instance_mask, confidence_map, n_instances, stats)
```

### `resolve_conflicts()` — 重叠裁决

| 策略 | 说明 |
|------|------|
| `argmax_prob` | **默认** — 重叠像素归属 sigmoid 输出最高的实例 |
| `first_write` | 先处理的 box 优先，后续不覆盖 |
| `last_write` | 后处理的 box 覆盖之前的 |

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
| `tools/eval_ablation.py` | **Oracle 最终评估** | GT boxes | 自动从实验目录读取 `best_model.pt`，默认评估 test(73) |
| `tools/evaluate_e2e.py` | **E2E 评估** | DAPI 检测 | 含检测误差，默认走 `locked_eval` 检测参数 |
| `tools/smoke_test_e2e.py` | **Oracle 开发冒烟** | GT boxes | 默认 30 样本，用于快速比较 |
| `tools/test_unified_regression.py` | **回归测试** | GT boxes | 防止推理/指标退化 |
| `tools/eval_t34_official_path.py` | **官方路径对照审计** | GT boxes | 仅用于 T34 类“官方路径 vs 统一核心”对照，不是 SSOT 主入口 |

归档:
- `tools/archive/comprehensive_eval.py`

### 4.1 检测参数 Profile 机制 (T4, 2026-02-16)

为降低“误用默认检测参数”风险，检测评估脚本统一从 `src/detection/profiles.py` 读取检测参数。当前只有一个活跃 profile：

| Profile | 用途 | 参数来源 |
|---------|------|----------|
| `locked_eval` | 统一评估/封板 | E34/E34b (DAPI) + T3b (Adaptive) 锁定参数 |

实现位置:
- `src/detection/profiles.py`

执行规则:
1. 最终汇报、阶段结论、test 封板必须用 `locked_eval`。
2. 若要测试 `dapi.py` 的运行时默认值，必须作为单独实验在脚本内显式硬编码并注明“非 SSOT / runtime default audit”。
3. 关键脚本启动时会打印参数快照 (`profile + dapi/adaptive params`)。

已接入脚本:
- `tools/evaluate_e2e.py` (`--detection-profile`, 默认 `locked_eval`)
- `tools/ablation_detection_lock.py` (`--profile`, 默认 `locked_eval`)
- `tools/ablation_detection_e34b.py` (`--profile`, 默认 `locked_eval`)
- `tools/ablation_adaptive_val.py` (`--profile`, 默认 `locked_eval`)

### 使用方式

```bash
# Oracle 评估 (推荐)
python tools/eval_ablation.py --exp-dir checkpoints/EXP_DIR

# E2E 评估
python tools/evaluate_e2e.py \
  --checkpoint checkpoints/E_phase1_rebalance_l4/best_model.pt

# E2E 评估 (显式指定 profile，推荐)
python tools/evaluate_e2e.py \
  --checkpoint checkpoints/E_phase1_rebalance_l4/best_model.pt \
  --detection-profile locked_eval

# 回归测试 (训练前必跑)
python tools/test_unified_regression.py

# 训练前完整验证
python tools/verify_training_config.py --config src/config/CONFIG.yaml

# 官方路径对照审计 (仅 T34 / 审计类实验使用)
python tools/eval_t34_official_path.py --checkpoint checkpoints/EXP_DIR/best_model.pt
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

> 说明：前两行是历史 `val` 结果；`Phase 1 Test Locked` 为当前对外汇报与阶段结论口径。

| 日期 | 模型 | Oracle Dice | Oracle PQ | 样本数 |
|------|------|-------------|-----------|--------|
| 2026-02-10 | Baseline (预训练) | 0.589 ± 0.237 | 0.337 ± 0.140 | 71 (val) |
| 2026-02-10 | Phase 1 (L4 best) | 0.695 | 0.464 | 71 (val) |
| 2026-02-11 | Phase 1 Test Locked | 0.6954 | 0.4641 | 73 (test) |

---

## 更新日志

- 2026-03-06: 修正 `apply_postprocess=True` 默认值；更新评估工具分工与用法示例
- 2026-02-13: 重写文档，以 `core.py` 为 SSOT，移除旧 API 引用
- 2026-02-07: 创建文档，确立 Best-Match 为标准方法
