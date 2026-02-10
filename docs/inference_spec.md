# 统一推理规范 (Oracle + E2E)

> **创建日期**: 2026-02-09
> **状态**: 生效中

---

## 一、双任务定义

| 任务 | Box 来源 | 用途 | 脚本 |
|------|---------|------|------|
| **Oracle** | GT boxes | Assess segmentation quality ceiling | `tools/standardized_inference.py` |
| **Oracle (multi-model)** | GT boxes | Compare BF vs Adapter | `tools/comprehensive_eval.py` |
| **E2E** | DAPI boxes | Assess real deployment quality | `tools/evaluate_e2e.py` |

> **Note**: `tools/run_inference.py` is **DEPRECATED** (uses legacy pipeline with `first_write` conflict).

---

## 二、统一推理口径

所有推理路径必须使用相同的核心参数：

```yaml
# 标准推理配置
inference:
  mask_threshold: 0.5           # 二值化阈值
  use_sam_iou_filter: false     # 是否启用 SAM iou 过滤
  sam_iou_threshold: 0.5        # SAM iou 阈值 (若启用)
  apply_box_clipping: true      # 是否裁剪到 box 区域
  box_expand: 0.1               # box 扩展比例
  conflict_policy: "argmax_prob" # 冲突像素策略
  apply_postprocess: false      # 后处理开关
```

---

## 三、冲突像素裁决策略

### 可选策略

| 策略 | 说明 | 优缺点 |
|------|------|--------|
| `argmax_prob` | 取置信度最高的实例 | ✅ 推荐，顺序无关 |
| `first_write` | 先处理的 box 占据 | ⚠️ 顺序依赖 |
| `last_write` | 后处理的 box 覆盖 | ⚠️ 顺序依赖 |

### 默认: `argmax_prob`

```python
# 实现逻辑
for each pixel:
    assigned_id = argmax(confidence_map, axis=instance)
```

---

## 四、统一 API 接口

### 核心函数: `segment_with_boxes`

```python
from inference.core import segment_with_boxes, InferenceConfig, load_cellsam_checkpoint

def segment_with_boxes(
    model,
    image: torch.Tensor,       # [C, H, W]
    boxes: torch.Tensor,       # [N, 4]
    config: InferenceConfig,   # InferenceConfig.default()
    device: str = 'cuda'
) -> InferenceResult:
    """
    Returns:
        InferenceResult with:
            instance_mask: [H, W] int32
            confidence_map: [N, H, W] float
            n_instances: int
            conflict_pixels: int
    """
```

---

## 五、指标统一

### 必须报告的指标

| 指标 | 计算方法 | 汇总方式 |
|------|---------|---------|
| BM-1to1 Dice | Hungarian optimal 1:1 matching | per-image mean | **Primary** |
| BM-Coverage Dice | Each GT takes best Pred | per-image mean | Diagnostic |
| Gap Dice | Coverage - 1to1 | per-image mean | Adhesion diagnostic |
| PQ@0.5 | SQ x RQ | per-image mean | **Primary** |
| AJI | Aggregated Jaccard Index | per-image mean | Auxiliary |
| Semantic Dice | Binary foreground Dice | per-image mean | Auxiliary |

### Diagnostic metrics

| Metric | Description |
|--------|-------------|
| conflict_pixels | Pixels claimed by >1 instance (resolved by conflict_policy) |

---

## 六、报告格式

每次评估必须输出:

```
=== Evaluation Report ===
Task: Oracle/E2E
Checkpoint: xxx.pt
Config: xxx.yaml
Inference Settings:
  - mask_threshold: 0.5
  - conflict_policy: argmax_prob
  - apply_box_clipping: true

Results:
  Oracle (GT boxes):    Dice=0.xx  PQ=0.xx
  E2E (DAPI boxes):     Dice=0.xx  PQ=0.xx
  Gap (Oracle-E2E):     Dice=+0.xx PQ=+0.xx
```
