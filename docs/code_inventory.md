# CellSAM 代码清单与版本记录

> **目的**: 记录所有代码文件的用途、版本、所属实验
> **创建**: 2026-01-23
> **更新规则**: 新增/修改代码后必须更新此文档

---

## 1. 核心代码 (src/)

| 文件 | 功能 | 版本 | 备注 |
|------|------|------|------|
| `detection/dapi.py` | DAPI核检测+框生成 | v3 | 含 detect_cardiomyocytes, detect_with_adaptive_box |
| `losses/combined.py` | 损失函数 | v2 | DiceLoss + BoundaryLoss + AJILoss |
| `train.py` | 训练脚本 | v1 | 标准训练流程 |
| `inference/postprocess.py` | 后处理 | v1 | 6步边界平滑 |

---

## 2. 实验代码 (tools/)

| 文件 | 所属实验 | 功能 | 创建日期 |
|------|---------|------|---------|
| `evaluate_box_generation.py` | E18扩展 | 对比 DAPI/Adaptive 框与 GT | 2026-01-23 |
| `test_sarcgraph_detection.py` | E18 | SarcGraph Z-线检测对比 | 2026-01-22 |
| `visualize_detection_comparison.py` | E18 | Napari 可视化检测对比 | 2026-01-23 |

---

## 3. 对比实验代码 (anti_test/)

| 文件 | 所属实验 | 功能 |
|------|---------|------|
| `test_dapi_detection.py` | E03 | DAPI 核检测验证 (基准) |
| `test_traditional_detection.py` | E02 | 传统检测方法对比 |
| `visualize_test_results.py` | 通用 | 结果可视化 |
| `eval_metrics.py` | 通用 | 评估指标计算 |

---

## 4. SarcGraph 对比代码 (src/comparison/)

| 文件 | 功能 | 来源 |
|------|------|------|
| `sarcgraph_pipeline/prompt_generator.py` | Z-线检测+框生成 | Claude 方案 |
| `sarcgraph_pipeline/preprocessing.py` | 语义通道映射 | Claude 方案 |

---

## 5. 版本更新日志

| 日期 | 文件 | 更新内容 | 版本 |
|------|------|---------|------|
| 2026-01-23 | `detection/dapi.py` | 添加 detect_with_adaptive_box, filter_by_actn2 | v3 |
| 2026-01-23 | `evaluate_box_generation.py` | 使用 detect_cardiomyocytes, 新增边缘过滤 | v2 |
| 2026-01-22 | `detection/dapi.py` | 添加 detect_cardiomyocytes (Actn2过滤) | v2 |

---

## 6. 文件头部模板

新代码文件必须包含以下头部注释:

```python
"""
[文件名]

功能: [简要描述]
所属实验: [E编号 或 "核心代码"]
创建日期: [YYYY-MM-DD]
最后修改: [YYYY-MM-DD]
版本: [vN]

依赖函数:
- [列出核心依赖]

更新日志:
- [日期]: [更新内容]
"""
```
