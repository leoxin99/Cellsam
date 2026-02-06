# CellSAM 代码清单与版本记录

> **目的**: 记录所有代码文件的用途、版本、所属实验
> **创建**: 2026-01-23
> **最后更新**: 2026-02-05
> **更新规则**: 新增/修改代码后必须更新此文档

---

## 1. 核心代码 (src/)

| 文件 | 功能 | 版本 | 最后更新 | 备注 |
|------|------|------|----------|------|
| `detection/dapi.py` | DAPI核检测+框生成 | v4 | 2026-02-05 | 参数更新为 1024px; detect_nuclei, detect_with_adaptive_box, detect_cardiomyocytes |
| `losses/combined.py` | 损失函数 | v3 | 2026-02-05 | 更新 min_area/max_area 为 1024px 缩放值 (×0.340) |
| `train.py` | 训练脚本 | v2 | 2026-02-05 | Instance-level training with box clipping |
| `inference/postprocess.py` | 后处理 | v2 | 2026-02-05 | 更新 MIN/MAX_CELL_AREA 为 1024px 值 |
| `adapters/channel_adapter.py` | Semantic Channel Adapter | v1 | 2026-01-30 | 3通道→RGB 映射 |
| `augmented_dataset.py` | 数据加载 | v2 | 2026-02-05 | Instance-level target (cell_id) |

---

## 2. 配置文件 (src/config/)

| 文件 | 实验 | 功能 | 创建日期 |
|------|------|------|----------|
| `bf_instance_p1_20260205.yaml` | E29 | BF单通道 Phase 1 (Instance训练) | 2026-02-05 |
| `adapter_instance_p1_20260205.yaml` | E30 | Semantic Adapter Phase 1 | 2026-02-05 |
| `bf_instance_p2_20260205.yaml` | E31 | BF + 全部Loss Phase 2 | 2026-02-05 |
| `adapter_instance_p2_20260205.yaml` | E32 | Adapter + 全部Loss Phase 2 | 2026-02-05 |
| `semantic_adapter.yaml` | E21 | Semantic Adapter 标准配置 | 2026-01-30 |

---

## 3. 实验代码 (tools/)

| 文件 | 所属实验 | 功能 | 创建日期 |
|------|---------|------|----------|
| `evaluate_box_generation.py` | E18扩展 | 对比 DAPI/Adaptive 框与 GT | 2026-01-23 |
| `test_sarcgraph_detection.py` | E18 | SarcGraph Z-线检测对比 | 2026-01-22 |
| `visualize_detection_comparison.py` | E18 | Napari 可视化检测对比 | 2026-01-23 |
| `verify_training_config.py` | 通用 | 训练前配置验证 | 2026-02-02 |
| `baseline_gt_cellsam_20260206.py` | E29基线 | GT框+预训练CellSAM对比 | 2026-02-06 |
| `visualize_segmentation_20260206.py` | 通用 | 多通道分割结果可视化 | 2026-02-06 |

---

## 4. 对比实验代码 (anti_test/)

| 文件 | 所属实验 | 功能 |
|------|---------|------|
| `test_dapi_detection.py` | E03 | DAPI 核检测验证 (基准) |
| `test_traditional_detection.py` | E02 | 传统检测方法对比 |
| `visualize_test_results.py` | 通用 | 结果可视化 |
| `eval_metrics.py` | 通用 | 评估指标计算 |

---

## 5. SarcGraph 对比代码 (src/comparison/)

| 文件 | 功能 | 来源 |
|------|------|------|
| `sarcgraph_pipeline/prompt_generator.py` | Z-线检测+框生成 | Claude 方案 |
| `sarcgraph_pipeline/preprocessing.py` | 语义通道映射 | Claude 方案 |

---

## 6. ALICE 脚本 (scripts/)

| 文件 | 功能 | 创建日期 |
|------|------|----------|
| `train_instance_20260205.sh` | Instance训练 SLURM 脚本 (E29-E32) | 2026-02-05 |
| `train_instance_alice.sh` | 通用 ALICE 训练脚本 | 2026-02-03 |

---

## 7. 版本更新日志

| 日期 | 文件 | 更新内容 | 版本 |
|------|------|---------|------|
| **2026-02-05** | `detection/dapi.py` | ⭐ 全部函数参数更新为 1024px (min=200, max=10000, margin=32, search_radius=256) | v4 |
| **2026-02-05** | `losses/combined.py` | ⭐ TopologyLoss/SizeLoss 参数更新 (×0.340 缩放) | v3 |
| **2026-02-05** | `train.py` | Instance-level training with box clipping | v2 |
| **2026-02-05** | `inference/postprocess.py` | MIN/MAX_CELL_AREA 更新为 13884/174735 | v2 |
| 2026-01-23 | `detection/dapi.py` | 添加 detect_with_adaptive_box, filter_by_actn2 | v3 |
| 2026-01-23 | `evaluate_box_generation.py` | 使用 detect_cardiomyocytes, 新增边缘过滤 | v2 |
| 2026-01-22 | `detection/dapi.py` | 添加 detect_cardiomyocytes (Actn2过滤) | v2 |

---

## 8. 文件头部模板

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

---

## 9. DAPI 参数速查 (1024px 分辨率)

| 参数 | 值 | 函数 | 说明 |
|------|-----|------|------|
| min_area | 200 | detect_nuclei | 核最小面积 |
| max_area | 10000 | detect_nuclei | 核最大面积 (P99) |
| margin | 32 | create_bounding_boxes | 边缘裁切距离 |
| search_radius | 256 | detect_zlines_in_region | Z-线搜索半径 |
| expansion_long | 5.0 | create_bounding_boxes | 长轴扩展因子 |
| expansion_short | 3.0 | create_bounding_boxes | 短轴扩展因子 |
| expansion_isotropic | 4.0 | create_bounding_boxes | 圆形核扩展因子 |

