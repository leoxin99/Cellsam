---
description: 运行模型评估并记录结果
---

# 运行模型评估 (Run Model Evaluation)

当需要评估模型性能时，按以下步骤执行：

## 1. 准备评估

确认要评估的模型路径，例如：
- 旧模型: `checkpoints/expanded_20260108_034352/best_model.pt`
- 新模型: `checkpoints/boundary_YYYYMMDD_HHMMSS/best_model.pt`

## 2. 运行评估指标

// turbo
```bash
conda activate cellsam
python anti_test/eval_metrics.py
```

或使用对比脚本：

// turbo
```bash
conda activate cellsam
python compare_models.py
```

## 3. 记录评估结果

评估完成后，按照 `/log-experiment` 工作流记录结果。

关键指标包括：

| 指标 | 说明 | 目标 |
|------|------|------|
| PQ@0.5 | 实例级质量 | > 0.5 |
| AJI | 聚合 Jaccard | > 0.5 |
| Dice | 像素级准确率 | > 0.8 |
| RI | Rand Index | > 0.9 (Allen 标准) |
| Max_IoU | 最佳实例匹配 | > 0.5 |

## 4. 解读结果

根据 PQ 分解诊断问题：

```
PQ = SQ × RQ

SQ 高 + RQ 低 → 检测问题（检测不出/假阳）
SQ 低 + RQ 高 → 分割问题（边界不准）
```

## 5. 决定下一步

| 结果 | 行动 |
|------|------|
| PQ 提升 | 记录成功，保存模型 |
| PQ 无变化 | 检查参数，调整策略 |
| PQ 下降 | 回滚，分析原因 |
