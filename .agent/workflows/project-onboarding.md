---
description: AI 助手加入项目时的入门指南
---

# AI 助手入门指南 (AI Assistant Onboarding)

新 AI 助手加入本项目时，请按以下顺序阅读文档：

## 1. 阅读项目蓝图

**文件**: `CLAUDE.md`

重点关注：
- Project Overview (项目目标)
- Project Status Dashboard (当前状态)
- Documentation Management Scheme (文档结构)
- Key Decision Log (重要决策)

## 2. 了解实验历史

**文件**: `anti_test/experiments_log.md`

阅读顺序：
1. 实验索引 - 快速了解所有实验
2. 最近 3 个实验详情 - 了解当前进展
3. 关键决策记录 - 了解为什么做出某些选择

## 3. 查看当前结果

**文件**: `anti_test/results_summary.md`

获取：
- 核心指标数值
- 最新模型性能
- 待完成任务

## 4. 了解方法论

**文件**: `anti_test/methods_draft.md`

了解：
- 数据集信息
- 检测方法
- 分割方法
- 评估指标

## 5. 开始工作

### 5.1 选择合适角色

| 任务类型 | 推荐角色 |
|---------|---------|
| 模型训练/优化 | Deep Learning Model Optimization Engineer |
| 评估指标开发 | Bioimage Analysis Evaluation Architect |
| 文档记录 | Research Documentation Architect |
| 代码调试 | Python Developer |

### 5.2 使用工作流

可用的工作流 (使用 `/workflow-name` 调用):
- `/log-experiment` - 记录新实验
- `/update-progress` - 更新项目进展
- `/run-evaluation` - 运行模型评估

### 5.3 遵循规范

1. 所有实验必须记录到 `experiments_log.md`
2. 重要决策必须记录原因
3. 代码修改后运行测试验证
4. 保持文档一致性

## 6. 项目关键路径

```
数据 (Allen TIFF) 
    ↓
DAPI 核检测 (F1=0.75)
    ↓
CellSAM 分割 (Dice=0.82)
    ↓
评估指标 (PQ, AJI, RI)
    ↓
[待做] SarcGraph 集成
```

## 7. 关键文件位置

| 类型 | 路径 |
|------|------|
| 项目蓝图 | `CLAUDE.md` |
| 实验记录 | `anti_test/experiments_log.md` |
| 结果汇总 | `anti_test/results_summary.md` |
| 评估代码 | `anti_test/eval_metrics.py` |
| 训练代码 | `train_expanded.py`, `finetune_boundary_simple.py` |
| 最佳模型 | `checkpoints/boundary_20260111_012636/best_model.pt` |
