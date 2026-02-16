---
name: experiment-analysis
description: 标准化分析实验结果 JSON，生成结构化报告
---

# Experiment Analysis Skill

当需要分析实验结果时，使用以下标准流程。

## 触发条件

当用户提供实验结果（JSON/日志/数值）并要求分析时自动激活。

## 分析流程

### Step 1: 读取数据
1. 读取实验结果文件（通常为 `experiments/<name>/results.json`）
2. 确认实验元信息: split, 搜索空间, 评估策略

### Step 2: 生成 Summary 表

```markdown
| 项目 | 值 |
|------|-----|
| 实验 ID | E34b |
| 日期 | 2026-02-14 |
| 目标 | 联合消融 edge/merge 参数 |
| Split | val(71) |
| 搜索空间 | edge_margin×size_ratio×merge_coeff = N 组合 |
| 最优参数 | edge=20, ratio=2.5, coeff=1.4 |
| 最优指标 | F1=0.8106, P=0.7639, R=0.8633 |
| 二优参数 | ... |
| 二优指标 | ... |
| Δ(最优-二优) | ... |
```

### Step 3: Config Diff 表 (对比上一版)

```markdown
| 参数 | 旧值 | 新值 | 变化 |
|------|------|------|------|
| edge_margin | 32 | 20 | -37.5% |
| merge_coeff | 1.2 | 1.4 | +16.7% |
```

### Step 4: Training Curve 表 (训练实验时)

如果是训练实验，提取 epoch 级数据:

```markdown
| Epoch | Train Loss | Val Dice | Val PQ | Best? |
|-------|-----------|----------|--------|-------|
| 10    | 0.234     | 0.651    | 0.312  |       |
| 20    | 0.198     | 0.673    | 0.389  | ✅    |
| 30    | 0.201     | 0.668    | 0.375  |       |
```

### Step 5: 关键发现

分析以下维度并明确记录:
- **参数敏感性**: 哪个参数对目标指标影响最大？
- **退化风险**: 是否有参数 ≥ 某值后指标不再变化（无效区间）？
- **泛化性**: val→test 的 Δ 是否 < 2pp？
- **显著性判断**: 最优与二优的差距是否显著（>1pp F1 / >0.5pp PQ）？

### Step 6: 结论与建议

```markdown
## 结论
- [通过/有条件通过/不通过]
- 核心发现摘要

## SSOT 回填清单 (审核通过后执行)
- [ ] `claude.md` — 更新 Step X 状态
- [ ] `task_backlog.md` — 标记 Tx Completed
- [ ] `experiments_log.md` — 追加实验记录
- [ ] `dapi_detection_design.md` — 锁定参数 (如适用)
```

## 输出格式约束

- 所有文件引用使用**仓库相对路径** (如 `src/detection/dapi.py:71`)
- 数值保留 **4 位小数** (如 F1=0.8106)
- 百分比变化保留 **1 位小数** (如 +16.7%)
- 口径限定: 写"在当前 valN + 当前搜索空间下"，不做无证据的泛化断言
