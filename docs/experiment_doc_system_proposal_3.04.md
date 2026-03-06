# [Codex | 2026-03-04] 实验文档体系改造提案

## 1. 问题

当前实验记录主要堆在 `docs/experiments_log.md`，随着实验数增加，已经出现三个结构性问题：

1. **单文档承载过多角色**
   - 既是时间流水账
   - 又承担方案说明
   - 又承担结果汇总
   - 又承担历史解释
2. **单个实验不可独立追溯**
   - 很难快速回答“这个实验的固定条件、变量、脚本、配置、结果、结论分别是什么”
3. **后续回填成本越来越高**
   - 每做一次实验都要在长文档中找插入点
   - 容易造成旧口径残留和局部冲突

因此，建议把“实验索引”和“实验详情”拆开。

## 2. 目标

新体系需要同时满足：

1. 每个实验有独立文档，能单独阅读
2. `experiments_log.md` 保留，但降级为索引/流水账
3. 实验开始前有方案，实验结束后在同一文档补结果
4. 新对话能快速定位当前关键实验
5. 不打破现有 `task_backlog.md` / `CLAUDE.md` / `agent_inbox.md` 的协作流程

## 3. 方案概览

### 3.1 文档分层

建议改成三层：

#### A. 实验索引层

- `docs/experiments_log.md`

只保留：

1. 实验 ID
2. 日期
3. 一句话结论
4. 状态
5. 指向独立实验文档的链接

不再在这里详细展开实验设计和长篇分析。

#### B. 实验详情层

新建目录：

- `docs/experiments/active/`
- `docs/experiments/completed/`
- `docs/experiments/templates/`

每个实验一个独立文档，例如：

- `docs/experiments/active/T31_cellpose_paper_aligned.md`
- `docs/experiments/completed/T27a_planb_decoder_only.md`

每个文档同时承载：

1. 实验背景
2. 假设
3. 固定条件
4. 改变量
5. 执行命令/脚本/配置
6. 结果
7. 结论
8. 下一步

#### C. 任务入口层

- `docs/task_backlog.md`
- `CLAUDE.md`

只负责：

1. 当前优先级
2. 指向实验详情文档
3. 当前状态摘要

也就是说：

- `task_backlog.md` 决定“先做什么”
- `experiments_log.md` 决定“做过什么”
- 独立实验文档负责“这个实验到底是什么”

## 4. 单个实验文档模板

建议固定模板如下。

```md
# T31 Cellpose Paper-Aligned Baseline

## 1. Metadata
- ID:
- Status:
- Owner:
- Priority:
- Related task:
- Related config:
- Related script:
- Related output dir:

## 2. Background

## 3. Question / Hypothesis

## 4. Fixed Conditions

## 5. Variables

## 6. Execution Plan

## 7. Expected Risks

## 8. Results
- Run 1:
- Run 2:
- Aggregate:

## 9. Interpretation

## 10. Decision
- Keep / drop / follow-up
```

## 5. 生命周期

建议一个实验文档有固定状态流转：

1. `Draft`
2. `Approved`
3. `Running`
4. `Completed`
5. `Archived`

规则：

1. 开跑前必须先有 `Draft`
2. 开跑后把状态改成 `Running`
3. 结果回填后改成 `Completed`
4. 如实验被废弃，状态写 `Archived` 并注明原因

## 6. 命名规则

建议命名统一为：

- `T31_cellpose_paper_aligned.md`
- `T30_lora_encoder.md`
- `T29_channel_encoding.md`

规则：

1. 文件名前缀保留实验 ID
2. 文件名只描述实验主题，不写结果
3. 结果写进文档正文，不写进文件名

## 7. 与现有文档的关系

### `docs/experiments_log.md`

改成：

- 索引页
- 一行摘要
- 指向实验详情文档

例如：

| ID | 日期 | 实验 | 结果摘要 | 详情 |
|----|------|------|---------|------|
| T31 | 2026-03-04 | Cellpose paper-aligned baseline | Running | `docs/experiments/active/T31_cellpose_paper_aligned.md` |

### `docs/task_backlog.md`

只写：

- 优先级
- 一句话目标
- 详情文档链接

### `CLAUDE.md`

只保留：

- 当前高优先级实验列表
- 每个实验的状态
- 指向独立实验文档

不再在 `CLAUDE.md` 里展开长篇实验内容。

## 8. 迁移策略

不建议一次性重构全部历史实验。建议分两步：

### Phase A: 新实验先执行新体系

从现在开始，新实验按新体系执行：

1. T31 Cellpose baseline 重跑
2. 后续 LoRA / channel / CellFinder 相关实验

### Phase B: 只迁移关键历史实验

从历史实验中只迁移最重要的几项：

1. `T27a`
2. `T28`
3. `T29`
4. `T30`
5. `T31`
6. `T16` baseline

其余仍保留在 `experiments_log.md`，不强制补全。

## 9. 最小落地方案

如果你确认采用，本轮建议最小改动是：

1. 新建目录：
   - `docs/experiments/active/`
   - `docs/experiments/completed/`
2. 新建模板：
   - `docs/experiments/templates/experiment_template.md`
3. 先把 `T31` 做成第一份标准实验文档
4. 在 `task_backlog.md` 和 `CLAUDE.md` 中都只保留链接
5. `experiments_log.md` 从下一个实验开始只写摘要索引

## 10. 预期收益

1. 单个实验可独立阅读
2. 新对话更容易接手
3. 实验与结果不再混在一个大流水账里
4. 旧口径残留更容易发现
5. 更适合后续论文写作和审计

## 11. 当前建议

当前建议先不要大规模迁移历史文档。

更稳妥的做法是：

1. 你先确认这个体系
2. 然后只对 `T31` 和后续新实验启用
3. 等跑通一轮后，再决定是否回迁 `T27a/T28/T29/T30`
