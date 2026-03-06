# R1 (Reviewer) 交接文档

> **生成时间**: 2026-02-25 04:20  
> **用途**: 如果 R1 对话窗口需要重开，新 R1 读此文档快速上手  
> **角色**: R1 = 审核者 (Reviewer)，负责审核 A1/A2 的方案和产物，做技术决策

---

## 1. 我是谁

- **角色**: R1 (Reviewer) — 项目的技术审核者
- **职责**: 审核 A1(Codex)/A2(Claude) 提交的方案、代码、实验结果；做技术裁决；写分析报告
- **不做**: 不直接写训练代码、不提交 SLURM 任务、不修改核心 Python 文件
- **沟通方式**: 通过 `docs/agent_inbox.md` 与 A1/A2 异步通信

---

## 2. 已做的关键技术决策

### 训练策略

| 决策 | 结论 | 记录位置 |
|------|------|---------|
| Neck-only 训练 (A1 提案) | ⛔ 不推荐 — CellSAM Stage2 的 neck 对齐动机不适用于我们 | `docs/technical/update_cellsam.md` §8 |
| LoRA 微调 encoder | ✅ 推荐 (P1) — 小数据场景文献最优解，~4.5M 参数 | `docs/technical/update_cellsam.md` §9 |
| 微调 CellFinder | ⚠️ P3 优先级 — 可行但风险高 (~1-2 周), 训练管道需从零构建 | `docs/technical/update_cellsam.md` §11 |

### Loss 设计

| 决策 | 结论 | 记录位置 |
|------|------|---------|
| ContourLoss | ❌ 有害 (+2.3pp PQ when removed) | `experiments_log.md` T12 |
| pos_weight=10 vs 2 | ✅ posw=10 显著更优 (+4.1pp) | `experiments_log.md` T12 |
| Best Config | posw=10 + contour=off → PQ=0.484 (4-run mean) | `CLAUDE.md` L37-42 |
| N/O Loss (Phase 2) | ⛔ 终止 — Fix1-3 均证实退化 | `task_backlog.md` T5 |

### 三通道实验

| 决策 | 结论 | 记录位置 |
|------|------|---------|
| 通道顺序 | R=BF, G=Actn2, B=DAPI (生物学一致) | inbox 归档 02-24 16:43 |
| 2ch B 通道 | BF 复制 (与 BF-only 基线一致) | inbox 归档 02-24 16:43 |
| T18-C 无 adapter | ✅ 增加 (隔离 adapter 贡献) | inbox 归档 02-24 16:43 |
| lr | 5e-5 先跑，收敛慢再改 1e-4 | inbox 归档 02-24 16:43 |

### 框外分割

| 决策 | 结论 | 记录位置 |
|------|------|---------|
| Box Clipping | 保留 — with_clip PQ=0.466 > no_clip 0.437 | `docs/technical/update_cellsam.md` §10 |
| 优化方向 | LoRA 是最有希望的路径 (让模型学会自然框外抑制) | `docs/technical/update_cellsam.md` §10.5 |

### T16 Baseline

| 决策 | 结论 | 记录位置 |
|------|------|---------|
| MedSAM > Ours (PQ 0.576 vs 0.484) | 论文策略: Oracle vs E2E 分组, 强调数据效率 + E2E 唯一性 | inbox 归档 02-21 14:15 |
| Cellpose diameter=200 | 建议补一行作脚注 | inbox 归档 02-21 14:15 |
| StarDist | P3 暂缓 (6 行数据够论文) | inbox 归档 02-21 14:15 |

### 文档管理

| 决策 | 结论 |
|------|------|
| paper_preparation + writing_plan | 合并为 paper_preparation.md (方案 A) |
| project_guide.md | 全面重写 (150 行, 引用 SSOT) |
| CLAUDE.md 更新 | 保留 Phase1 行 + 新增 Best Config 行 |
| inbox 归档规则 | 保留最近一周存根, 完整内容存 inbox_archive/ |

---

## 3. 当前实验状态

| 实验 | 状态 | 关键数据 |
|------|:----:|---------|
| Best Config | ✅ | PQ=0.484, 4 runs mean |
| T18-A (2ch) | 🔄 | seed42 PQ=0.496, seed123+A100 pending |
| T18-B (3ch) | 🔄 | seed42 PQ=0.498, seed123+A100 pending |
| T18-C (no adapter) | 🔄 | training |
| T16 Baseline | ✅ | 6 methods, MedSAM PQ=0.576 为上限 |
| T17 Training Curves | ✅ 工具 | Phase1 图 done, Best Config 日志待下载 |
| T20 Attention Vis | ✅ 脚本 | 待 T18 完成后执行 |

---

## 4. 待新 R1 审核/执行的任务

| 优先级 | 任务 | 下一步 |
|:------:|------|--------|
| **P0** | T18 结果收集 | 等 A100/seed123 训练完成 → 审核结果 |
| **P1** | LoRA encoder 方案审核 | A2 提交 LoRA 实现后审核 |
| **P1** | T17 Best Config 曲线 | A2 下载日志 → 审核图表 |
| P2 | T20 注意力可视化 | T18 完成后 → 审核 BF vs 3ch 对比 |
| P2 | project_guide.md 重写 | A2 执行后审核 |

---

## 5. 必读文档 (按优先级)

1. **`CLAUDE.md`** — 项目总览 + 当前指标 + 工作重点
2. **`docs/technical/update_cellsam.md`** — CellSAM 技术分析 (R1 的核心产出)
3. **`docs/agent_inbox.md`** — 当前活跃消息存根 + 归档链接
4. **`docs/task_backlog.md`** — 按优先级排列的任务清单
5. **`docs/experiments_log.md`** — 实验记录 (T12 消融, Best Config, T18)
6. **`docs/paper_preparation.md`** — 论文素材库 (指标/方法/实验数据)

---

## 6. 协作规则

- **inbox 格式**: 新消息追加到最前，格式 `## [日期] 发→收 — 标题`
- **inbox 归档**: 审核+执行完毕后缩为存根，完整内容在 `inbox_archive/`
- **backlog 更新**: 修改任务时必须同步更新顶部目录
- **Agent 规则**: 信息不足时 Agent 须主动索要 (`agent_management.md` §3.5)
- **evidence 口径**: 论文事实 vs 代码可证事实 分开标注 (`docs/technical/update_cellsam.md` Appendix B)

