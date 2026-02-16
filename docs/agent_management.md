# 多 Agent 协作管理规范

> 状态: 🟢 Active  
> 创建日期: 2026-02-15  
> SSOT 级别: 本文件是多 Agent 协作的唯一权威参考  

---

## 1. Agent 清单与职能

### 1.1 当前 Agent 配置

| ID | 名称 | 平台 | 运行方式 | 角色 |
|----|------|------|---------|------|
| **A1** | **Codex** | Antigravity (VS Code) | 交互式对话 | 实施 Agent |
| **A2** | **Claude** | Antigravity (VS Code) | 交互式对话 | 实施 Agent |
| **R1** | **Reviewer** | Antigravity (VS Code) | 交互式对话 | 审核 Agent |

> 三个 Agent 均为独立的 Antigravity 对话窗口，共享同一个 Git 仓库。  
> **Skills 自动发现**: `.claude/skills/` 和 `.github/skills/` 中的 Skill 文件在**新对话启动时自动发现**（已验证），Agent 会根据任务自动匹配加载。已有对话内创建的 Skill 需要重开对话才能被发现。

### 1.2 职能边界

```
┌─────────────────────────────────────────────────┐
│                   用户（决策者）                   │
│           最终审批 · 任务分配 · 消息中转             │
└────────┬──────────────┬──────────────┬───────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐
    │ Codex   │   │  Claude   │  │ Reviewer │
    │  (A1)   │   │   (A2)    │  │   (R1)   │
    │ 实施     │   │  实施     │  │  审核     │
    └─────────┘   └───────────┘  └──────────┘
```

| 角色 | 可以做 | 不可以做 |
|------|--------|----------|
| **实施 Agent (A1/A2)** | 修改 `src/`; 编写脚本 `tools/`; 运行实验; 创建 config | 自行封板重大参数; 直接修改 SSOT 文档的"已锁定"字段 |
| **审核 Agent (R1)** | 读取所有代码/产物; 写审核报告; 回填 SSOT 文档; 更新 `task_backlog.md` | 修改 `src/` 核心代码; 修改训练配置 YAML; 运行实验 |
| **用户** | 最终审批; 分配任务; 在 Agent 之间传递消息 | — |

---

## 2. 层级结构与决策权

```
Level 0  用户                        → 最终决策权，Go/No-Go
Level 1  审核 Agent (R1)             → 质量门禁，SSOT 回填权
Level 2  实施 Agent (A1, A2)         → 代码与实验执行权
```

**决策升级规则**:
- 实施 Agent 完成任务 → 产物交用户 → 用户转交审核 Agent
- 审核 Agent 通过 → 回填文档 → 闭环
- 审核 Agent 不通过 → 问题清单交用户 → 用户转交实施 Agent 修复
- 重大方向变更 (如放弃 Phase 2) → 必须由用户决定

---

## 3. 沟通机制

### 3.1 通信拓扑

Agent 之间**无法直接通信**（各自独立的 Antigravity 窗口）。通过三个信道协作:

| 信道 | 实时性 | 内容 |
|------|--------|------|
| **用户口头中转** | 实时 | 紧急指令、Go/No-Go 决策 |
| **Agent 信箱 `docs/agent_inbox.md`** | 异步 | 产物摘要、审核结论、任务交接 |
| **共享文件系统 (Git Repo)** | 异步 | 代码、实验结果 JSON、文档 |

### 3.2 产物提交格式 (实施 Agent → 用户 → 审核 Agent)

实施 Agent 完成任务后，向用户提交的摘要**必须包含**:

| 字段 | 说明 | 示例 |
|------|------|------|
| `commit_sha` | 最近一次相关 commit | `a3f2c1d` |
| `cmd` | 执行的命令 | `python tools/ablation_detection_e34b.py` |
| `config_path` | 使用的配置文件 | `src/config/phase2a_neighbor_overlap.yaml` |
| `split` | 数据划分 | `val(71)` / `test(73)` |
| `output_path` | 产物文件路径 | `experiments/ablation_detection_e34b/results.json` |
| `key_metrics` | 关键数值 | `F1=0.8106, P=0.7639, R=0.8633` |
| `regression` | 回归测试结果 | `10 passed, 0 failed` |
| `modified_files` | 修改的文件列表 | `src/detection/dapi.py:71,231,539` |

### 3.3 审核报告格式 (审核 Agent 输出)

文件路径: `docs/temp_reviews/<experiment>_review.md`

必含章节:
1. **审核结论** — 通过 / 有条件通过 / 不通过
2. **代码验证** — 参数链路、逻辑正确性
3. **实验验证** — 数值核对、搜索空间覆盖、策略合规
4. **关键发现** — 如参数退化、泛化性分析
5. **下一步建议** — 回填清单或修复清单

---

## 4. 文件与文档管理

### 4.1 文件所有权

| 文件/目录 | 主要写入者 | 次要写入者 | 说明 |
|-----------|----------|----------|------|
| `src/` | 实施 Agent | — | 审核 Agent 只读 |
| `tools/` | 实施 Agent | — | 审核 Agent 只读 |
| `scripts/` | 实施 Agent | — | SLURM 脚本 |
| `src/config/` | 实施 Agent | — | 训练配置 |
| `experiments/` | 实施 Agent (自动写盘) | — | 实验结果 JSON |
| `docs/temp_reviews/` | 审核 Agent | 实施 Agent (复核) | 临时审核报告 |
| `CLAUDE.md` | 审核 Agent (回填) | 实施 Agent (紧急修正) | 项目总览 SSOT |
| `docs/task_backlog.md` | 审核 Agent (勾选) | 用户 (新增任务) | 待办清单 |
| `docs/experiments_log.md` | 审核 Agent (回填) | 实施 Agent (初始记录) | 实验流水账 |
| `docs/dapi_detection_design.md` | 审核 Agent (锁定标记) | 实施 Agent (设计更新) | 检测参数 SSOT |
| `docs/agent_management.md` | 审核 Agent | — | 本文件 |
| `.agent/workflows/` | 审核 Agent | 用户 | 工作流定义 |

### 4.2 并发冲突防护 (A 模式约束)

审核 Agent 执行文档回填**前**，必须确认:

1. 实施 Agent 无未 commit 的文档修改（用户口头确认或 `git status` 干净）
2. 回填范围声明：审核 Agent 在开始回填前列出将要修改的文件清单
3. 回填完成后通知用户，用户可转告实施 Agent "文档已更新，请 `git pull`"

### 4.3 `docs/temp_reviews/` 清理规则

- 审核通过 + 结论已合并进 SSOT 文档 → **可删除**对应报告
- 审核不通过 → 保留至修复闭环后再删
- 建议月度清理

---

## 5. 约束文件索引

| 文件 | 约束内容 | 适用对象 |
|------|---------|---------|
| `CLAUDE.md` | 项目总览、AI 工作规范、禁止估算原则 | 所有 Agent |
| `docs/agent_inbox.md` | 产物摘要、审核结论、任务交接 | 所有 Agent |
| `docs/agent_management.md` (本文件) | Agent 职能、通信、文件所有权 | 所有 Agent |
| `docs/agent_status.md` | 实时任务状态板（谁在做什么） | 所有 Agent |
| `docs/task_backlog.md` | 可执行任务、完成标准 | 所有 Agent |
| `.agent/workflows/review-agent.md` | 审核流程步骤 | 审核 Agent |
| `docs/error_log_and_checklist.md` | 训练前检查清单 | 实施 Agent |
| `docs/inference_standard.md` | 推理评估口径 | 实施 Agent |

**新 Agent 入职必读顺序**:
1. `CLAUDE.md` — 项目背景与规范
2. `docs/agent_management.md` — 协作规则
3. `docs/task_backlog.md` — 当前待办
4. 按角色读对应 SSOT 文档

---

## 6. 更新日志

| 日期 | 内容 |
|------|------|
| 2026-02-15 | 初版创建: Agent 清单、职能边界、通信协议、文件所有权、并发防护 |
| 2026-02-15 | 修正: 三个 Agent 均为 Antigravity (VS Code) 交互式对话，非 Codex 异步平台 |
| 2026-02-15 | 新增: `agent_status.md` 实时状态板，写入约束文件索引 |
