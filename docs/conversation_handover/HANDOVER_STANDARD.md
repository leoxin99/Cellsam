# 对话交接规范 (Conversation Handover Standard)

> 状态: 🟢 Active  
> 生效日期: 2026-03-06  
> 目的: 当旧对话窗口上下文过长或角色切换时，确保新窗口无信息断层。

---

## 1. 触发条件

满足任一条件即必须执行交接:

1. 对话窗口上下文接近上限，准备废弃旧窗口。
2. Agent 更换执行窗口（A1/A2/R1 任一）。
3. 任务跨阶段切换（如从实验执行转入论文封板）。
4. 关键结论已形成，需要固化到 SSOT 文档后再继续。

---

## 2. 关窗前必做清单 (强制)

旧窗口结束前，执行者必须完成以下同步:

1. `docs/task_backlog.md`  
状态、优先级、完成标准、下一步任务更新为最新。
2. `CLAUDE.md`  
项目状态仪表板与当前重点任务同步。
3. `docs/experiments_log.md` + 对应实验单文档  
实验设计、命令、结果、结论至少一处可追溯。
4. 相关 SSOT 文档  
涉及参数/流程/指标变更时，回填对应 SSOT（例如 `inference_standard.md`、`dataset_parameters.md`、`dapi_detection_design.md`）。
5. `docs/agent_inbox.md`  
向 A2/R1 发送“本窗口完成摘要 + 下一窗口入口”通知。
6. 本目录下的角色交接文档  
写入本次交接记录（见 §4 命名规则）。

---

## 3. 重要文档与更新门禁

| 文档 | 角色 | 门禁规则 |
|------|------|----------|
| `CLAUDE.md` | 全局总览 SSOT | 每次阶段状态变更必须更新 |
| `docs/task_backlog.md` | 执行清单 SSOT | 每次任务状态变化必须更新 |
| `docs/experiments_log.md` | 实验流水账 SSOT | 每次实验完成后更新 |
| `docs/experiments/active/*.md` | 实验单文档 | 新实验必须先建文档 |
| `docs/agent_inbox.md` | Agent 异步通信 | 每次任务交接必须更新 |
| `docs/agent_management.md` | 协作规则 SSOT | 规则变化时更新 |
| `.agent/workflows/project-onboarding.md` | 新窗口入口 | 交接流程变化时更新 |
| `docs/agent_status.md` | 实时看板 | 至少每周一次全局刷新 |
| `docs/paper_preparation.md` | 论文素材总文档 | 关键结论定稿后更新 |

---

## 4. 交接目录与命名

目录固定:

- `docs/conversation_handover/A1/`
- `docs/conversation_handover/A2/`
- `docs/conversation_handover/R1/`

文件命名:

- `handover_XXX_YYYY-MM-DD.md`
- `XXX` 为三位递增序号（001, 002, ...）

---

## 5. 交接文档模板

```markdown
# <Agent> 交接记录 #XXX (YYYY-MM-DD HH:mm)

## 1. 本窗口完成
- ...

## 2. 进行中任务
- ...

## 3. 下一窗口第一步
1. ...
2. ...

## 4. 风险与未决事项
- ...

## 5. 关键证据/产物
- 代码:
- 结果:
- 文档:
```

---

## 6. 新窗口必读顺序 (Project Onboarding 对齐)

1. `CLAUDE.md`
2. `docs/task_backlog.md`
3. `docs/agent_inbox.md` 最新 3-5 条
4. `docs/conversation_handover/HANDOVER_STANDARD.md`
5. 自身角色目录下最新 `handover_*.md`

---

## 7. 执行约束

1. 未完成 §2 清单，不得宣告“交接完成”。
2. 如果实验/参数有冲突信息，以 SSOT 文档为准，交接文档仅做导航。
3. 交接后 24 小时内若发现遗漏，必须补一条 inbox 更正消息。
