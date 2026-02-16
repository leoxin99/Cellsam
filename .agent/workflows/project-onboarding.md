---
description: AI 助手加入项目时的入门指南
---

# AI 助手入门指南

新 AI 助手（无论是实施 Agent 还是审核 Agent）加入本项目时，按以下顺序阅读。

## 1. 阅读项目蓝图

// turbo
读取 `CLAUDE.md`，重点关注：
- 项目状态仪表板（当前阶段、关键指标）
- 核心文档表（哪些文档是 Active）
- 多 Agent 协作模式（你的角色和约束）
- AI 工作规范（审查制度、禁止估算原则）

## 2. 了解协作规则

// turbo
读取 `docs/agent_management.md`，重点关注：
- §1 你的 Agent ID 和角色（A1/A2/R1）
- §2 层级结构与决策权
- §3.2 产物提交格式（8 个必填字段）
- §4.1 文件所有权（哪些文件你可以写）
- §4.2 并发冲突防护（A 模式约束）

## 3. 查看通信信箱

// turbo
读取 `docs/agent_inbox.md`：
- 查看是否有发给你的未处理消息
- 了解消息格式（日期 + 发送方 → 接收方 + 必填字段）

## 4. 了解当前任务

// turbo
读取 `docs/task_backlog.md`：
- 确认哪些任务分配给你
- 了解每个任务的完成标准和产物要求

## 5. 按角色读 SSOT 文档

| 你的角色 | 额外必读 |
|---------|---------|
| **实施 Agent (A1/A2)** | `docs/inference_standard.md` (推理口径) + `docs/code_inventory.md` (代码入口) + `docs/error_log_and_checklist.md` (训练前检查) |
| **审核 Agent (R1)** | `.agent/workflows/review-agent.md` (审核流程) + `docs/dapi_detection_design.md` (检测参数) |

## 6. 开始工作

### 6.1 任务完成后必做

1. 将产物摘要**追加**到 `docs/agent_inbox.md`（格式参考已有条目）
2. 运行回归测试 `python tools/test_unified_regression.py`
3. 确保修改已 commit

### 6.2 可用工作流

| 命令 | 用途 |
|------|------|
| `/review-agent` | 审核 Agent 执行审核流程 |
| `/cellsam-commands` | 常用命令速查 (含安全级别) |
| `/daily-github-sync` | 每日提交到 GitHub |
| `/project-onboarding` | 本入职指南 |

## 7. 项目关键路径

```
DAPI 核检测         → F1=0.8033 (test73 封板)
    ↓
CellSAM 分割        → Oracle BM-Dice=0.6954, PQ=0.4641 (Phase 1 锁定)
    ↓
Phase 2 结构改进     → L_neighbor + L_overlap (P2-A 训练中)
    ↓
Phase 3 三通道适配   → Channel Adapter (待开始)
```
