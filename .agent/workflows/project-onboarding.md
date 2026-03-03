---
description: AI 助手加入项目时的入门指南
---

# AI 助手入门指南

新 AI 助手加入本项目时，按以下顺序阅读。

## 1. 阅读项目蓝图

// turbo
读取 `CLAUDE.md`，重点关注：
- 项目状态仪表板（当前阶段、关键指标）
- 核心文档表（哪些文档是 Active）
- 多 Agent 协作模式（你的角色和约束）
- AI 工作规范（审查制度、禁止估算原则）

## 2. 阅读移交文档 (如有)

// turbo
根据你的角色查找对应的移交文档：

| 角色 | 移交文档 | 说明 |
|------|---------|------|
| **A2 (Claude)** | `docs/a2_handoff_20260225.md` | 上一任 A2 的完整工作移交 |
| **R1 (Reviewer)** | `docs/r1_handoff.md` | R1 的技术决策记录 |
| **A1 (Codex)** | 无专属移交 | 直接看 §3-4 |

移交文档包含：已完成工作、进行中任务、关键上下文、ALICE HPC 信息。

## 3. 了解协作规则

// turbo
读取 `docs/agent_management.md`，重点关注：
- §1.2 你的职能边界 (A1: 数据/参数/检测; A2: 训练/SLURM/评估; R1: 审核)
- §2 层级结构与决策权
- §3.1 通信信道 + **inbox 规则 #5** (R1 审核必须先写 inbox)
- §4.1 文件所有权（哪些文件你可以写）

## 4. 查看通信信箱

// turbo
读取 `docs/agent_inbox.md`：
- **只看头部 3-5 条消息**，了解最新审核状态和 R1 指令
- 注意归档规则 (§顶部 5 条规则)
- 历史消息在 `docs/inbox_archive/`

## 5. 了解当前任务

// turbo
读取 `docs/task_backlog.md`：
- 确认哪些任务分配给你 / 标记为你的优先级
- 了解每个任务的完成标准和产物要求
- 查看 §4 已完成任务（避免重复劳动）

## 6. 按角色读 SSOT 文档

| 你的角色 | 额外必读 |
|---------|---------|
| **A1 (Codex)** | `docs/dataset_parameters.md` (数据集参数) + `docs/inference_standard.md` (推理口径) |
| **A2 (Claude)** | `docs/experiments_log.md` (实验记录) + `docs/update_cellsam.md` (技术分析) + `docs/paper_preparation.md` (论文素材) |
| **R1 (Reviewer)** | `.agent/workflows/review-agent.md` (审核流程) + `docs/experiments_log.md` (实验数据) |

## 7. 查看实时状态

// turbo
读取 `docs/agent_status.md`：
- 其他 Agent 当前在做什么
- 活跃 ALICE Jobs（哪些实验在跑）
- 近期审核闭环记录

## 8. 开始工作

### 8.1 产物提交后必做

1. 将产物摘要**追加**到 `docs/agent_inbox.md` 顶部（格式参考已有条目）
2. 更新 `docs/agent_status.md` 中你自己的行
3. 确保修改已 commit

### 8.2 可用工作流

| 命令 | 用途 |
|------|------|
| `/review-agent` | 审核 Agent 执行审核流程 |
| `/cellsam-commands` | 常用命令速查 (含安全级别) |
| `/daily-github-sync` | 每日提交到 GitHub |
| `/project-onboarding` | 本入职指南 |

## 9. 项目关键路径 (2026-02-25 更新)

```
DAPI 核检测              → F1=0.8106 (val71 锁定, profiles.py)
    ↓
CellSAM 分割 (BF-only)  → Oracle PQ=0.484 (Best Config: posw=10, contour=off)
    ↓
三通道实验 (T18)         → PQ≈0.495~0.500 (⚠️ 待对照组确认非训练混淆)
    ↓                       Job 1036799 (T18-C s123) + Job 1036827 (对照组)
论文撰写                 → 素材收集 + 消融表整理
```
