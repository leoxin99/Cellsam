# AI Handoff Protocol (可复用模板)

> **版本**: v1.0 | **创建日期**: 2026-01-28
> **用途**: 确保 AI 助手之间的项目交接无缝衔接

---

## 一、交接前准备 (For Outgoing Agent)

### 1.1 必须更新的文件

| 文件 | 内容 | 优先级 |
|------|------|-------|
| `CLAUDE.md` | 项目状态仪表板、当前阶段、关键决策 | **P0** |
| `docs/design_decisions.md` | 所有技术决策的"为什么" | P1 |
| `anti_test/experiments_log.md` | 实验记录 | P1 |

### 1.2 检查清单

- [ ] `CLAUDE.md` 日期已更新为当前日期
- [ ] 任务列表中的已完成项目已勾选 `[x]`
- [ ] 当前进行中的任务标记为 `[/]`
- [ ] 关键代码路径已在 Deep Link Index 中列出
- [ ] 未解决的问题/Blockers 已记录

---

## 二、文档结构 (Standard)

```
CellSam/
├── CLAUDE.md                    ← 🏠 AI 入口文档 (必读)
├── docs/
│   ├── design_decisions.md      ← 技术决策记录
│   ├── dataset_parameters.md    ← 数据集统计
│   ├── claude_pipeline_analysis.md  ← 三通道设计
│   ├── technical_details.md     ← 实现细节
│   ├── troubleshooting.md       ← 常见问题
│   └── archive/                 ← 过时文档存档
└── anti_test/experiments_log.md ← 实验日志
```

---

## 三、新 AI 接入流程 (For Incoming Agent)

### Step 1: 读取入口文档
```
📖 首先阅读 CLAUDE.md
```

### Step 2: 验证上下文
根据 `CLAUDE.md` 中的 **Deep Link Index** 查阅相关文档：

| 主题 | 文档 |
|------|------|
| 三通道设计 | `docs/claude_pipeline_analysis.md` |
| 数据集统计 | `docs/dataset_parameters.md` |
| 检测逻辑 | `src/detection/dapi.py` |
| 设计原因 | `docs/design_decisions.md` |

### Step 3: 确认当前任务
从 `CLAUDE.md` 任务列表中找到 `[/]` 标记的任务，这是当前进行中的工作。

### Step 4: 开始工作
1. 执行任务
2. 记录实验到 `anti_test/experiments_log.md`
3. 更新 `CLAUDE.md` 仪表板

---

## 四、交接 Prompt 模板

将以下内容发送给新 AI 以快速建立上下文：

```markdown
# Role
You are an expert BioImage Analysis assistant working on the **CellSAM** project.

# Project Goal
Adapt SAM (Segment Anything Model) for high-precision cardiomyocyte segmentation.

# Entry Point
**First, read `CLAUDE.md`** - it contains the project dashboard, current status, and all critical links.

# Current Phase
[从 CLAUDE.md 复制当前阶段]

# Immediate Task
[从 CLAUDE.md 复制当前 [/] 任务]

# Key Constraints
1. Image Encoder is FROZEN (only Mask Decoder and Adapter are trainable)
2. Detection uses Hybrid DAPI+Actn2 (not CellFinder)
3. Parameters are DATA-DRIVEN (check docs/dataset_parameters.md)
```

---

## 五、版本控制

### 文档归档规则

| 条件 | 操作 |
|------|------|
| 文档内容已完全整合到其他文档 | 移至 `docs/archive/` |
| 进度报告超过 7 天 | 移至 `docs/archive/` |
| 实验被明确标记为"废弃" | 移至 `docs/archive/` |

### 归档命名约定
```
docs/archive/
├── handoff_20260111.md          ← 带日期的旧交接文档
├── progress_report_20260108.md  ← 旧进度报告
└── ...
```

---

## 六、常见问题

### Q1: 新 AI 应该从哪个文件开始？
**A**: 始终从 `CLAUDE.md` 开始。它是项目的"大脑"，包含所有关键链接。

### Q2: 发现文档之间有矛盾怎么办？
**A**: 以 `CLAUDE.md` 为准。如果 `CLAUDE.md` 与代码矛盾，以代码为准。

### Q3: 如何记录实验？
**A**: 使用 `anti_test/experiments_log.md`，遵循已有格式。

### Q4: 旧文档可以删除吗？
**A**: 不要删除。移至 `docs/archive/` 保留历史记录。

---

## 七、更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-28 | v1.0 初始版本 |
