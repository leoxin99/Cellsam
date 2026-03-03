# Agent Team 架构优化方案

> ⚠️ **状态**: 🟡 Historical — 本文档为 2026-02-15 的架构提案，方案 A 已实施至 `agent_management.md`  
> **实施状态**: 改动 1 ✅ (A1/A2 按任务线分工) · 改动 2 ❌ (JSON 格式未采用，保持 markdown) · 改动 3 ✅ (`agent_status.md`) · 改动 4 ✅ (R1 主动读 inbox)  
> **方案 B/C**: 未实施，长期参考  
>  
> 初版修正声明 (2026-02-15):  
> 本方案基于"A1(Codex)为异步平台"的假设，实际三个 Agent 均为 **Antigravity (VS Code) 交互式对话**。  
> 已由 R1+A1 联合审核修正，`agent_management.md` 中已是正确版本。以下 `[已修正]` 标注受影响内容。

## 1. 当前架构分析

### 1.1 现有拓扑

```
                ┌─────────┐
                │  用户    │  ← 中转所有消息
                └──┬─┬─┬──┘
                   │ │ │
          ┌────────┘ │ └────────┐
          ▼          ▼          ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ A1:Codex│ │A2:Claude│ │R1:Review│
     │ (交互)  │ │ (交互)  │ │ (交互)  │  [已修正: 三者均为 Antigravity 交互式]
     └─────────┘ └─────────┘ └─────────┘
```

**通信方式**: 星形拓扑，所有 Agent 间通信必须经用户中转。
**共享状态**: Git 仓库 + `agent_inbox.md`

### 1.2 识别出的 5 个瓶颈

| # | 瓶颈 | 影响 | 严重性 |
|---|------|------|:---:|
| **B1** | **用户是唯一消息总线** | 每次 A→R 审核都需要用户手动复制/转发摘要，延迟高 | 🔴 |
| **B2** | **inbox.md 手写格式易出错** | 字段漏填、格式不一致（如 Fix1/Fix2 commit_sha 格式差异），R1 解析困难 | 🟡 |
| **B3** | **A1/A2 职能重叠无分工** | 两个实施 Agent 做同类工作，没有按任务类型分工（如 A1 做检测、A2 做训练），可能重复劳动或冲突 | 🟡 |
| **B4** | **无并行任务管理** | A1 做 E34b 的同时 A2 做 P2-A training，但没有"谁在做什么"的实时状态板 | 🟡 |
| **B5** | **审核是阻塞式的** | R1 必须等实施完成才能审核，不能边做边审 | 🟢 |

---

## 2. 业界 Agent Team 模式调研

### 2.1 主流模式对比

| 模式 | 代表框架 | 核心思想 | 适用场景 |
|------|---------|---------|---------|
| **Supervisor** | LangGraph, CrewAI | 中央 Agent 分配任务、汇总结果 | 需要统一决策的复杂流程 |
| **Pipeline** | AutoGen, CrewAI Sequential | 线性链式，A 的输出是 B 的输入 | 步骤固定的流水线 |
| **Fan-out/Gather** | LangGraph Parallel | 多 Agent 并行，汇总 Agent 合并 | 可并行的独立子任务 |
| **Shared State** | CrewAI Knowledge, LangGraph State | Agent 读写共享内存/文件 | 需要持续协作的长任务 |
| **Reflect & Critique** | Anthropic Patterns | 生成 → 审核 → 修改循环 | 高精度任务（我们的审核流） |
| **Debate** | CAMEL, ChatDev | Agent 间辩论得出最优解 | 需要探索多方案的设计决策 |

### 2.2 与我们场景的匹配度

我们的场景特点：
- **ML 实验迭代**（改 config → 训练 → 分析 → 下一轮）
- **两条并行工作线**（检测 + 训练）
- **需要审核门禁**（防止参数封板错误）
- **人工受限**（用户不能全职中转）

最匹配的模式组合：**Shared State + Reflect & Critique + Fan-out**

---

## 3. 优化方案

### 方案 A: 轻量改进（推荐，可立即实施）

**核心改动**：减少用户中转负担，明确 A1/A2 分工

```
                ┌─────────┐
                │  用户    │  ← 只做 Go/No-Go 决策
                └──┬───┬──┘
                   │   │
          ┌────────┘   └────────┐
          ▼                     ▼
     ┌─────────────┐      ┌─────────┐
     │ A1+A2       │      │R1:Review│
     │ 按任务线分工 │─────→│ 自助读   │
     │ 共享 inbox   │←─────│ inbox    │
     └─────────────┘      └─────────┘
```

#### 改动 1: A1/A2 按 **任务线** 分工

| Agent | 专责任务线 | 理由 |
|-------|----------|------|
| **A1 (Codex)** | 检测参数消融 (T1/T2/E34) | [已修正] 原因非"异步"，而是 A1 已有检测消融经验积累 (E34b/lock) |
| **A2 (Claude)** | 训练迭代 (P2-A/P2-B Fix) | A2 已有训练流程经验积累 (P1/P2-A Fix1-2) |

> 好处：消除"两个 agent 都能做训练"的模糊地带，减少 git 冲突。

#### 改动 2: inbox.md → 结构化 JSON

将 `agent_inbox.md` 的消息体改为 **JSON block**，保留 markdown 壳：

```markdown
## [2026-02-15 02:40] A2 → R1

```json
{
  "task": "P2-A Fix2 training",
  "commit_sha": "f865e64",
  "config_path": "src/config/phase2a_neighbor_overlap.yaml",
  "split": "val(71)",
  "output_path": ["logs/p2a_981146.log", "logs/p2a_a100_981147.log"],
  "key_metrics": {"best_pq": 0.3929, "best_dice": 0.6867},
  "regression": "12/12 gradient gate pass",
  "status": "pending_review"
}
```​
```

> 好处：R1 可以用工具解析 JSON，减少格式歧义。

#### 改动 3: 引入 **任务状态板** (`docs/agent_status.md`)

```markdown
# Agent Status Board

| ID | 当前任务 | 状态 | ETA | 阻塞? |
|----|---------|------|-----|-------|
| A1 | E34b SSOT 回填 | ✅ Done | — | — |
| A2 | P2-A Fix3 延迟启用 | 🔄 In Progress | 4h | 等 Fix2 审核 |
| R1 | Fix2 审核 | ⏳ Pending | — | 等用户转发 |
```

> 好处：用户一目了然，不用分别问每个 agent "你在干嘛"。

#### 改动 4: R1 **主动读 inbox** 而非被动等待

当前流程：A2完成 → 写inbox → 通知用户 → 用户转发给R1 → R1开始审核

优化流程：A2完成 → 写inbox（status=pending_review）→ **用户唤起 R1 时，R1 第一步自动扫 inbox** → 发现待审核项 → 开始审核

> [已修正] R1 是独立对话窗口，无法自启动。但用户唤起 R1 后，R1 默认先读 inbox 的 pending 项。

---

### 方案 B: 中度重构（下周可实施）

在方案 A 基础上增加：

#### 改动 5: 引入 **审核模板自动生成**

让实施 Agent 在完成任务时，自动生成一个审核报告的**骨架文件**：

```
docs/temp_reviews/fix2_review_DRAFT.md
```

内容：
- 自动填入 metrics、config diff、log 路径
- 留空：代码验证、策略合规、结论
- R1 只需填空 + 签名

> 好处：减少 R1 的 50% 重复工作量。

#### 改动 6: 引入 **回归门禁脚本** (`tools/regression_gate.py`)

实施 Agent 提交前自动运行，生成 pass/fail JSON：

```python
# 检查项:
# 1. 梯度门禁 (gradient gate)
# 2. Config 一致性 (YAML vs 脚本 vs 文档)
# 3. 数据划分正确性 (val/test 不混用)
# 4. Checkpoint 存在性验证
```

> 好处：R1 审核清单中 50% 的项目可自动化检查。

---

### 方案 C: 深度改造（需要工具链支持，长期）

| 改动 | 内容 | 依赖 |
|------|------|------|
| Agent 间直接通信 | 通过 MCP 或共享数据库让 Agent 互读 context | MCP 2.1 / 工具链支持 |
| 自动化 Orchestrator | 用户定义 DAG（如 LangGraph），Agent 自动按依赖执行 | 需要编排工具 |
| 实验版本管理 | 每个实验自动创建 Git branch，审核通过后 merge | Git workflow 改造 |

---

## 4. 推荐实施路径

```
本周:  方案 A (改动 1-4)  ← 零代码改动，只改文档 + 流程
下周:  方案 B (改动 5-6)  ← 写两个辅助脚本
长期:  方案 C             ← 等工具链成熟
```

## 5. 对 agent_management.md 的具体修改建议

将方案 A 的 4 项改动更新到 `agent_management.md`：
1. §1.1 Agent 清单增加"专责任务线"列
2. §3.2 产物提交格式改为 JSON block
3. 新增 §3.4 任务状态板 (`docs/agent_status.md`)
4. §2 决策升级规则增加"R1 主动检查 inbox"流程
