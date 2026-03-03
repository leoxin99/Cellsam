# AI Agent 项目管理模板

> **用途**: 将本项目的 Agent 管理模式复制到新项目中
> **来源**: CellSAM 项目实践总结

---

## 一、项目目录结构模板

```
project/
├── .agent/
│   └── workflows/               # Agent 工作流指令
│       ├── project-onboarding.md # 入职指南 (/project-onboarding)
│       ├── commands.md           # 常用命令速查 (/commands)
│       ├── daily-github-sync.md  # 每日 Git 同步 (/daily-github-sync)
│       └── review-agent.md       # 审核流程 (/review-agent)
├── .gitignore                    # 顶部加白名单 !src/ !tools/ !scripts/
├── CLAUDE.md                     # 项目蓝图 (Agent 首读文件)
│
├── src/                          # 源代码
│   ├── config/                   # YAML 实验配置
│   ├── train.py                  # 训练入口
│   └── ...
├── tools/                        # 评估/分析/可视化脚本
├── scripts/                      # SLURM / bash 脚本
├── data/                         # 数据目录 (gitignore)
├── checkpoints/                  # 模型权重 (gitignore)
├── experiments/                  # 实验结果 JSON (gitignore binary)
├── logs/                         # 训练日志 (gitignore)
│
└── docs/                         # 文档中心
    ├── experiments_log.md         # 实验流水账 (SSOT)
    ├── task_backlog.md            # 任务清单 (按优先级)
    ├── paper_preparation.md      # 论文素材
    ├── alice_quick_reference.md  # HPC 集群指南 + 踩坑记录
    ├── agent_inbox.md            # Agent 间通信信箱
    ├── agent_management.md       # 协作规则
    ├── agent_status.md           # 实时状态面板
    ├── meeting_notes_*.md        # 导师会议纪要
    ├── report_*.md               # 进展报告 (按日期)
    └── inbox/                    # 审核产物暂存区
```

---

## 二、核心文档模板

### 2.1 CLAUDE.md (项目蓝图)

每个项目的根目录放一个 `CLAUDE.md`，Agent 首次加入时必读：

```markdown
# 项目名称

## 状态仪表板
| 指标 | 当前值 | 目标 |
|------|--------|------|
| 主指标 | xxx | xxx |

## 关键路径
当前阶段 → 下一步 → 最终目标

## 核心文档
| 文档 | 状态 | 说明 |
|------|------|------|

## AI 工作规范
1. 禁止估算 — 所有数据必须实测
2. 审查制度 — 重大变更需 R1 审核
3. 文档优先 — 先更新文档再写代码
```

### 2.2 docs/experiments_log.md (实验记录)

```markdown
# 实验记录

| ID | 日期 | 实验名称 | 结果 | 状态 |
|----|------|---------|------|------|
| E01 | 日期 | 描述 | 指标 | ✅/❌ |
```

### 2.3 docs/task_backlog.md (任务清单)

```markdown
# Task Backlog

## P0 — 紧急
- [ ] 任务1: 完成标准 + 产物 + 执行者

## P1 — 重要
- [ ] 任务2

## P2 — 有时间就做
- [ ] 任务3

## ✅ 已完成
- [x] 任务4
```

---

## 三、Workflow 文件模板

### 3.1 project-onboarding.md

```yaml
---
description: AI 助手加入项目时的入门指南
---
```

内容顺序：
1. 读 CLAUDE.md (项目蓝图)
2. 读移交文档 (如有)
3. 读 agent_management.md (协作规则)
4. 读 agent_inbox.md (最新通信)
5. 读 task_backlog.md (当前任务)
6. 按角色读 SSOT 文档
7. 开始工作

### 3.2 daily-github-sync.md

```yaml
---
description: 每日项目完成后提交到 GitHub
---
```

步骤:
1. `git status` → 检查变更
2. `git add src/ tools/ scripts/ docs/` → 暂存代码+文档
3. `git commit -m "type(scope): description"` → 提交
4. `git push origin main` → 推送
5. 如有 HPC: `ssh server "cd ~/project && git pull"` → 远程同步

### 3.3 commands.md (常用命令)

```yaml
---
description: 项目常用命令 — 自动执行
---
```

分为两类:
- `// turbo` 安全命令 (只读: 查看状态/日志)
- 需审批命令 (训练/评估/代码修改)

### 3.4 review-agent.md (审核流程)

```yaml
---
description: 审核 Agent 工作流 — 第三方审核实施 Agent 产物
---
```

---

## 四、HPC 集群管理模式

### 4.1 代码部署流程

```
本地开发 → git push → HPC git pull → sbatch 提交
```

### 4.2 SLURM 脚本模板

```bash
#!/bin/bash -l
#SBATCH --job-name=实验名_分区
#SBATCH --partition=gpu-xxx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/实验名_%j.log
#SBATCH --error=logs/实验名_%j.err

set -o pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate env_name
set -u  # 必须在 conda activate 之后

export PYTHONPATH="${HOME}/project/src:${HOME}/project:${PYTHONPATH:-}"
cd ~/project
mkdir -p logs checkpoints

# 训练
python src/train.py --config src/config/xxx.yaml --seed $SEED
```

### 4.3 踩坑备忘

在 `docs/alice_quick_reference.md` 中记录每次踩坑:

| 日期 | 问题 | 影响 | 修复 |
|------|------|------|------|

**必须遵守的规则**:
1. `set -u` 放在 `conda activate` 之后
2. PYTHONPATH 包含所有 import 路径
3. 新文件 `git add` 后确认 HPC 上 `ls` 存在
4. Checkpoint 目录名包含 seed (防并发冲突)
5. 用 before/after snapshot 而非 `ls -td` 找最新 checkpoint

---

## 五、实验管理模式

### 5.1 配置驱动

每个实验一个 YAML 文件:
```
src/config/
├── t27a_planb_decoder.yaml    # 实验 T27a
├── t28_planb_3ch.yaml         # 实验 T28
└── ...
```

消融实验只改一个变量，其余完全相同。

### 5.2 多 Seed 复现

- 每个配置跑 2 个 seed (42, 123)
- Checkpoint 目录名包含 seed: `{exp_name}_seed{seed}_{timestamp}`
- 结果取 mean ± std

### 5.3 结果追踪

每个实验在 `experiments_log.md` 新增一行:
```
| T27a | 日期 | 描述 | PQ=0.638 | ✅ |
```

---

## 六、进展汇报模式

用日期命名: `docs/report_3.2.md`

结构:
1. 导师会议任务完成情况 (清单对照)
2. 新增实验总结 (含指标表)
3. PQ 进展总览 (可视化进度条)
4. 剩余实验计划
5. 论文写作状态

---

## 七、.gitignore 模板

```gitignore
# 白名单 — 项目源码目录必须跟踪
!src/
!src/**
!tools/
!tools/**
!scripts/
!scripts/**
!docs/
!docs/**

# 大数据
data/
checkpoints/
*.pt
*.pth

# Python
__pycache__/
*.pyc

# IDE
.vscode/
.idea/

# 日志
*.log
logs/

# 临时文件
*.tmp
```

---

## 八、快速启动清单

在新项目中按以下顺序设置:

- [ ] 1. 创建 `.agent/workflows/` 目录 + 4 个 workflow 文件
- [ ] 2. 创建 `CLAUDE.md` 项目蓝图
- [ ] 3. 创建 `docs/` 文档目录 + 核心文档
- [ ] 4. 配置 `.gitignore` (白名单 + 排除大文件)
- [ ] 5. 创建 `src/config/` 配置目录
- [ ] 6. 如有 HPC: 创建 `docs/alice_quick_reference.md`
- [ ] 7. 首次 `git push` 验证所有文件上传
- [ ] 8. 用 `/project-onboarding` 测试 Agent 入职流程
