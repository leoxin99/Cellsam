---
name: config-diff-audit
description: 自动检查训练 config/脚本一致性，防止配置漂移
---

# Config Diff Audit Skill

训练前或审核时，自动检查 config YAML、SLURM 脚本、checkpoint 路径之间的一致性。

## 触发条件

- 审核训练实验产物时自动激活
- 用户要求检查 config 一致性时手动激活

## 审计清单

### 1. Config YAML 内部一致性

检查 `src/config/<name>.yaml`:

| 检查项 | 方法 | 常见问题 |
|--------|------|---------|
| `experiment_name` 是否与实际用途匹配 | 读 YAML → 检查命名 | 复制旧 config 忘改名 |
| `checkpoint_path` 是否指向正确的 Phase | 读 YAML → 验证路径存在 | 指向已删 checkpoint |
| `data.splits_dir` 是否与 split 策略匹配 | 读 YAML → 对比 `docs/dataset_parameters.md` | 用了错的 split |
| Loss 权重是否与设计文档一致 | 读 YAML → 对比 `docs/phase2_design.md` | 权重与设计意图不符 |
| `use_*` 开关是否与实验目的一致 | 读 YAML → 检查 bool 字段 | 忘了开/关某个 loss |

### 2. SLURM 脚本一致性

检查 `scripts/train_*.sh`:

| 检查项 | 方法 |
|--------|------|
| `--config` 参数指向的 YAML 是否存在 | grep config 路径 → 验证文件存在 |
| Job name 是否与 experiment_name 匹配 | 对比 `#SBATCH --job-name` 与 YAML `experiment_name` |
| GPU 分区是否与预期一致 | 检查 `#SBATCH --partition` |
| checkpoint glob 是否能匹配到文件 | 检查 `ls` 模式 |

### 3. 跨文件一致性

| 对比 | 文件 A | 文件 B | 检查 |
|------|--------|--------|------|
| 权重 | config YAML | `docs/phase2_design.md` | 数值一致 |
| 实验名 | config YAML | SLURM 脚本 comment | 名称匹配 |
| Checkpoint | config YAML | Alice 实际目录 | 路径存在 |
| Split 数量 | config YAML | `docs/dataset_parameters.md` | 数量一致 |

### 4. 历史 Config 变更追溯

对比当前 config 与上一个版本:

```markdown
## Config Diff: phase2a_neighbor_overlap.yaml

| 字段 | 旧值 (Fix1) | 新值 (Fix2) | 变化原因 |
|------|------------|------------|---------|
| loss.neighbor_weight | 0.3 | 0.1 | Fix1 PQ 退化，降权 |
| loss.overlap_weight | 0.1 | 0.05 | 同上 |
| experiment_name | fix1 | fix2_low_weight | 区分实验 |
```

## 输出格式

```markdown
## Config Audit Report

### 通过项 ✅
- [x] experiment_name 匹配
- [x] checkpoint 路径存在
- [x] split 与 dataset_parameters 一致

### 警告项 ⚠️
- [ ] neighbor_weight=0.1 低于 phase2_design.md 推荐的 0.3

### 阻塞项 ❌
- [ ] checkpoint_path 指向不存在的路径

### Config Diff 表
(如上格式)
```

## 常见配置陷阱

| 陷阱 | 表现 | 根因 |
|------|------|------|
| 训练用了旧 checkpoint | PQ 远低于预期 | config 里 checkpoint 没更新 |
| Loss 权重不生效 | 指标无变化 | `use_neighbor=False` 但设了权重 |
| 脚本与 config 错配 | 实验名字混乱 | 复制脚本忘改 `--config` |
| Split 泄露 | test 指标异常高 | config 指错 splits_dir |
