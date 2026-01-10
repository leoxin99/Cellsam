---
description: 每日项目完成后提交到 GitHub
---

# 每日 GitHub 同步 (Daily GitHub Sync)

每日工作结束时，按以下步骤将项目同步到 GitHub，确保版本控制规范。

## 1. 检查当前状态

// turbo
```bash
cd d:/AI/paper/CellSam
git status
```

查看：
- 修改的文件 (M)
- 新增的文件 (??)
- 删除的文件 (D)

## 2. 选择性暂存文件

### 2.1 暂存文档更新
// turbo
```bash
git add CLAUDE.md
git add anti_test/*.md
git add .agent/workflows/*.md
```

### 2.2 暂存代码更新
// turbo
```bash
git add *.py
git add src/*.py
git add anti_test/*.py
```

### 2.3 暂存配置文件
// turbo
```bash
git add *.json
git add *.csv
```

**注意**: 不要暂存以下文件：
- `checkpoints/` (模型太大)
- `experiments/` (实验数据)
- `data/` (原始数据)
- `*.pt` (模型权重)

## 3. 编写提交信息

使用 Conventional Commits 格式：

```
<type>(<scope>): <description>

[optional body]
```

### 类型 (type)
| Type | 说明 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| docs | 仅文档变更 |
| refactor | 重构代码 |
| perf | 性能优化 |
| test | 测试相关 |
| chore | 维护任务 |

### 示例
```bash
git commit -m "feat(training): add boundary loss for edge precision

- Implemented BoundaryLoss class
- Added finetune_boundary_simple.py
- PQ@0.5 improved from 0.02 to 0.09"
```

### 每日提交模板
```bash
git commit -m "docs: daily update [YYYY-MM-DD]

Experiments completed:
- E12: Boundary loss fine-tuning

Current metrics:
- PQ@0.5: 0.087
- Dice: 0.822"
```

## 4. 推送到 GitHub

// turbo
```bash
git push origin main
```

如果是第一次推送新分支：
```bash
git push -u origin main
```

## 5. 版本标签（里程碑时）

当达到重要里程碑时，创建标签：

```bash
# 创建标签
git tag -a v0.1.0 -m "First working pipeline with boundary loss"

# 推送标签
git push origin v0.1.0
```

### 版本号规范
- `v0.x.x` - 开发阶段
- `v1.0.0` - 首个稳定版本
- 次版本号 - 新功能
- 补丁号 - bug 修复

## 6. 完整每日流程（一键脚本）

// turbo
```bash
cd d:/AI/paper/CellSam

# 暂存所有文档和代码（不包括大文件）
git add CLAUDE.md anti_test/*.md .agent/ *.py src/ docs/ --ignore-errors

# 提交
git commit -m "docs: daily update $(Get-Date -Format 'yyyy-MM-dd')"

# 推送
git push origin main
```

## 7. 常见问题处理

### 7.1 推送被拒绝（远程有更新）
```bash
git pull --rebase origin main
git push origin main
```

### 7.2 撤销最近的提交
```bash
git reset --soft HEAD~1
```

### 7.3 查看提交历史
```bash
git log --oneline -10
```
