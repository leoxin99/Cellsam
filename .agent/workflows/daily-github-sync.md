---
description: 每日项目完成后提交到 GitHub
---

# 每日 GitHub 同步 (Daily GitHub Sync)

每日工作结束时，按以下步骤将项目同步到 GitHub。

## 1. 检查当前状态

// turbo
```powershell
Set-Location d:\AI\paper\CellSam
git status
```

查看：
- 修改的文件 (M)
- 新增的文件 (??)
- 删除的文件 (D)

## 2. 选择性暂存文件

### 2.1 暂存文档更新
// turbo
```powershell
git add CLAUDE.md
git add docs/*.md
git add .agent/workflows/*.md
```

### 2.2 暂存代码更新
// turbo
```powershell
git add src/ tools/ scripts/
```

### 2.3 暂存实验结果 (仅 JSON/小文件)
// turbo
```powershell
git add experiments/**/*.json
```

**注意**: 不要暂存以下文件：
- `checkpoints/` (模型太大)
- `data/` (原始数据)
- `logs/` (训练日志，通常从 Alice 同步)
- `*.pt` / `*.pth` (模型权重)

## 3. 编写提交信息

使用 Conventional Commits 格式：

```
<type>(<scope>): <description>
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
```powershell
git commit -m "feat(detection): add E34b joint ablation for edge/merge params"
```

## 4. 推送到 GitHub

// turbo
```powershell
git push origin main
```

如果推送被拒绝（远程有更新）：
```powershell
git pull --rebase origin main
git push origin main
```

## 5. 版本标签（里程碑时）

当达到重要里程碑时，创建标签：

```powershell
git tag -a v0.2.0 -m "Phase 1 complete: Oracle PQ=0.464, Detection F1=0.803"
git push origin v0.2.0
```

## 6. 完整每日流程（一键脚本）

// turbo
```powershell
Set-Location d:\AI\paper\CellSam
git add CLAUDE.md docs/ .agent/ src/ tools/ scripts/ experiments/**/*.json --ignore-errors 2>$null
git commit -m "docs: daily update $(Get-Date -Format 'yyyy-MM-dd')"
git push origin main
```

## 7. 常见问题

### 撤销最近的提交
```powershell
git reset --soft HEAD~1
```

### 查看提交历史
```powershell
git log --oneline -10
```
