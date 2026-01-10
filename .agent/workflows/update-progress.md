---
description: 更新项目进展状态到各文档
---

# 更新项目进展 (Update Project Progress)

当项目有重大进展时，按以下步骤同步更新所有相关文档：

## 1. 更新 CLAUDE.md

主项目蓝图文件位于 `CLAUDE.md`：

### 1.1 更新 Dashboard

找到 "Project Status Dashboard" 部分，更新相关指标：

```markdown
| Metric | Value | Status |
|--------|-------|--------|
| Detection F1 | 0.750 | ✅ |
| Instance Dice | 0.82 | ✅ |
| PQ@0.5 | 0.087 | 🔄 改进中 |
```

### 1.2 更新 Changelog

在 Changelog 部分添加新条目：

```markdown
### [YYYY-MM-DD]
- [变更描述]
```

## 2. 更新 results_summary.md

文件位于 `anti_test/results_summary.md`：

- 更新 "关键数值摘要" 部分
- 如有新的对比数据，添加新表格
- 更新 "指标实现状态" 表

## 3. 更新 methods_draft.md

文件位于 `anti_test/methods_draft.md`：

- 如有新方法变更（如边界损失），添加到相应章节
- 更新评估指标说明

## 4. 检查文档一致性

确保以下信息在各文档中保持一致：

| 信息 | 出现位置 |
|------|---------|
| 最佳模型路径 | CLAUDE.md, experiments_log.md |
| 核心指标数值 | CLAUDE.md, results_summary.md |
| 方法描述 | methods_draft.md, experiments_log.md |

## 5. 更新 Git Commit（可选）

// turbo
```bash
cd d:/AI/paper/CellSam
git add anti_test/*.md CLAUDE.md
git commit -m "docs: update progress [YYYY-MM-DD]"
```
