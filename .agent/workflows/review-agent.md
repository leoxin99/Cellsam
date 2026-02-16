---
description: 审核 Agent 工作流 — 第三方审核实施 Agent 产物
---

# 审核 Agent (R1) 工作流

> SSOT: `docs/agent_management.md` 定义完整协作规范。本文件仅描述审核 Agent 的执行步骤。

## 审核流程

### 1. 接收产物摘要
用户将实施 Agent 的产物摘要发送过来。摘要**必含**以下字段:

| 字段 | 说明 | 示例 |
|------|------|------|
| `commit_sha` | 最近一次相关 commit | `a3f2c1d` |
| `cmd` | 执行的命令 | `python tools/ablation_detection_e34b.py` |
| `config_path` | 使用的配置文件 | `src/config/phase2a_neighbor_overlap.yaml` |
| `split` | 数据划分 | `val(71)` / `test(73)` |
| `output_path` | 产物文件路径 | `experiments/ablation_detection_e34b/results.json` |
| `key_metrics` | 关键数值 | `F1=0.8106, P=0.7639, R=0.8633` |
| `regression` | 回归测试结果 | `10 passed, 0 failed` |
| `modified_files` | 修改的文件列表 | `src/detection/dapi.py:71,231,539` |

> 若摘要缺少以上必填字段，审核 Agent 应要求用户补充后再开始审核。

### 2. 独立验证
// turbo
- 读取实际产物文件 (JSON / 代码 / 文档)
- 验证数值与用户摘要一致
- 检查代码逻辑正确性
- 检查搜索空间完整性
- 检查实验策略合规性 (如 val/test 分离)

### 3. 输出审核报告
- 写入 `docs/temp_reviews/<review_name>.md`
- 使用仓库相对路径引用文件 (如 `src/detection/dapi.py:71`，不用 `file:///d:/...`)
- 包含: 审核结论、代码验证、实验验证、关键发现、建议

### 4. 文档回填 (审核通过后, A 模式)

**前置检查**:
- 确认实施 Agent 无未 commit 的文档修改（向用户确认或检查 `git status`）
- 声明将要修改的文件清单

**回填范围**:
- 更新 `claude.md` 状态仪表板
- 更新 `docs/task_backlog.md` 任务勾选
- 更新 `docs/experiments_log.md` 新增实验记录
- 更新 `docs/dapi_detection_design.md` 等 SSOT 文档

**回填后**:
- 通知用户 → 用户可转告实施 Agent `git pull`

### 5. 审核不通过时
- 列出问题清单 (编号 + 严重程度: High/Medium/Low)
- 用户转交给实施 Agent 修复后重新提交

## 审核清单模板

```markdown
- [ ] 代码修改是否正确打通参数链路
- [ ] 搜索空间是否覆盖 task_backlog 定义
- [ ] 实验结果 JSON 数值与摘要一致
- [ ] val/test 分离策略是否合规
- [ ] 回归测试是否通过
- [ ] 最优参数是否有显著 runner-up 差异
- [ ] val→test 泛化性是否可接受 (Δ<2pp)
```
