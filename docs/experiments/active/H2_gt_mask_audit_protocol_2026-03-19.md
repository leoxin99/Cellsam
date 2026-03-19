# H2 GT Mask Audit Protocol (2026-03-19)

## 1) 目标与边界

目标：
- 处理 `GT mask` 已知误差（漏标、误标、边界不一致）对评估结论的影响。

边界：
- 不允许“用模型预测直接改 GT 后回填主指标”。
- H2 作为独立数据质量线，不与 H1b detector 改造混在同一因果结论中。

---

## 2) 已确认的评估口径

1. 主表固定 `GT-v1`（当前锁定标签）。  
2. 增设 `GT-v2`（人工复核修订版）作为 sensitivity 附表。  
3. 论文中同时报告：
   - 主结论：`GT-v1`
   - 稳健性：`GT-v2` 相对 `GT-v1` 的指标变化

该原则已在 inbox 记录：
- [agent_inbox.md#L210](d:/AI/paper/CellSam/docs/agent_inbox.md#L210)

---

## 3) GT-v2 生成流程

### 3.1 疑似样本清单（machine-assisted, human-reviewed）

输入线索：
- `Actn2` 强信号 + `DAPI` 核存在 + 细胞轮廓连续，但 `GT` 缺失（疑似漏标）。
- `GT` 标注实例缺乏可信细胞证据（疑似误标）。
- H1/H1b 输出中高置信候选与 GT 冲突区域。

产物：
- `suspect_missing_cm.csv`
- `suspect_non_cm_gt.csv`

### 3.2 双人复核

- Reviewer-A 与 Reviewer-B 独立判定。
- 冲突样本进入仲裁（Reviewer-C 或共同复核）。
- 仅保留最终一致判定进入 GT-v2。

### 3.3 变更日志（必须全量可追溯）

每条变更记录最小字段：
- `sample_id`
- `instance_id`（或 new_instance_id）
- `action`（add/remove/edit）
- `reason`
- `evidence`（DAPI/Actn2/形态依据）
- `reviewer_a`
- `reviewer_b`
- `arbiter`（如有）
- `timestamp`

---

## 4) 论文呈现建议

主文：
- 所有核心结论使用 `GT-v1` 主表。

附录/补充：
- 报告 `GT-v2` 指标与 `GT-v1` 的差值（`ΔF1`, `ΔPQ`, `ΔRecall`, `ΔPrecision`）。
- 明确声明 `GT-v2` 是 sensitivity analysis，不替代主表。

---

## 5) 当前状态

- H2 方案已确认并写入 inbox（见上）。
- 目前仍缺少一份“已执行完成”的 GT-v2 变更清单与最终评估报告。
- 下一步应先落地：疑似清单生成脚本 + 双人复核记录模板 + 变更日志模板。
