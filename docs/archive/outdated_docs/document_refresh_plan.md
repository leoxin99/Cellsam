# 文档整理与更新方案（2026-02-13）

> 目标：修复编码风险，收敛“单一事实来源（SSOT）”，降低训练/评估阶段误用旧文档的概率。  
> 范围：`CLAUDE.md` 中当前引用的 8 份核心文档。

---

## 1. 审核结论（按优先级）

| 文档 | 现状 | 风险级别 | 处理策略 |
|------|------|----------|----------|
| `docs/experiments_log.md` | 历史上存在 UTF-8/GBK 混合编码，已修复为 UTF-8 | High | 继续保留为“实验流水账主文档”，补齐 Phase2 记录 |
| `docs/inference_standard.md` | 仍含旧推理描述（如旧 API 路径） | High | 以 `src/inference/core.py` + `tools/*eval*.py` 重写“唯一推理口径” |
| `docs/code_inventory.md` | 最后更新 2026-02-05，未覆盖近期统一推理/Phase2 代码 | High | 重建为“Active / Deprecated / Archive”三段清单 |
| `docs/boundary_enhancement_design.md` | 仍是“待审批 + scipy 伪代码” | Medium | 标记为 Historical，并补“已落地实现”链接 |
| `docs/claude_pipeline_analysis.md` | 状态仍为“待集成”，与现主线不一致 | Medium | 改为历史分析文档，避免被当作执行规范 |
| `docs/dataset_parameters.md` | 顶部更新时间过旧（2026-01-25） | Medium | 更新“最后校验时间”和当前 split/分辨率说明 |
| `docs/naming_convention.md` | 阶段命名停留在 E29-E32 语境 | Medium | 增加 Phase1/Phase2a 新命名规则 |
| `docs/error_log_and_checklist.md` | 主体可用，但缺少近期错误条目 | Low | 追加 Phase2 新错误与对应预检项 |

---

## 2. 分阶段执行

### Phase A（当天完成）
1. 编码与可读性基线：
   - 所有核心文档统一 UTF-8。
   - `docs/experiments_log.md` 编码修复后，人工检查章节标题与表格对齐。
2. 在每份核心文档头部增加统一元信息：
   - `最后更新`
   - `状态: Active / Historical / Deprecated`
   - `事实来源: 代码路径`

### Phase B（1 天）
1. 重写 `docs/inference_standard.md`：
   - 以 `src/inference/core.py` 为准写明冲突裁决策略。
   - 明确 Oracle / E2E / val / test 的工具分工。
2. 重写 `docs/code_inventory.md`：
   - 只保留当前活跃入口 + 归档入口，不再按历史实验堆叠。
3. 更新 `docs/dataset_parameters.md` 与 `docs/naming_convention.md`：
   - 统一当前 split、实验命名、Phase2a 配置命名。

### Phase C（半天）
1. 将历史设计文档降级：
   - `docs/boundary_enhancement_design.md`
   - `docs/claude_pipeline_analysis.md`
2. 在 `CLAUDE.md` 文档链接区补充“状态列”，默认只引导 Active 文档。

---

## 3. 验收标准（必须全部满足）

1. `CLAUDE.md` 中每个链接文档都有明确状态（Active/Historical）。
2. `docs/inference_standard.md` 与 `src/inference/core.py` 参数命名一致。
3. `docs/code_inventory.md` 可一眼定位训练、验证、Oracle、E2E、回归入口。
4. `docs/experiments_log.md` 可被 UTF-8 正常读取，无乱码符号。
5. 新开对话时，仅依赖 `CLAUDE.md + inference_standard.md + code_inventory.md` 可定位下一步工作。

---

## 4. 建议执行顺序

1. `inference_standard.md`
2. `code_inventory.md`
3. `dataset_parameters.md`
4. `naming_convention.md`
5. `error_log_and_checklist.md`
6. `boundary_enhancement_design.md`
7. `claude_pipeline_analysis.md`
8. 最后回填 `CLAUDE.md` 的文档状态表

