# CellSAM 文件整理方案（Codex Draft，供 Claude 审核）

> 文档目标：在**不删除文件**、不改变核心逻辑的前提下，先降低代码混用风险与入口混乱问题。
> 适用日期：2026-02-10
> 执行策略：分波次非破坏整理（rename + archive + DEPRECATED + 文档收口）

---

## 1. 背景与问题

当前仓库存在以下风险：

1. 根目录历史脚本过多，和 `src/`、`tools/` 主线入口并存，易误用。
2. `tools/` 下一次性实验脚本、历史评估脚本、可视化脚本混在同级目录，语义不清。
3. 推理入口存在 legacy 与 unified 并行，容易误走旧口径。
4. 文档中的“代码清单/入口说明”与现状存在时间差。

---

## 2. 整理原则（非删除）

1. **不删文件**：只移动、重命名、补充 `DEPRECATED` 说明。
2. **主入口收口**：明确唯一推荐入口，其他脚本标记为历史/兼容。
3. **路径可追溯**：归档脚本统一加 `deprecated_` 前缀。
4. **先降混用风险，再决定物理删除**：删除动作留到后续独立审批。

---

## 3. 主线入口定义（Single Source of Truth）

### 3.1 训练与评估主入口（保留在活跃目录）

1. 训练：`src/train.py`
2. Oracle 评估（GT 框）：`tools/standardized_inference.py`
3. E2E 评估（DAPI 框）：`tools/evaluate_e2e.py`
4. 多模型 Oracle 对比：`tools/comprehensive_eval.py`
5. 回归检查：`tools/test_unified_regression.py`

### 3.2 兼容入口（保留但显式 deprecated）

1. `tools/run_inference.py`（legacy pipeline）

### 3.3 辅助入口（非主流程）

1. `tools/compare_models.py`（辅助分析脚本，不纳入训练/标准评估/E2E 主流程）

---

## 3.4 Phase A0（执行前补齐覆盖）

目标：先补齐 `anti_test/`、`scripts/` 与同名冲突，再执行 Phase A/B。

### A0.1 anti_test 顶层脚本纳入归档（不删除）

当前 `anti_test/` 顶层 `.py`（13个）：

1. `analyze_and_test.py`
2. `debug_inference_range.py`
3. `eval_metrics.py`
4. `extract_docx.py`
5. `generate_report.py`
6. `test_cellfinder.py`
7. `test_dapi_actn2_detection.py`
8. `test_dapi_detection.py`
9. `test_dapi_improved.py`
10. `test_full_pipeline.py`
11. `test_traditional_detection.py`
12. `test_with_napari.py`
13. `visualize_test_results.py`

处理策略：

1. 新建 `anti_test/archive/deprecated_py/`
2. 以上 13 个文件迁移到该目录并重命名为 `deprecated_<原名>.py`
3. 批量补 `DEPRECATED` 头（标注替代入口：`src/train.py`、`tools/standardized_inference.py`、`tools/evaluate_e2e.py`、`tools/comprehensive_eval.py`、`tools/test_unified_regression.py`）

#### A0.1b anti_test 非 `.py` 产物处理（补充）

目录中存在 `.tif/.txt/.md/.docx` 等实验产物文件。为避免误删与丢失上下文，先采用保守策略：

1. 默认保留原位（不改名、不迁移）
2. 后续若要归档，仅迁移到 `anti_test/archive/artifacts/`，不改文件内容
3. 归档前在 `anti_test/README.md` 记录来源与用途

---

### A0.2 scripts 目录状态化（先标注，后迁移）

当前 `scripts/` 脚本（9个）：

1. `train_a100_pending.sh`
2. `train_ablation_l4.sh`
3. `train_ablation_v2.sh`
4. `train_bf_adapter.sh`
5. `train_bf_baseline_full.sh`
6. `train_instance_20260205.sh`
7. `train_instance_alice.sh`
8. `train_lr_ablation.sh`
9. `train_semantic.sh`

处理策略：

1. 新建 `scripts/README.md`
2. 对每个脚本标注状态：`active` 或 `legacy`
3. `legacy` 脚本迁移到 `scripts/archive/`（不删除）
4. `active` 脚本保留原路径，避免训练命令失效

建议初始状态（待项目负责人确认后执行迁移）：

1. `active` 候选：`train_instance_20260205.sh`、`train_ablation_v2.sh`、`train_lr_ablation.sh`
2. `review` 候选：`train_instance_alice.sh`、`train_a100_pending.sh`
3. `legacy` 候选：`train_semantic.sh`、`train_bf_adapter.sh`、`train_bf_baseline_full.sh`、`train_ablation_l4.sh`

---

### A0.3 同名冲突修正（必须执行）

问题：根目录 `compare_models.py` 与 `tools/compare_models.py` 同名并存，易误调用。

处理策略：

1. 将根目录 `compare_models.py` 迁移到 `archive/root_scripts/deprecated_compare_models_root.py`
2. 保留 `tools/compare_models.py` 作为辅助分析入口（非主流程）
3. 可选：在根目录保留 5-10 行 stub，打印 deprecated 提示并指向 `tools/compare_models.py`

---

### A0.4 验证门槛（新增硬门槛）

1. 根目录活跃 `.py` 数量 = 0（仅保留项目元脚本可例外）
2. `anti_test/` 顶层活跃 `.py` 数量 = 0
3. `scripts/README.md` 覆盖 9/9 脚本状态
4. `python tools/test_unified_regression.py` 通过

---
## 4. 执行方案

执行顺序（必须）：`A0 -> A -> B`

## Phase A（Wave 1，低风险）

目标：清理根目录与明显过时脚本，先完成“入口去歧义”。

### A1. 新建归档目录

1. `archive/root_scripts/`
2. `tools/archive/tests_deprecated/`
3. `tools/archive/legacy_eval/`
4. `tools/archive/legacy_experiment/`（预留）

### A2. 根目录历史脚本迁移（不删除）

迁移并重命名为 `deprecated_*.py`：

1. `debug_class_imbalance.py`
2. `debug_trained_model.py`
3. `debug_validation.py`
4. `evaluate_test_set.py`
5. `finetune_boundary.py`
6. `finetune_boundary_simple.py`
7. `run_cellsam.py`
8. `test_loss_fn.py`
9. `test_model.py`
10. `train_expanded.py`
11. `verify_cell_matching.py`
12. `verify_env.py`

说明：根目录 `compare_models.py` 已在 **A0.3** 单独处理，A2 不重复定义。

迁移目标：`archive/root_scripts/deprecated_<原名>.py`

### A3. tools 过时脚本迁移（不删除）

迁移并重命名：

1. `tools/test_bestmatch_validation.py` -> `tools/archive/tests_deprecated/deprecated_test_bestmatch_validation.py`
2. `tools/test_unified_inference.py` -> `tools/archive/tests_deprecated/deprecated_test_unified_inference.py`
3. `tools/eval_e24_e28.py` -> `tools/archive/legacy_eval/deprecated_eval_e24_e28.py`
4. `tools/debug_eval.py` -> `tools/archive/legacy_eval/deprecated_debug_eval.py`

### A4. 补 `DEPRECATED` 文件头

对以下文件统一增加说明头：

1. `archive/root_scripts/*.py`
2. `tools/archive/**/*.py`
3. `tools/run_inference.py`

头部需说明：

1. 归档原因
2. 推荐替代入口
3. 该脚本是否 legacy 口径

### A5. 入口收口文档

新增：

1. `docs/ENTRYPOINTS.md`：主入口与禁用入口总览
2. `docs/ARCHIVE_PLAN.md`：归档目录规则与命名规则

---

## Phase B（Wave 2，低风险）

目标：归档“日期型/一次性实验脚本”，进一步减少同级噪音。

建议迁移（不删除）：

1. `tools/baseline_gt_cellsam_20260206.py` -> `tools/archive/legacy_experiment/deprecated_baseline_gt_cellsam_20260206.py`
2. `tools/visualize_segmentation_20260206.py` -> `tools/archive/legacy_visualization/deprecated_visualize_segmentation_20260206.py`
3. `tools/visualize_baseline_results.py` -> `tools/archive/legacy_visualization/deprecated_visualize_baseline_results.py`
4. `tools/visualize_e29_results.py` -> `tools/archive/legacy_visualization/deprecated_visualize_e29_results.py`
5. `tools/test_baseline_inference.py` -> `tools/archive/legacy_experiment/deprecated_test_baseline_inference.py`
6. `tools/test_e29_inference.py` -> `tools/archive/legacy_experiment/deprecated_test_e29_inference.py`
7. `tools/test_e29_fixed_inference.py` -> `tools/archive/legacy_experiment/deprecated_test_e29_fixed_inference.py`
8. `tools/test_e29_dapi_inference.py` -> `tools/archive/legacy_experiment/deprecated_test_e29_dapi_inference.py`
9. `tools/compare_models_v2.py` -> `tools/archive/legacy_compare/deprecated_compare_models_v2.py`

并新增：

1. `docs/TOOLS_ACTIVE.md`：tools 层仅保留的活跃入口

---

## 5. 命名与结构规范（后续持续执行）

### 5.1 命名前缀

1. 活跃入口：保持语义名（如 `standardized_inference.py`）
2. 归档脚本：统一 `deprecated_` 前缀
3. 一次性脚本：完成后迁入 `tools/archive/...`

### 5.2 禁止继续增加的混乱模式

1. 不再在根目录新增 `.py` 训练/评估脚本。
2. 不再新增与现有主入口功能重复的新入口脚本。
3. 不再让 legacy 脚本与主入口同级且无说明并存。

---

## 6. 验证与验收

### 6.1 必跑验证

1. `python tools/test_unified_regression.py`（应通过）
2. `rg -n "DEPRECATED" tools/run_inference.py archive/root_scripts tools/archive`
3. 人工检查 `docs/ENTRYPOINTS.md` 与实际入口是否一致

### 6.2 验收标准

1. 用户从文档可在 30 秒内定位主入口。
2. 根目录无活跃训练/评估脚本残留。
3. 归档脚本具备可读的替代路径说明。
4. 主流程训练和评估命令不受影响。

---

## 7. 回滚策略

若任一步骤引发问题，可全量回滚：

1. `git status`
2. `git restore .`

或按文件回滚：

1. `git restore <file_path>`

---

## 8. Claude 审核清单（请重点核对）

1. Phase A/B 迁移名单是否有误迁风险。
2. 是否仍有遗漏的 legacy 入口未标注。
3. `docs/ENTRYPOINTS.md` 推荐入口是否与当前统一口径一致。
4. 归档目录命名是否满足团队后续维护习惯。
5. 是否需要在 CI 中加入“禁止根目录新增脚本”的检查。

---

## 9. 后续建议（不在本次执行）

1. 第三波（中风险）：将 `view_* / visualize_* / analyze_*` 分域归类到 `tools/archive` 或 `tools/analysis`。
2. 更新 `docs/code_inventory.md`，替换过期字段并纳入 Phase 0 之后的统一入口。
3. 引入轻量 CI 规则：检查 deprecated 脚本不可被主文档推荐为入口。

---

## 10. Claude 审核反馈模板（可复制）

> 使用方式：Claude 审核时直接复制本节，逐项填写。

### 10.1 审核结论

- [ ] 通过
- [ ] 有条件通过（需完成 10.3 修正项）
- [ ] 不通过

结论说明：

- 审核日期：
- 审核人：
- 总体判断（1-3 句）：

### 10.2 逐项核对（对应本方案）

1. 主入口收口是否准确（`src/train.py`、`tools/standardized_inference.py`、`tools/evaluate_e2e.py`、`tools/comprehensive_eval.py`、`tools/test_unified_regression.py`）。
2. 兼容入口是否明确标注 deprecated（尤其 `tools/run_inference.py`）。
3. Phase A 迁移名单是否存在误迁风险（会影响当前训练/评估流程）。
4. Phase B 迁移名单是否合理（日期型/一次性脚本归档边界是否清晰）。
5. DEPRECATED 头模板是否清晰且可追溯到替代入口。
6. 归档目录规划是否便于后续维护与检索。
7. 验证与回滚步骤是否可执行。

### 10.3 必须修正项（如有）

1. [优先级 High/Medium/Low] 问题描述：
   - 位置：
   - 风险：
   - 建议修复：
2. [优先级 High/Medium/Low] 问题描述：
   - 位置：
   - 风险：
   - 建议修复：
3. [优先级 High/Medium/Low] 问题描述：
   - 位置：
   - 风险：
   - 建议修复：

### 10.4 建议优化项（非阻塞）

1. 
2. 
3. 

### 10.5 审核后决策

- [ ] 直接执行 Phase A
- [ ] 先修正 10.3，再执行 Phase A
- [ ] 暂缓执行，等待进一步方案


