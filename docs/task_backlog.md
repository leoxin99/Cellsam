# Task Backlog (Active)

> 状态: 🟢 Active  
> 维护原则: 只记录“可执行任务”，每项必须有口径/产物/完成标准。  
> 更新时间: 2026-02-14

---

## 1. Short-Term (本周优先)

### T1. E34 Test 封板评估 (DAPI/Adaptive) ✅ Completed (2026-02-14)
- 优先级: P0
- 目标: 固定 val(71) 锁定候选参数，在 test(73) 上仅执行一次最终评估，不反向调参。
- 输入:
  - DAPI 候选: `min=1500, max=20000, relative_1.2x`
  - Adaptive 候选: `radius=200, min_zlines=5, zline_threshold=0.01`
- 产物:
  - `experiments/ablation_detection_lock/results.json`
  - 文档回填: `docs/dapi_detection_design.md`, `docs/experiments_log.md`, `CLAUDE.md`
- 完成标准:
  - [x] test(73) 结果写盘
  - [x] 文档标记 “参数已封板”
- 结果:
  - DAPI: P=0.7462, R=0.8699, F1=0.8033
  - Adaptive: P=0.6968, R=0.8123, F1=0.7502
  - winner: DAPI (`experiments/ablation_detection_lock/results.json`)

### T2. E34b 边缘/双核联合消融 (val71) ✅ Completed (2026-02-14)
- 优先级: P0
- 目标: 在 val(71) 上联合重调 `edge_margin`, `size_ratio_threshold`, `merge_coeff`。
- 背景: 当前三项仍沿用历史经验值，尚未纳入统一口径 val 锁定。
- 建议搜索空间:
  - `edge_margin`: [20, 32, 50]
  - `size_ratio_threshold`: [2.0, 2.5, 3.0, 3.5]
  - `merge_coeff`: [1.0, 1.2, 1.4, 1.5]
- 产物:
  - `experiments/ablation_detection_e34b/results.json`
  - [x] 已固定 E34b 参数用于 test 封板
- 结果:
  - 最优: `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`
  - 指标: P=0.7639, R=0.8633, F1=0.8106
- 完成标准:
  - 输出 micro P/R/F1（IoU=0.3）
  - 记录最优参数和二优参数，避免偶然点

### T3. Adaptive 退化诊断补充
- 优先级: P1
- 目标: 用 `adaptive_ratio/fallback_count/mean_zlines` 判断 B2/B3 不敏感原因。
- 产物:
  - `experiments/ablation_adaptive_val/results.json` 中诊断字段
  - 诊断摘要写入 `docs/dapi_detection_design.md`
- 完成标准:
  - 明确是“参数确实不敏感”还是“大量 fallback 导致”

### T4. 默认参数与锁定参数的执行防呆
- 优先级: P1
- 目标: 降低“误用默认参数”风险。
- 方案:
  - 推理/评估脚本显式打印当前 detection 参数
  - 引入 `profile` 机制（`runtime_default` / `locked_eval`）
- 完成标准:
  - 关键脚本输出参数快照
  - 文档写清 profile 选择规则

---

## 2. Mid-Term (Phase 2)

### T5. P2-A 训练与评估闭环
- 优先级: P1
- 目标: 完成 P2-A 训练、Oracle/E2E 评估、回归验证。
- 修复路径: 参照 [`phase2_design.md §7.5`](file:///d:/AI/paper/CellSam/docs/phase2_design.md) 的 fix1-fix4
- 完成标准:
  - fix1 (从 P1 微调) 已执行，config 中 `checkpoint` 指向 P1 best
  - 训练完成且无关键报错
  - 指标与 Phase 1 对比完成（PQ 不低于 P1 baseline 0.475）
  - 若 PQ 仍退化，需追查 loss 设计（fix2/fix3/fix4），不可标记为"完成"

### T6. P2-B 决策门
- 优先级: P2
- 目标: 基于 P2-A 结果决定是否进入 P2-B（Contour/Topology/学习率路线）。
- 完成标准:
  - 有明确 go/no-go 结论与理由

---

## 3. Long-Term (Phase 3+)

### T7. 三通道 Adapter 对比实验
- 优先级: P2
- 目标: BF-only vs Adapter (三通道映射) 在统一推理口径下对比。
- 完成标准:
  - 统一脚本与统一指标
  - 结果可直接用于论文表格

### T8. 推理冲突区域高级策略探索
- 优先级: P3
- 目标: 评估 soft-boundary / watershed / CRF(MRF) 对冲突像素归属的影响。
- 完成标准:
  - 至少一个策略实现可复现实验

---

## 3.5 Documentation Audit (逐文档深度审核)

> 背景: 之前对文档的批量更新不够细致，存在过时参数、口径不一致等遗留问题。
> 原则: 每个文档独立审核，逐节对照源码/config/实验结果，确保无陈旧/错误信息。

### T9. `dataset_parameters.md` 剩余章节更新
- 优先级: P1（随 E34b 结果一起做）
- 前置: T2 (E34b) 完成后才有 val 复核数据
- 待更新章节:
  - §6 边缘过滤参数: 补充 val(71) 复核小节 (`edge_margin` 20/32/50)
  - §7 双核合并参数: 补充 val(71) 复核小节 (`merge_coeff + size_ratio_threshold` 联合)
  - §9 框扩展参数: 区分 DAPI-only vs Adaptive fallback 扩展逻辑，对照 `dapi.py` 函数签名
  - §11 后处理参数: 标注 SSOT 为 `inference_standard.md`，本节仅保统计依据
- 完成标准:
  - 每节参数值与 `dapi.py` / config YAML 一致
  - §12 更新方案表全部标 ✅ Done

### T10. CLAUDE.md 关联文档逐个深度审核
- 优先级: P1
- 目标: 对 CLAUDE.md 核心文档表中的每个 🟢 Active 文档做独立审核，消除过时/不一致信息。
- 审核方式: 每篇独立过一遍，逐节对照源码、config、实验结果。
- 待审核清单:

  | # | 文档 | 审核重点 | 状态 |
  |---|------|---------|------|
  | 10a | `docs/inference_standard.md` | 推理参数与 `src/inference/core.py` 一致性 | [ ] |
  | 10b | `docs/dapi_detection_design.md` | 检测参数锁定表与 E34 结果、代码默认值一致 | [ ] |
  | 10c | `docs/code_inventory.md` | 文件路径/入口是否仍存在、描述是否准确 | [ ] |
  | 10d | `docs/experiments_log.md` | 实验编号连续性、结果数值与日志一致 | [ ] |
  | 10e | `docs/naming_convention.md` | 命名规则与实际代码/文件命名一致 | [ ] |
  | 10f | `docs/error_log_and_checklist.md` | 已解决问题标记完成、新问题补录 | [ ] |
  | 10g | `docs/alice_quick_reference.md` | SLURM 模板/module 名与当前集群环境一致 | [ ] |
  | 10h | `docs/phase2_design.md` | §7 P2A 分析已修正，其余章节是否有陈旧信息 | [ ] |
  | 10i | `docs/progress_timeline_2.13.md` | 时间线节点与实际完成时间一致 | [ ] |
  | 10j | `CLAUDE.md` 自身 | 核心文档表、参数分工表、阶段摘要与最新状态一致 | [ ] |

- 完成标准:
  - 每篇审核后在上表标 [x]
  - 发现的问题就地修复（不积压）
  - 每篇更新后在该文档的更新日志中追加条目

## 4. Completed Recently

- ✅ E34b val(71) 联合消融完成:
  - 最优: `edge_margin=20`, `size_ratio_threshold=2.5`, `merge_coeff=1.4`, F1=0.8106
- ✅ E34 test(73) 单次封板完成:
  - DAPI: F1=0.8033
  - Adaptive: F1=0.7502
  - winner: DAPI
- ✅ 检测消融脚本 GT 面积过滤移除（不再静默修改 GT 分母）
- ✅ `ablation_adaptive_val.py` 支持 `--stage` + `--resume` + detector 参数一致性校验
