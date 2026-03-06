# T34: T27a Official-Path Evaluation Ablation (No Clipping / No Unified Conflict Rule)

## 1. Metadata
- ID: `T34`
- Status: `Planned`
- Owner: `A2`
- Priority: `P0`
- Related task: `docs/task_backlog.md` (official-vs-unified evaluation gap)
- Related script (to create): `tools/eval_t34_official_path.py`
- Related output dir: `experiments/t34_official_path_ablation/`
- Related checkpoints: `T27a_PlanB_DecoderOnly_*`

## 2. Background

当前验证/评估主线使用 `segment_with_boxes()`（统一推理核心）。
你要求新增一个对比：移除 box clipping 与统一冲突裁决策略，改走 CellSAM 官方推理路径，检验指标变化。

## 3. Question / Hypothesis

1. T27a 高分有多少来自 unified 规则（box_clipping + argmax_prob 冲突裁决）？
2. 若改走官方 `model.predict` 聚合路径，PQ/BM-Dice 会变化多少？

## 4. Fixed Conditions

- 同一 checkpoint（T27a）
- 同一数据集（先 val71，再 test73）
- 同一指标函数（`compute_all_metrics`）
- 同一检测框输入（GT boxes）

## 5. Arms

### Arm A: Unified default (当前口径)
- 入口: `src/inference/core.py::segment_with_boxes`
- `apply_box_clipping=True`
- `conflict_policy=argmax_prob`
- `apply_postprocess=True`

### Arm B: Unified no-clip
- 同 Arm A，仅 `apply_box_clipping=False`

### Arm C: Official path
- 入口: `cellSAM_source/cellSAM/model.py::predict`
- mask 聚合: `np.max`（无显式 argmax_prob/first_write 规则）
- 不使用 unified 的 per-box clipping
- 按脚本显式记录 `postprocess=False/True` 两个子设置

## 6. Flow Comparison (核心差异)

### 6.1 Unified (segment_with_boxes)

Input image + boxes
-> official_preprocess_and_encode
-> for each box: prompt_encoder + mask_decoder
-> sigmoid mask
-> box clipping (可开关)
-> stack N masks
-> resolve_conflicts(policy=argmax_prob/first_write/last_write)
-> optional morphological postprocess
-> instance mask

### 6.2 Official (CellSAM predict)

Input image + boxes
-> prep_2 + forward
-> for each box: prompt_encoder + mask_decoder
-> threshold binary masks
-> thresholded_masks * instance_id
-> np.max across instances
-> optional postprocess_predictions
-> fill_holes_and_remove_small_masks
-> optional subtract_boundaries
-> instance mask

### 6.3 关键差别总结

1. 冲突归属机制
- Unified: 显式可选策略（默认 argmax_prob）
- Official: 通过 `np.max` 叠加隐式决定像素归属

2. clipping
- Unified: 支持 per-box clipping
- Official: 不做 unified clipping

3. 后处理入口
- Unified: `postprocess_instance_mask`
- Official: `postprocess_predictions + fill_holes_and_remove_small_masks (+subtract_boundaries)`

## 7. Execution Plan

1. 新建 `tools/eval_t34_official_path.py`
2. 跑 val71 三臂对比（A/B/C）
3. 若趋势稳定，再在 test73 复跑封板
4. 回填 `experiments_log.md` 与实验文档

## 8. Expected Risks

1. Official path 与 unified path 在后处理细节不同，绝对数值可能不可直接横向替代
2. Official path 的 `np.max` 冲突归属存在顺序隐式效应
3. 若 C arm 明显下降，需避免误解为“模型退化”，本质是评估口径变化

## 9. Success Criteria

- 可复现实验脚本输出 A/B/C 三臂结果
- 明确量化 clipping 与冲突策略对 PQ/BM-Dice 的贡献
- 给论文方法学部分提供“评估口径敏感性”证据

## 10. Decision Rule

- 若 A 明显优于 C 且差异稳定，则论文主结果继续使用 unified SSOT，并单独报告官方路径敏感性
- 若 C 接近 A，则后续可考虑进一步向官方路径收敛
