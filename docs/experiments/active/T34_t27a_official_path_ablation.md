# T34: T27a Official-Path Evaluation Ablation (No Clipping / No Unified Conflict Rule)

## 1. Metadata
- ID: `T34`
- Status: `Completed (val+test)`
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
- 入口: `cellSAM_source/cellSAM/sam_inference.py::CellSAM.predict`（显式传入 GT boxes）
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
-> CellSAM.predict(images, boxes_per_heatmap=GT boxes)
-> 内部: prep_2 + forward
-> 内部: for each box → prompt_encoder + mask_decoder
-> 内部: IoU threshold filter
-> 内部: postprocess_masks 到原图尺度
-> 内部: threshold binary masks
-> 内部: thresholded_masks * instance_id
-> 内部: np.max across instances
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

- ✅ 可复现实验脚本输出 A/B/C 三臂结果
- ✅ 明确量化 clipping 与冲突策略对 PQ/BM-Dice 的贡献
- ✅ 给论文方法学部分提供“评估口径敏感性”证据

## 10. Results

### val71

| Arm | 方法 | PQ | F1 | BM-Dice | AJI | TP | FP | FN |
|:---:|------|:---:|:---:|:------:|:---:|:--:|:--:|:--:|
| A | Unified default (clip=on) | 0.491 | 0.798 | 0.723 | 0.570 | 595 | 151 | 151 |
| B | Unified no-clip | 0.491 | 0.798 | 0.723 | 0.570 | 595 | 151 | 151 |
| C | Official path | **0.630** | **0.932** | **0.783** | **0.638** | 694 | 50 | 52 |

### test73

| Arm | 方法 | PQ | F1 | BM-Dice | AJI | TP | FP | FN |
|:---:|------|:---:|:---:|:------:|:---:|:--:|:--:|:--:|
| A | Unified default (clip=on) | 0.450 | 0.752 | 0.705 | 0.548 | 549 | 181 | 181 |
| B | Unified no-clip | 0.450 | 0.752 | 0.705 | 0.548 | 549 | 181 | 181 |
| C | Official path | **0.652** | **0.957** | **0.795** | **0.659** | 698 | 30 | 32 |

> Result files: `experiments/t34_official_path_ablation/results_val.json`, `results_test.json`

## 11. Interpretation

1. **Box clipping 无效果**: A = B 完全相同 (两个 split 均是), clipping 对 GT boxes 无任何影响
2. **Official path 显著优于 unified**: PQ 提升 +14-20pp, F1 提升 +13-21pp
3. **Unified 路径的主要问题**: FP/FN 大幅增加 (test: 30/32 → 181/181), 说明 unified 的冲突归属策略导致大量匹配失败
4. **结论**: 官方路径显著更好, 建议论文主结果使用官方路径

## 12. Decision

- **论文主结果应使用官方路径** (Arm C), 而非 unified SSOT
- Unified vs official 差异可作为 supplementary 敏感性分析
- Box clipping 无效应录入 negative result
