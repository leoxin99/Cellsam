# H1b CellFinder 实验总表（截至 2026-03-19）

## 0) 分支与封板策略

- 当前工作分支：`rescue/h1b-sync-20260319`
- 封板策略：先在 `rescue` 完成收敛与口径统一，再合并回 `main`。
- 本文档目标：把 H1b 线 CellFinder 相关实验、指标、可复现状态一次性对齐，供 A1/A3 直接使用。

---

## 1) H1b 线 CellFinder 实验全景

### 1.1 预备线（T33 基础适配）

- `T33`: head-only，F1@0.5 监控（双 seed）
- `T33b`: head-only，加入 COCO AP 指标与 AP50 早停（双 seed）
- `T33c`: 无早停长训复核（双 seed）
- `T33d`: LoRA 探索（种子敏感，不作为主线结论）

参考文档：
- [T33_cellfinder_finetune_plan.md](d:/AI/paper/CellSam/docs/experiments/active/T33_cellfinder_finetune_plan.md)
- [T33b_cellfinder_finetune_v2_coco_map.md](d:/AI/paper/CellSam/docs/experiments/active/T33b_cellfinder_finetune_v2_coco_map.md)

### 1.2 H1bA 正式线（prior-conditioned candidate-aware）

- `T33e`: smoke（流程联通）
- `T33f`: `adaptive + strict + q35 + early_stop=candidate_aligned_f1@0.3`
- `T33g`: `dapi_cm + strict + q35 + early_stop=candidate_aligned_f1@0.3`
- `T33h`: `adaptive + strict + q35 + early_stop=candidate_aligned_f1@0.5`
- `T33i`: `adaptive + strict + q35 + early_stop=candidate_aligned_f1@0.7`

参考文档：
- [H1bA_t33e_to_t33i_experiment_summary_2026-03-19.md](d:/AI/paper/CellSam/docs/experiments/active/H1bA_t33e_to_t33i_experiment_summary_2026-03-19.md)
- [H1bA_t33fg_candidateaware_retrain_update_2026-03-18.md](d:/AI/paper/CellSam/docs/experiments/active/H1bA_t33fg_candidateaware_retrain_update_2026-03-18.md)

---

## 2) Detector 侧结果（框质量）

### 2.1 T33c runtime 变体（test73，IoU=0.3）

来源：[h1ba_recall_recovery_detector_eval_t33c.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_detector_eval_t33c.json)

| Variant | P@0.3 | R@0.3 | F1@0.3 | pred/img |
|---|---:|---:|---:|---:|
| raw_cellfinder | 0.6331 | 0.6973 | 0.6636 | 11.014 |
| h1ba_adaptive_strict_fixed0.30 | 0.7959 | 0.5877 | 0.6761 | 7.384 |
| h1ba_adaptive_strict_fixed0.28 | 0.7983 | 0.6288 | 0.7034 | 7.877 |
| h1ba_adaptive_strict_fixed0.25 | 0.7946 | 0.6890 | 0.7381 | 8.671 |
| h1ba_adaptive_candidate_aligned_nodrop | 0.7415 | 0.8644 | 0.7982 | 11.658 |
| h1ba_adaptive_hybrid_open_fixed0.30 | 0.5054 | 0.8274 | 0.6275 | 16.370 |
| h1ba_adaptive_hybrid_open_fixed0.28 | 0.4724 | 0.8562 | 0.6089 | 18.123 |
| h1ba_adaptive_hybrid_open_fixed0.25 | 0.4314 | 0.9041 | 0.5841 | 20.959 |

结论：
- `candidate_aligned_nodrop` 的检测 F1@0.3 最高（0.7982）。
- `hybrid_open` 召回高但 FP 明显增加。

### 2.2 candidate source 对比（T33c，test73）

来源：[h1b_runtime_source_ablation_t33c.json](d:/AI/paper/CellSam/tmp/h1b_runtime_source_ablation_t33c.json)

| Source | P@0.3 | R@0.3 | F1@0.3 |
|---|---:|---:|---:|
| adaptive | 0.8054 | 0.5274 | 0.6374 |
| dapi_cm | 0.8013 | 0.5027 | 0.6178 |

结论：在 detector 侧，`adaptive` 优于 `dapi_cm`。

### 2.3 score policy 对比（T33c，test73）

来源：[h1b_runtime_threshold_ablation_t33c.json](d:/AI/paper/CellSam/tmp/h1b_runtime_threshold_ablation_t33c.json)

| Policy | P@0.3 | R@0.3 | F1@0.3 | pred/img |
|---|---:|---:|---:|---:|
| dynamic | 0.6854 | 0.4000 | 0.5052 | 5.836 |
| fixed_0p3 | 0.6802 | 0.4603 | 0.5490 | 6.767 |
| none | 0.1605 | 0.8027 | 0.2676 | 50.000 |

结论：`fixed_0p3` 明显优于 `dynamic`，`none` 会导致过量框。

---

## 3) Candidate-aware 训练结果（多 seed）

来源：
- [h1ba_t33fg_alice_multiseed_inventory_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_t33fg_alice_multiseed_inventory_20260319.json)
- [h1ba_t33fg_alice_multiseed_summary_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_t33fg_alice_multiseed_summary_20260319.json)

统计口径：
- 仅纳入 `has_best=true` 且 `training_history.json` 存在的成功作业。
- 覆盖 L4/A100，seed42/123，共 8 个成功 run（`T33f` 4 个，`T33g` 4 个）。

| Group | n_runs | best val cand-F1@0.3 | mean val cand-F1@0.3 | std |
|---|---:|---:|---:|---:|
| T33f (adaptive) | 4 | 0.8420 | 0.8408 | 0.0009 |
| T33g (dapi_cm) | 4 | 0.8255 | 0.8220 | 0.0025 |

结论：
- `T33f` 在多 seed + 多 GPU 下稳定优于 `T33g`。

---

## 4) E2E 结果（test73）

### 4.1 当前可复现主结果（2026-03-19 重跑）

| Detector + Candidate | Seg backend | P | R | F1 | PQ |
|---|---|---:|---:|---:|---:|
| T33f + adaptive candidate_aligned | T27a (`t27a_bf3`) | 0.4301 | 0.5014 | 0.4630 | 0.2732 |
| T33g + dapi_cm candidate_aligned | T27a (`t27a_bf3`) | 0.4132 | 0.4562 | 0.4336 | 0.2590 |
| T33f + adaptive candidate_aligned | T28 (`t28_legacy3ch`) | 0.6028 | 0.7027 | 0.6490 | 0.4169 |
| T33g + dapi_cm candidate_aligned | T28 (`t28_legacy3ch`) | 0.5943 | 0.6562 | 0.6237 | 0.4030 |

来源：
- [h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_rerun_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_rerun_20260319.json)
- [h1ba_recall_recovery_e2e_t33g_s123_t27a_q35_test.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_e2e_t33g_s123_t27a_q35_test.json)
- [h1ba_recall_recovery_e2e_t33f_s123_t28legacy_q35_test_rerun_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_e2e_t33f_s123_t28legacy_q35_test_rerun_20260319.json)
- [h1ba_recall_recovery_e2e_t33g_s123_t28legacy_q35_test_rerun_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_e2e_t33g_s123_t28legacy_q35_test_rerun_20260319.json)

关键比较：
- `adaptive(T33f)+T28` vs `dapi_cm(T33g)+T28`：
  - `ΔF1 = +0.0253`
  - `ΔPQ = +0.0139`

### 4.2 历史快照（非锁定主表口径）

`T33f + T27a` 曾出现历史高分：
- `P=0.5828, R=0.6795, F1=0.6275, PQ=0.3981`
- 文件：[h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json](d:/AI/paper/CellSam/tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json)

当前处理原则：
- 该结果与同协议复跑绝对值不一致，先作为历史快照保留，不直接用于封板主表。

---

## 5) 当前结论（给 A1/A3）

1. 检测端（多 seed）`T33f(adaptive)` 稳定优于 `T33g(dapi_cm)`。  
2. E2E 端当前可复现最优是 `T33f + adaptive candidate_aligned + T28 legacy3ch`。  
3. 若必须保留 Actn2 过滤路线，`dapi_cm` 当前最优是 `T33g + dapi_cm + T28`，但仍次于 `adaptive + T28`。  
4. 论文主表应优先使用“2026-03-19 可复现实验口径”，历史高分快照放附录说明。  

---

## 6) 本次汇总产物

- [h1b_cellfinder_master_summary_20260319.json](d:/AI/paper/CellSam/tmp/h1b_cellfinder_master_summary_20260319.json)
- [h1ba_e2e_metrics_inventory_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_e2e_metrics_inventory_20260319.json)
- [h1ba_e2e_metrics_inventory_20260319.csv](d:/AI/paper/CellSam/tmp/h1ba_e2e_metrics_inventory_20260319.csv)
- [h1ba_t33fg_alice_multiseed_inventory_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_t33fg_alice_multiseed_inventory_20260319.json)
- [h1ba_t33fg_alice_multiseed_summary_20260319.json](d:/AI/paper/CellSam/tmp/h1ba_t33fg_alice_multiseed_summary_20260319.json)
