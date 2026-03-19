# H1bA T33f/T33g Candidate-Aware Retrain Update (2026-03-18)

## 1) Scope

This update records the completed `q35` candidate-aware CellFinder retraining runs:

- `T33f`: `adaptive` candidates (`strict`, `apply_candidate_mask=True`)
- `T33g`: `dapi_cm` candidates (`strict`, `apply_candidate_mask=True`)

All runs use:

- `epochs=150`, `patience=20`
- `early_stop_metric=candidate_aligned_f1_0p3`
- `num_queries=35`

---

## 2) Completed Training Runs

| Run | Job | GPU | Elapsed | Early stop epoch | Best candidate_aligned_f1@0.3 |
|---|---:|---|---:|---:|---:|
| `t33f_q35_s42_l4` | `1233323` | L4 | `01:30:50` | `27` | `0.8395` |
| `t33f_q35_s123_l4` | `1233324` | L4 | `01:27:43` | `26` | `0.8420` |
| `t33f_q35_s42_a100` | `1233296` | A100 | `00:55:50` | `24` | `0.8408` |
| `t33f_q35_s123_a100` | `1233297` | A100 | `00:53:22` | `25` | `0.8408` |
| `t33g_q35_s42_l4` | `1233325` | L4 | `01:16:16` | `25` | `0.8255` |
| `t33g_q35_s123_l4` | `1233326` | L4 | `01:17:08` | `25` | `0.8203` |
| `t33g_q35_s42_a100` | `1233301` | A100 | `00:44:51` | `26` | `0.8191` |
| `t33g_q35_s123_a100` | `1233302` | A100 | `00:43:01` | `25` | `0.8229` |

---

## 3) Primary Metric Comparison (Current Decision Metric)

- `T33f` mean candidate-aligned F1@0.3: `0.8408`
- `T33g` mean candidate-aligned F1@0.3: `0.8220`
- Absolute gap: `+0.0188` (`T33f - T33g`)

Interpretation:

- Under current one-candidate-one-box objective, `adaptive` remains better than `dapi_cm`.
- Cross-hardware direction is consistent (L4 and A100 both favor `T33f`).

---

## 4) About AP50 (Secondary Diagnostic Only)

`AP50` is retained as a secondary detector ranking diagnostic, but it is **not** the primary promotion metric for H1bA candidate-aligned runtime policy.

- Historical `T33c` best `AP50`:
  - seed42: `0.5111`
  - seed123: `0.5241`
- `T33f/T33g` runs reached higher AP50 peaks (roughly `0.57~0.61`), indicating better score-ranking behavior.

Important:

- AP-type metrics summarize ranking curves and do not directly encode the deployment rule of "emit one refined box per valid candidate".
- Promotion decisions for this branch should prioritize candidate-aligned metrics and downstream E2E metrics.

---

## 5) Relation To Previous F1@0.3 / Box Quality Results

### 5.1 Previous formal detector eval (old detector checkpoint: `T33c`)

From `tmp/h1ba_recall_recovery_detector_eval_t33c.json`:

- `h1ba_adaptive_candidate_aligned_nodrop`
  - val: `F1@0.3=0.8018`, `recall@0.3=0.8539`, `precision@0.3=0.7556`, `avg_matched_box_iou=0.5632`
  - test: `F1@0.3=0.7982`, `recall@0.3=0.8644`, `precision@0.3=0.7415`, `avg_matched_box_iou=0.5652`

### 5.2 Current retrain logs (`T33f/T33g`)

Best logged candidate-aligned values in training:

- `T33f` best (seed123 L4): `CandF1@0.3=0.8420`, `CandP@0.3=0.7936`, `CandR@0.3=0.8968`
- `T33g` best (seed42 L4): `CandF1@0.3=0.8255`, `CandP@0.3=0.7943`, `CandR@0.3=0.8592`

Caveat:

- Training-log candidate metrics and formal detector-table metrics are close but not perfectly identical protocols.
- For strict "box quality vs previous" claims (especially IoU/count-error tables), run the same formal detector harness with the new checkpoints.

---

## 6) Downstream E2E Status

Formal E2E with promoted run has been submitted:

- Job: `1253855`
- Arm: `T33f seed123` checkpoint vs `raw_cellfinder`
- Output target:
  - `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json`

Status should be checked via `squeue/sacct` before citing final E2E deltas.
