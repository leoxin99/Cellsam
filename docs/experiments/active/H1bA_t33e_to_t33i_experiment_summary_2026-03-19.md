# H1bA T33e~T33i Experiment Summary (2026-03-19)

## 1) Scope

This note consolidates the H1bA candidate-aware CellFinder fine-tuning line:

- `T33e`: candidate-aware smoke checks
- `T33f`: adaptive + strict + q35 + early stop on `candidate_aligned_f1@0.3`
- `T33g`: dapi_cm + strict + q35 + early stop on `candidate_aligned_f1@0.3`
- `T33h`: adaptive + strict + q35 + early stop on `candidate_aligned_f1@0.5`
- `T33i`: adaptive + strict + q35 + early stop on `candidate_aligned_f1@0.7`

Goal: determine which detector checkpoint and downstream segmentation pairing should be frozen for paper reporting.

---

## 2) Training Supervision / Metric Definitions

- Detector training supervision is **box-supervised**:
  - instance GT mask -> per-instance GT box (`cx,cy,w,h`) -> DETR matching/loss.
  - implementation:
    - `tools/train_cellfinder.py` (`AllenDetectionDataset._masks_to_cxcywh`)
    - `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py` (`loss_ce`, `loss_bbox`, `loss_giou`)
- H1bA candidate-aware training (`tools/train_cellfinder_candidate_aware.py`) reuses the same supervision, and adds prior-conditioned queries during train/val.
- Candidate-source difference used by `T33f` vs `T33g`:
  - `adaptive` (`T33f`): nucleus candidates without Actn2 hard coverage filtering in center-only mode.
  - `dapi_cm` (`T33g`): nucleus groups are filtered by Actn2 coverage (`filter_by_actn2`, threshold `0.3`) before entering query priors.
- Official CellSAM paper training description (for reference):
  - Stage-1: train ViT + CellFinder on object detection by converting GT masks to GT boxes.
  - Stage-2: freeze ViT and fine-tune SAM neck with GT boxes + segmentation labels.
  - source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/ (Fig. 1 / Results section text)
- E2E F1 used in this repo is TP/FP/FN aggregated at instance IoU threshold 0.5.
- CellSAM paper-eval F1 (`cellSAM_source/paper_evaluation/cpm.py`) is:
  - `F1 = TP / (TP + 0.5 * (FP + FN))`
  - this is numerically equivalent to harmonic F1 from precision/recall when computed from the same TP/FP/FN.

---

## 3) Checkpoint-Level Results (Key Points)

### 3.1 `T33e` smoke only

- `checkpoints/T33e_candidateaware_smoke`
- `checkpoints/T33e_candidateaware_smoke_metric03`
- 1 epoch smoke runs used to verify candidate-aware pipeline and metric plumbing, not for final selection.

### 3.2 `T33f` vs `T33g` (primary retrain round)

From detector retrain logs and existing summary:

- `T33f` best val `candidate_aligned_f1@0.3`: `0.8420`
- `T33g` best val `candidate_aligned_f1@0.3`: `0.8203`

E2E (test73, T27a protocol, candidate-aligned arm):

- `T33f`: `P=0.5828, R=0.6795, F1=0.6275, PQ=0.3981`
  - source: `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json`
- `T33g`: existing direction remains behind `T33f` in this round (see `H1bA_t33fg_candidateaware_retrain_update_2026-03-18.md`).

### 3.3 Reproducibility audit (local rerun, 2026-03-19)

A same-protocol local rerun was executed to verify current code-path behavior on `test73`:

- protocol:
  - detector checkpoint: `T33f` or `T33g`
  - segmentation checkpoint: `T27a`
  - `prior_mode=strict`, `query_output_mode=candidate_aligned`, `apply_candidate_mask=True`
  - `num_queries=35`, split=`test73`

Results:

- `T33f` (`adaptive` candidates):
  - `P=0.4301, R=0.5014, F1=0.4630, PQ=0.2732`
  - source: `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_rerun_20260319.json`
- `T33g` (`dapi_cm` candidates):
  - `P=0.4132, R=0.4562, F1=0.4336, PQ=0.2590`
  - source: `tmp/h1ba_recall_recovery_e2e_t33g_s123_t27a_q35_test.json`
- `T33f - T33g`:
  - `ΔF1=+0.0294`
  - `ΔPQ=+0.0142`
  - source: `tmp/h1ba_t33f_vs_t33g_local_rerun_compare_20260319.json`

Interpretation:

- On current reproducible local path, `T33f (adaptive)` remains better than `T33g (dapi_cm)`, but margin is modest.
- This does not support switching default source to `dapi_cm` in the current round.

### 3.4 About the older high-score snapshot

Historical result file:

- `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json`

contains substantially higher values (`F1=0.6275`, `PQ=0.3981`) than current local rerun.  
Given the mismatch, treat that snapshot as **non-locked historical evidence** (likely protocol/runtime drift), and do not use it as sole basis for paper final tables.

Important: this mismatch is not explained by edge filtering alone.

- edgeoff run (`T33f + T27a`): `F1=0.4262`, `PQ=0.2536`
- edge-filter restored local rerun (`T33f + T27a`): `F1=0.4630`, `PQ=0.2732`

Edge-filter restoration helps, but does not recover the historical high-score snapshot.

### 3.5 `T33h` / `T33i` (0.5 / 0.7 early-stop probes)

Training completion was verified in inbox and checkpoints were synced locally:

- `T33h` best early-stop metric (`candidate_aligned_f1@0.5`): `0.7086`
- `T33i` best early-stop metric (`candidate_aligned_f1@0.7`): `0.2756`

E2E (test73, T27a protocol, candidate-aligned arm):

- `T33h`: `P=0.4913, R=0.7315, F1=0.5878, PQ=0.3780`
  - source: `tmp/h1ba_recall_recovery_e2e_t33h_s123_t27a_q35_test.json`
- `T33i`: `P=0.4940, R=0.7356, F1=0.5911, PQ=0.3786`
  - source: `tmp/h1ba_recall_recovery_e2e_t33i_s123_t27a_q35_test.json`

Result: both `T33h/T33i` are below current `T33f` best E2E point.

---

## 4) Edge-Filter Removal Impact (Adaptive Candidate)

Ablation tested removing edge-nucleus dropping in candidate generation, then reverted after metrics dropped:

- `src/detection/h1b_priors.py` (`_detect_adaptive_candidates`, `_detect_dapi_cm_candidates`)

Measured impact for `T33f + T27a` (candidate-aligned arm, test73):

- before edge-filter removal:
  - `F1=0.6275`, `PQ=0.3981`
  - `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json`
- after edge-filter removal:
  - `F1=0.4262`, `PQ=0.2536`
  - `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_edgeoff.json`

Delta (`edgeoff - before`):

- `F1: -0.2012`
- `PQ: -0.1445`

Conclusion: under current locked protocol, removing edge filtering is strongly negative.

---

## 5) T27a vs T28 Segmentation Backend Check (No retraining)

For the same `T33f` detector on edgeoff candidate generation:

- `T27a` backend:
  - `F1=0.4262`, `PQ=0.2536`
- `T28 legacy 3ch` backend:
  - `F1=0.6035`, `PQ=0.3923`

Source:

- `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_edgeoff.json`
- `tmp/h1ba_recall_recovery_e2e_t33f_s123_t28legacy_q35_test_edgeoff.json`
- compare file: `tmp/h1ba_t33f_t27a_vs_t28_e2e_test_edgeoff_compare.json`

Interpretation: T28 mapping recovers most of edgeoff damage, but does not surpass `T33f` pre-edgeoff + T27a best point.

### 5.1 Current-code rerun (edge filtering restored)

Using current code path (edge filtering restored), same `T33f` detector + candidate-aligned strict:

- `T27a` backend (rerun):
  - `P=0.4301, R=0.5014, F1=0.4630, PQ=0.2732`
  - source: `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_rerun_20260319.json`
- `T28 legacy 3ch` backend (rerun):
  - `P=0.6028, R=0.7027, F1=0.6490, PQ=0.4169`
  - source: `tmp/h1ba_recall_recovery_e2e_t33f_s123_t28legacy_q35_test_rerun_20260319.json`

In this rerun pair, `T28` is clearly better than `T27a`.

---

## 6) Freeze Recommendation (Current)

If we freeze now on available evidence:

1. Keep `T33f` as best detector checkpoint among T33e~i.
2. Do not keep unconditional edge-filter removal in final reported configuration.
3. Use one locked E2E protocol for final tables (avoid mixing protocol shifts with detector ablations).
