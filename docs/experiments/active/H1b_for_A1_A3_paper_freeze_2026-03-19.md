# H1b Paper Freeze Brief (A1/A3)

Date: 2026-03-19  
Owner: A1-H1b

## 1) First-Principles Goal

Freeze one reproducible H1b line for paper writing:

- detector side: candidate-aware CellFinder (`one candidate -> one refined box`)
- E2E side: same detector protocol, only compare segmentation backends
- reporting side: prefer rerun-verified numbers, keep historical outliers as appendix evidence

## 2) Reproducibility Status

- Local and ALICE core H1b files are hash-matched via preflight.
- `cellSAM_source` modifications are inside a nested git repo, not tracked file-by-file in parent repo.
- A reproducible patch artifact is exported:
  - `patches/h1b_cellsam_source_rescue_20260319.patch`

## 3) Locked Comparison (test73)

Protocol: strict + candidate_aligned + q35 + apply_candidate_mask=True

| Detector + Candidate | Seg Backend | P | R | F1 | PQ |
|---|---|---:|---:|---:|---:|
| T33f + adaptive | T27a (`t27a_bf3`) | 0.4301 | 0.5014 | 0.4630 | 0.2732 |
| T33g + dapi_cm | T27a (`t27a_bf3`) | 0.4132 | 0.4562 | 0.4336 | 0.2590 |
| T33f + adaptive | T28 (`t28_legacy3ch`) | 0.6028 | 0.7027 | 0.6490 | 0.4169 |
| T33g + dapi_cm | T28 (`t28_legacy3ch`) | 0.5943 | 0.6562 | 0.6237 | 0.4030 |

Interpretation:

- Under the same detector protocol, `T33f/adaptive` is consistently better than `T33g/dapi_cm`.
- `T28` backend is clearly better than `T27a` for this detector line.

## 4) Paper-Freeze Recommendation

Primary line (main table):

- Detector: `T33f` (adaptive, candidate-aligned, q35)
- Segmenter: `T28` legacy 3ch backend

Secondary line (biology-aware control):

- Detector: `T33g` (dapi_cm)
- Segmenter: `T28`
- role: control/sensitivity, not main best line

Historical high-score note:

- Keep `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35.json` as historical snapshot only.
- Do not use it as the sole freeze number because same-protocol reruns diverge.

## 5) Git/ALICE Minimal Safe Workflow

1. Commit/push parent-repo H1b scripts + docs.
2. Keep `cellSAM_source` changes reproducible via patch artifact:
   - `patches/h1b_cellsam_source_rescue_20260319.patch`
   - apply helper: `scripts/apply_h1b_cellsam_patch.sh`
3. On ALICE, checkout exact commit first, then apply patch before run.
4. Record checkpoint path + eval JSON path in inbox and experiment doc.

## 6) Ready-to-Cite Artifacts

- Detector/E2E inventory:
  - `tmp/h1ba_e2e_metrics_inventory_20260319.json`
  - `tmp/h1ba_t33fg_alice_multiseed_summary_20260319.json`
- Current rerun JSONs:
  - `tmp/h1ba_recall_recovery_e2e_t33f_s123_t28legacy_q35_test_rerun_20260319.json`
  - `tmp/h1ba_recall_recovery_e2e_t33g_s123_t28legacy_q35_test_rerun_20260319.json`
  - `tmp/h1ba_recall_recovery_e2e_t33f_s123_t27a_q35_test_rerun_20260319.json`
  - `tmp/h1ba_recall_recovery_e2e_t33g_s123_t27a_q35_test.json`
