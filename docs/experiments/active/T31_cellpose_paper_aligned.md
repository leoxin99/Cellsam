# T31 Cellpose Paper-Aligned Baseline

## 1. Metadata
- ID: `T31`
- Status: `Draft`
- Owner: `A2`
- Priority: `P0`
- Related task: `docs/task_backlog.md` -> `T31. Cellpose paper-aligned baseline rerun`
- Related plan: `docs/experiments/active/t31_cellpose_baseline_rerun_plan.md`
- Related script: `tools/cellpose_paper_aligned_eval.py` ✅
- Related output dir: `experiments/cellpose_paper_aligned_test73/`

## 2. Background

Historical project Cellpose baseline was run through a deprecated BF-grayscale path in `tools/baseline_eval.py`, which is not aligned with the CellSAM paper/public evaluation methodology.

Direct code evidence:
- `tools/baseline_eval.py:147-152`
- `cellSAM_source/paper_evaluation/eval_main.py:85`
- `cellSAM_source/paper_evaluation/models.py:47`
- `cellSAM_source/paper_evaluation/models.py:92`

## 3. Question / Hypothesis

1. If Cellpose is rerun under the CellSAM public-evaluation methodology, what is its real performance on Allen iPSC-CM `test(73)`?
2. Was the historical near-zero result mostly a methodology error, or is Cellpose intrinsically weak on this task?

## 4. Fixed Conditions

- Dataset split: `test(73)`
- Model type: `cyto3`
- Input encoding: `[blank, DAPI, BF]`
- Channels: `[3,2]`
- Output metrics:
  - project metrics: `PQ`, `BM-1to1 Dice`, `BM-Coverage Dice`, `AJI`, `Semantic Dice`, `TP/FP/FN`
  - CellSAM-paper metrics: `F1`, `Recall`

## 5. Variables

Primary run:
- `diameter=None`

Optional supplementary run:
- `diameter=200`

Optional val-only sensitivity scan:
- `diameter in {120, 160, 200, 240}`

## 6. Execution Plan

1. Create `tools/cellpose_paper_aligned_eval.py`
2. Run main result on `test(73)` using `cyto3 + [0,DAPI,BF] + channels=[3,2] + diameter=None`
3. Save:
   - `results.json`
   - `per_sample_*.json`
4. If necessary, add one supplementary `diameter=200` run
5. Backfill:
   - `docs/experiments_log.md`
   - paper baseline table
   - `docs/agent_inbox.md`

## 7. Expected Risks

1. Allen data does not have a true whole-cell fluorescence channel, so `BF -> whole-cell proxy` remains an approximation.
2. Even after methodological correction, Cellpose may remain weak because cell size/shape domain is mismatched.

## 8. Results

- Run 1: pending
- Run 2: pending
- Aggregate: pending

## 9. Interpretation

Pending execution.

## 10. Decision

- Pending execution
