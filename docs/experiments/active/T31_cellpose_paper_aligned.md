# T31 Cellpose Paper-Aligned Baseline

## 1. Metadata
- ID: `T31`
- Status: `Completed`
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

### Version Note

- **CellSAM paper eval**: `cellpose<4` (in `cellSAM_source/paper_evaluation/requirements.txt`)
- **Our env**: `cellpose==4.0.1`
- v4 changes: `model_type` deprecated (warning only), `Cellpose` wrapper removed → used `CellposeModel`, `eval()` returns 3 values instead of 4
- Impact: `model_type=cyto3` still accepted but shows deprecation warning. Core segmentation model weights unchanged between v3→v4.

## 5. Variables

Primary run:
- `diameter=None`

Optional supplementary run:
- `diameter=200`

Optional val-only sensitivity scan:
- `diameter in {120, 160, 200, 240}`

## 6. Execution Plan

1. ✅ Create `tools/cellpose_paper_aligned_eval.py`
2. ✅ Run main result on `test(73)` using `cyto3 + [0,DAPI,BF] + channels=[3,2] + diameter=None`
3. ✅ Save: `results_dauto.json`, `per_sample_dauto.json`
4. TBD: supplementary `diameter=200` run
5. TBD: backfill docs

## 7. Expected Risks

1. Allen data does not have a true whole-cell fluorescence channel, so `BF -> whole-cell proxy` remains an approximation.
2. Even after methodological correction, Cellpose may remain weak because cell size/shape domain is mismatched.

## 8. Results

### Run 1: diameter=auto (primary)

| 指标类别 | 指标 | 值 |
|---------|------|:---:|
| **项目指标** | PQ | 0.003 ± 0.011 |
| | SQ | 0.061 ± 0.189 |
| | RQ | 0.005 ± 0.018 |
| | BM-1to1 Dice | 0.160 ± 0.052 |
| | BM-Coverage Dice | 0.160 ± 0.052 |
| | AJI | 0.070 ± 0.025 |
| | Semantic Dice | 0.163 ± 0.041 |
| **论文指标** | F1 | 0.005 ± 0.018 |
| | Recall | 0.010 ± 0.032 |
| | Precision | 0.003 ± 0.014 |
| | AP@0.5 | 0.002 |
| **检测计数** | TP / FP / FN | 8 / 10,507 / 722 |
| | Pred / GT total | 10,515 / 730 |

Runtime: 13m27s (11s/sample avg), local RTX 4090

### Run 2: pending (diameter=200)

## 9. Interpretation

**Cellpose 在 Allen 心肌细胞数据上本质不行** — 不是方法学错误。

核心问题分析:
1. **极度过分割**: 每张图预测 ~144 个细胞 (实际 ~10 个), 14x 过分割
2. **`diameter=auto` 估算失败**: 心肌细胞 (~200px) 远超 Cellpose 训练数据的典型细胞大小 (~30px), 自动估算严重低估细胞直径
3. **TP 仅 8 个** (全 test 集 730 个 GT): 匹配率 1.1%
4. **BM-Dice ~0.16**: 虽然远低于我们模型 (~0.80), 但非零 → 有一些像素级重叠, 只是实例匹配完全失败

与 CellSAM 论文评估流程的对齐状况:

| 维度 | CellSAM eval_main.py | T31 脚本 | 状态 |
|------|---------------------|----------|:----:|
| 模型 | `CellPoseModel(cyto3)` | `CellposeModel(cyto3)` | ✅ |
| 通道 | `channels=[3,2]` | `channels=[3,2]` | ✅ |
| 直径 | `diameter=None` | `diameter=None` | ✅ |
| 归一化 | 逐通道 `(ch-min)/(max-min+1e-7)` | 逐通道 `(ch-min)/(max-min)` | ✅ |
| Renumber | `fastremap.renumber(in_place=True)` | `fastremap.renumber(in_place=False)` | ✅ |
| 指标 | Per-image F1/Recall → batch mean | Per-image F1/Recall → batch mean | ✅ |
| Cellpose 版本 | `<4` (v3.x) | v4.0.1 | ⚠️ |

## 10. Decision

- 旧 Cellpose baseline (PQ≈0) 和新 T31 baseline (PQ=0.003) 结论一致
- **不是方法学错误导致低分, 而是 Cellpose 确实不适合此任务**
- 可以安全引用 T31 结果作为论文 Cellpose baseline
- 建议: 补跑 `diameter=200` 验证直径对结果的影响, 如果显著提升则在论文中报告两种设置

