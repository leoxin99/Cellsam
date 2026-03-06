# T31 Cellpose Paper-Aligned Baseline

## 1. Metadata
- ID: `T31`
- Status: `Completed`
- Owner: `A2`
- Priority: `P0`
- Related task: `docs/task_backlog.md` -> `T31. Cellpose paper-aligned baseline rerun`
- Related plan: `docs/experiments/active/t31_cellpose_baseline_rerun_plan.md`
- Related script: `tools/cellpose_paper_aligned_eval.py` ✅
- Related output dirs:
  - `experiments/cellpose_paper_aligned_test73/` (v4 runs)
  - `experiments/cellpose_v3_test73_dauto/` (v3 d=auto)
  - `experiments/cellpose_v3_test73_d250/` (v3 d=250)

### Code Files

| 文件 | 用途 |
|------|------|
| `tools/cellpose_paper_aligned_eval.py` | 主评估脚本, 支持 `--diameter`, `--output-dir` 参数 |
| `cellSAM_source/paper_evaluation/eval_main.py` | CellSAM 官方 Cellpose eval 参考 |
| `cellSAM_source/paper_evaluation/models.py` | CellSAM 官方 Cellpose model wrapper 参考 |
| `cellSAM_source/paper_evaluation/requirements.txt` | `cellpose<4` 版本约束 |
| `tools/napari_cellpose_vs_t27a.py` | Napari 可视化对比 (Cellpose vs T27a vs GT) |

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
3. Does Cellpose version (v3 vs v4) matter? Does diameter correction change the outcome?

## 4. Fixed Conditions

- Dataset split: `test(73)`
- Model type: `cyto3`
- Input encoding: `[blank, DAPI, BF]`
- Channels: `[3,2]` (cytoplasm=Blue=BF, nucleus=Green=DAPI)
- Output metrics:
  - project metrics: `PQ`, `BM-1to1 Dice`, `BM-Coverage Dice`, `AJI`, `Semantic Dice`, `TP/FP/FN`
  - CellSAM-paper metrics: `F1`, `Recall`

### Version Note

- **CellSAM paper**: `cellpose<4` (v3.x) — `cellSAM_source/paper_evaluation/requirements.txt`
- **v3.1.1.3**: U-Net 架构, model_type=cyto3, 有 SizeModel, 有 `models.Cellpose` class
- **v4.0.1**: **Cellpose-SAM (CP4)**,完全不同的架构, 无 SizeModel, 无 `Cellpose` class, bfloat16 权重, **不兼容 v3 权重**
- **影响**: v4 cyto3 和 v3 cyto3 是**完全不同的模型**, 不只是 API 变化

## 5. Variables

| Run | Version | diameter | 结果文件 |
|:---:|:-------:|:--------:|---------|
| 1 | v4.0.1 | auto | `experiments/cellpose_paper_aligned_test73/results_dauto.json` |
| 2 | v4.0.1 | 250 | `experiments/cellpose_paper_aligned_test73/results_d250.0.json` |
| 3 | v3.1.1 | auto | `experiments/cellpose_v3_test73_dauto/results_dauto.json` |
| 4 | v3.1.1 | 250 | `experiments/cellpose_v3_test73_d250/results_d250.0.json` |

## 6. Execution Plan

1. ✅ Create `tools/cellpose_paper_aligned_eval.py`
2. ✅ Run v4 d=auto on test(73)
3. ✅ Run v4 d=250 on test(73)
4. ✅ Install cellpose v3.1.1.3, run v3 d=auto on test(73)
5. ✅ Run v3 d=250 on test(73)
6. ✅ Restore env to cellpose v4.0.1

## 7. Expected Risks

1. Allen data does not have a true whole-cell fluorescence channel, so `BF -> whole-cell proxy` remains an approximation.
2. Even after methodological correction, Cellpose may remain weak because cell size/shape domain is mismatched.

## 8. Results

### Full Comparison Table

| Version | diameter | PQ | F1 | Recall | Precision | TP | FP | FN | Pred/GT | Runtime |
|:-------:|:-------:|:---:|:---:|:------:|:---------:|:---:|:----:|:---:|:------:|:-------:|
| **v3.1.1** | **250** | **0.273** | **0.425** | **0.474** | **0.427** | **334** | **429** | **396** | **763/730** | 5m05s |
| v4.0.1 | 250 | 0.120 | 0.190 | 0.231 | 0.168 | 182 | 1088 | 548 | 1270/730 | 4m29s |
| v4.0.1 | auto | 0.003 | 0.005 | 0.010 | 0.003 | 8 | 10507 | 722 | 10515/730 | 13m27s |
| v3.1.1 | auto | 0.000 | 0.001 | 0.001 | 0.000 | 1 | 1832 | 729 | 1833/730 | 4m29s |

### Run 1: v4.0.1 d=auto

| 指标类别 | 指标 | 值 |
|---------|------|:---:|
| **项目指标** | PQ | 0.003 ± 0.011 |
| | SQ | 0.061 ± 0.189 |
| | RQ | 0.005 ± 0.018 |
| | BM-1to1 Dice | 0.160 ± 0.052 |
| | AJI | 0.070 ± 0.025 |
| | Semantic Dice | 0.163 ± 0.041 |
| **论文指标** | F1 | 0.005 ± 0.018 |
| | Recall | 0.010 ± 0.032 |
| | AP@0.5 | 0.002 |
| **检测计数** | TP / FP / FN | 8 / 10,507 / 722 |

### Run 2: v4.0.1 d=250

| 指标类别 | 指标 | 值 |
|---------|------|:---:|
| **项目指标** | PQ | 0.120 ± 0.163 |
| | SQ | 0.275 ± 0.313 |
| | BM-1to1 Dice | 0.362 ± 0.212 |
| | AJI | 0.189 ± 0.146 |
| | Semantic Dice | 0.383 ± 0.256 |
| **论文指标** | F1 | 0.190 ± 0.257 |
| | Recall | 0.231 ± 0.316 |
| | AP@0.5 | 0.132 |
| **检测计数** | TP / FP / FN | 182 / 1,088 / 548 |

### Run 3: v3.1.1 d=auto

| 指标类别 | 指标 | 值 |
|---------|------|:---:|
| **项目指标** | PQ | 0.000 ± 0.003 |
| | BM-1to1 Dice | 0.206 ± 0.056 |
| | AJI | 0.091 ± 0.027 |
| | Semantic Dice | 0.202 ± 0.044 |
| **论文指标** | F1 | 0.001 ± 0.006 |
| | Recall | 0.001 ± 0.010 |
| | AP@0.5 | 0.000 |
| **检测计数** | TP / FP / FN | 1 / 1,832 / 729 |

### Run 4: v3.1.1 d=250 ⭐ (CellSAM paper-aligned best)

| 指标类别 | 指标 | 值 |
|---------|------|:---:|
| **项目指标** | PQ | 0.273 ± 0.152 |
| | SQ | 0.600 ± 0.155 |
| | RQ | 0.425 ± 0.225 |
| | BM-1to1 Dice | 0.505 ± 0.203 |
| | BM-Coverage Dice | 0.522 ± 0.195 |
| | AJI | 0.285 ± 0.164 |
| | Semantic Dice | 0.603 ± 0.196 |
| **论文指标** | F1 | 0.425 ± 0.225 |
| | Recall | 0.474 ± 0.284 |
| | Precision | 0.427 ± 0.226 |
| | AP@0.5 | 0.297 |
| **检测计数** | TP / FP / FN | 334 / 429 / 396 |

Runtime: 5m05s (4.2s/sample), local RTX 4090

## 9. Interpretation

### 核心发现

1. **v3 d=250 是最佳 Cellpose 结果** (PQ=0.273, F1=0.425)
   - CellSAM 论文用 v3 (`cellpose<4`), 所以 v3 是正确的对比版本
   - diameter=250 匹配心肌细胞实际大小 (GT equiv diameter mean=255px on 1024x1024)

2. **v4 架构变化导致性能下降** — v4 cyto3 ≠ v3 cyto3
   - v4 是 Cellpose-SAM (全新架构), 不兼容 v3 权重
   - v4 d=250: PQ=0.120 vs v3 d=250: PQ=0.273 (下降 56%)

3. **diameter=auto 对两个版本都完全失败** — SizeModel (v3) 或默认行为 (v4) 都无法处理 ~250px 细胞

4. **即使最优 Cellpose (v3 d=250, PQ=0.273), 仍远低于我们的模型 (T27a PQ~0.67)**

### CellSAM eval 对齐表

| 维度 | CellSAM eval_main.py | T31 脚本 | 状态 |
|------|---------------------|----------|:----:|
| 模型 | `CellPoseModel(cyto3)` | `Cellpose/CellposeModel(cyto3)` | ✅ |
| 通道 | `channels=[3,2]` | `channels=[3,2]` | ✅ |
| 直径 | `diameter=None` | `diameter=None/250` | ✅ |
| 归一化 | 逐通道 `(ch-min)/(max-min+1e-7)` | 逐通道 `(ch-min)/(max-min)` | ✅ |
| Renumber | `fastremap.renumber` | `fastremap.renumber` | ✅ |
| 指标 | Per-image F1/Recall → batch mean | Per-image F1/Recall → batch mean | ✅ |
| Cellpose 版本 | `<4` (v3.x) | v3.1.1 (Run 3,4) / v4.0.1 (Run 1,2) | ✅ |

## 10. Decision

- **论文 Cellpose baseline**: 使用 **v3.1.1 d=250** 结果 (PQ=0.273, F1=0.425)
  - 理由: CellSAM 论文用 `cellpose<4`, diameter=250 匹配数据实际细胞大小
- v4 结果 可作为补充说明 ("newer Cellpose-SAM version performs worse on this task")
- v3/v4 d=auto 结果表明自动直径估算在心肌细胞上完全失败
- **最终结论**: 即使在最公平设置下, Cellpose PQ=0.273 仍远低于 T27a PQ~0.67

