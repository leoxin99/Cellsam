# T27a: Plan B Decoder-Only (BF-only)

## 1. Metadata
- ID: T27a
- Status: ✅ Completed (both seeds)
- Owner: A2
- Priority: P0
- Related config: `src/config/t27a_planb_decoder.yaml`
- Related script: `scripts/train_t27a_l4.sh`
- Related output dir: `checkpoints/T27a_PlanB_DecoderOnly_seed{42,123}_*`

## 2. Background

Plan B 架构: 使用 `model_cp` 分支 (Stage 2 weights) + 官方预处理管线。
只训练 mask_decoder, 冻结 image_encoder + prompt_encoder。
BF-only 输入 (BF 复制到 3 通道: [BF, BF, BF])。

## 3. Question / Hypothesis

用 model_cp 的 decoder 微调能否超越 MedSAM Oracle baseline (PQ=0.576)?

## 4. Fixed Conditions

- Encoder: frozen (model_cp.image_encoder)
- Prompt encoder: frozen (512 params, positional only)
- Input: BF-only [BF, BF, BF]
- checkpoint: null (从 CellSAM 官方预训练权重开始)
- lr=1e-4, posw=10, boundary=0.3, focal=true, iou_weight=0.1
- Early stop: PQ patience=15, max 80 epochs

## 5. Variables

- Seed: 42, 123

## 6. Execution Plan

- L4 × 2 seeds → 完成

## 7. Expected Risks

- BF 单通道信息有限, 可能无法充分利用 encoder

## 8. Results

| Seed | Val PQ | Val Dice | Best Epoch | Runtime |
|:----:|:------:|:--------:|:----------:|:-------:|
| 42 | 0.6378 | 0.7911 | ~E54 | ~4.5h |
| 123 | 0.6481 | 0.7985 | — | 10h49m |
| **Mean** | **0.6430** | **0.7948** | | |

### 8.1 Full Metrics (eval_checkpoint.py, IoU=0.5)

| Split | PQ | SQ | RQ | F1 | P | R | BM-Dice | AJI | Sem Dice | TP | FP | FN |
|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:------:|:---:|:--------:|:--:|:--:|:--:|
| val71 | 0.649 | 0.684 | 0.946 | **0.944** | 0.944 | 0.944 | 0.798 | 0.667 | 0.842 | 704 | 42 | 42 |
| test73 | 0.659 | 0.683 | 0.964 | **0.960** | 0.960 | 0.960 | 0.800 | 0.669 | 0.837 | 701 | 29 | 29 |

> Checkpoint: `T27a_PlanB_DecoderOnly_20260302_033621/best_model.pt`
> Result file: `experiments/t27a_eval/results.json`

### 8.2 DAPI Detection Evaluation (E2E, IoU=0.5)

用 DAPI 检测替代 GT boxes 的端到端评估。两种检测方法：

| 方法 | Split | PQ | F1 | P | R | BM-Dice | AJI | 检测/GT | TP/FP/FN |
|------|:-----:|:---:|:---:|:---:|:---:|:------:|:---:|:------:|:-------:|
| **核检测** | val | 0.254 | 0.434 | 0.409 | 0.462 | 0.602 | 0.366 | 11.9/10.5 | 345/498/401 |
| **核检测** | test | 0.252 | 0.433 | 0.402 | 0.469 | 0.599 | 0.364 | 11.7/10.0 | 342/509/388 |
| **Z 线自适应** | val | 0.299 | 0.507 | 0.478 | 0.540 | 0.615 | 0.368 | 11.9/10.5 | 403/440/343 |
| **Z 线自适应** | test | 0.293 | 0.497 | 0.462 | 0.538 | 0.612 | 0.361 | 11.7/10.0 | 393/458/337 |

> Detection profile: `locked_eval`
> Result file: `experiments/t27a_dapi_eval/results.json`
> Script: `tools/eval_dapi_detection.py`

**DAPI 检测分析**：
- Z 线自适应优于纯核检测（F1 +7%），因 Actn2 Z 线提供了更准确的细胞边界估计
- RQ 从 GT boxes 的 ~0.95 降到 ~0.42-0.48，说明**检测质量是 E2E 瓶颈**
- FP 高（~400-500）说明检测产生大量错误 box
- SQ 差距较小（0.58 vs 0.68），分割模型本身性能尚可

## 9. Interpretation

- 显著超越 MedSAM Oracle (0.576 → 0.643, +6.7pp)
- Plan B 架构有效: model_cp decoder fine-tuning 在 BF-only 下已超越所有 baseline
- 双 seed 结果稳定 (std ~0.5pp)
- E2E 瓶颈在检测而非分割

## 10. Decision
- **Keep as BF-only baseline for后续实验**
- T30 在此基础上加 LoRA

