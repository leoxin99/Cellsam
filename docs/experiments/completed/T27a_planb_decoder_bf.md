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

## 9. Interpretation

- 显著超越 MedSAM Oracle (0.576 → 0.643, +6.7pp)
- Plan B 架构有效: model_cp decoder fine-tuning 在 BF-only 下已超越所有 baseline
- 双 seed 结果稳定 (std ~0.5pp)

## 10. Decision
- **Keep as BF-only baseline for后续实验**
- T30 在此基础上加 LoRA
