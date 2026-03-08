# T30: LoRA on Encoder Q/V Attention (BF-only)

## 1. Metadata
- ID: T30
- Status: ✅ Completed (both seeds)
- Owner: A2
- Priority: P1
- Related config: `src/config/t30_lora_qv_bf.yaml`
- Related scripts: `scripts/train_t30_s42_l4.sh`, `scripts/train_t30_s123_l4.sh`
- Related output dir: `checkpoints/T30_LoRA_QV_BF_seed{42,123}_*`

## 2. Background

T27a 证明 decoder-only fine-tuning 有效 (PQ=0.643)。
SAMed (ICLR 2024) 验证: LoRA on encoder Q/V + full decoder fine-tuning 可进一步提升分割性能。
T30 在 T27a 基础上仅添加 LoRA (rank=4), 测试 encoder 微调对 PQ 的增益。

## 3. Question / Hypothesis

在 frozen encoder + decoder fine-tuning 基础上, 添加 LoRA Q/V 是否能进一步提升 PQ?

## 4. Fixed Conditions

- 与 T27a 完全相同 (BF-only, lr=1e-4, posw=10, etc.)
- checkpoint: null
- freeze_encoder: true (base weights frozen)
- Prompt encoder: frozen (hardcoded in train.py L179-183)
- Neck: frozen (not included in LoRA target)

## 5. Variables

| 参数 | T27a (baseline) | T30 |
|------|:---:|:---:|
| use_lora | false | **true** |
| lora_rank | — | **4** |
| LoRA target | — | Q/V attention (12 blocks) |
| LoRA params | 0 | **147,456** |
| Total trainable | ~4.06M | ~4.21M |

## 6. Execution Plan

- L4 × 2 seeds (42, 123)
- Jobs: 1132286 (s42), 1132287 (s123)
- wall-time: 12h

## 7. Expected Risks

- LoRA rank=4 可能太小, 增益有限
- 梯度 checkpointing 增加训练时间 (~30%)
- L4 VRAM 24GB, 需确认不 OOM (gradient checkpoint 应降至 ~3GB)

## 8. Results

| Seed | Val PQ | Val Dice | Status |
|:----:|:------:|:--------:|:------:|
| 42 | **0.6770** | 0.8136 | ✅ |
| 123 | **0.6699** | 0.8093 | ✅ |
| **Mean** | **0.6735** | **0.8115** | |

## 9. Interpretation

- T30 (LoRA+Decoder) PQ=0.674 vs T27a (Decoder-only) PQ=0.643: **LoRA 提供 +3.1pp 增益**
- 仅增加 147K LoRA 参数 (4.06M → 4.21M, +3.6%), 却获得显著提升
- 双 seed 一致性好 (std < 0.5pp)

## 10. Decision
- **LoRA rank=4 已证实有效**, 后续可考虑 rank=8 或三通道+LoRA 组合
- T30 PQ=0.674 接近 T28 (3ch) PQ=0.686, 但仅用了 BF 单通道
