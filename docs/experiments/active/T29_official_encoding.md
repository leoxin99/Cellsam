# T29: Official Channel Encoding Ablation

## 1. Metadata
- ID: T29a / T29b / T29c
- Status: ✅ Completed (both seeds)
- Owner: A2
- Priority: P0
- Related configs:
  - `src/config/t29a_official_bf.yaml`
  - `src/config/t29b_official_3ch.yaml`
  - `src/config/t29c_official_3ch_actn2.yaml`
- Related scripts: `scripts/train_t29{a,b,c}_l4.sh`, `scripts/train_t29{a,b,c}_s123_*.sh`
- Related output dir: `checkpoints/T29{a,b,c}_Official_*`

## 2. Background

CellSAM 论文定义输入编码: (R=blank, G=nuclear, B=whole-cell)。
代码确认: `cellsam_pipeline.py` L84-87, L158-163。
我们的 T28 使用 [BF,Actn2,DAPI] — 三个通道全部与官方不一致。
T29 系列测试对齐官方编码的效果。

## 3. Question / Hypothesis

1. BF-only: [0,0,BF] vs [BF,BF,BF] 有差异吗?
2. 官方 3ch [0,DAPI,BF] vs 旧 3ch [BF,Actn2,DAPI] 哪个好?
3. R 通道放 Actn2 [Actn2,DAPI,BF] 是否有增益?

## 4. Fixed Conditions

- 与 T27a/T28 完全相同的超参 (lr=1e-4, posw=10, boundary=0.3, focal, etc.)
- checkpoint: null, freeze_encoder: true, PE frozen
- use_official_encoding: true

## 5. Variables

| 子实验 | R (Ch0) | G (Ch1) | B (Ch2) | 对照 |
|--------|---------|---------|---------|------|
| T29a | zeros | zeros | BF | vs T27a |
| T29b | zeros | DAPI | BF | vs T28 |
| T29c | Actn2 | DAPI | BF | vs T28, T29b |

## 6. Execution Plan

- L4 × 3 实验 × 2 seeds = 6 runs
- Seed=42 全部完成, seed=123 正在 L4 运行

## 7. Expected Risks

- 官方编码理论上更匹配预训练 encoder, 但实际差异可能小

## 8. Results

### Seed=42 (L4, completed)

| 实验 | R | G | B | Val PQ | Val Dice | Runtime |
|------|---|---|---|:------:|:--------:|:-------:|
| T29a | 0 | 0 | BF | 0.6422 | 0.7947 | 4h01m |
| T29b | 0 | DAPI | BF | 0.6648 | 0.8051 | 8h18m |
| T29c | Actn2 | DAPI | BF | 0.6849 | 0.8195 | 5h16m |

### Seed=123 (L4, completed)

| 实验 | Val PQ | Val Dice |
|------|:------:|:--------:|
| T29a s123 | 0.6411 | 0.7948 |
| T29b s123 | 0.6548 | 0.8020 |
| T29c s123 | 0.6783 | 0.8127 |

### Dual-Seed Average

| 实验 | R | G | B | Mean PQ | Mean Dice |
|------|---|---|---|:-------:|:---------:|
| T29a | 0 | 0 | BF | **0.642** | 0.795 |
| T29b | 0 | DAPI | BF | **0.660** | 0.804 |
| T29c | Actn2 | DAPI | BF | **0.682** | 0.816 |

## 9. Interpretation

1. **BF-only**: [0,0,BF] PQ=0.642 ≈ [BF,BF,BF] PQ=0.643 → 无显著差异 (-0.1pp)
2. **3ch 官方**: [0,DAPI,BF] PQ=0.660 < [BF,Actn2,DAPI] PQ=0.686 → 旧编码更高 (-2.6pp)
3. **Actn2 贡献**: [Actn2,DAPI,BF] PQ=0.682 追平 T28 → **Actn2 提供 +2.2pp** (双 seed 确认)
4. **双 seed 一致性良好**: 所有子实验 std < 0.6pp

## 10. Decision
- 通道编码选择: Actn2+DAPI+BF (T29c) 与旧编码 T28 效果持平, 而官方 2ch (T29b) 略低
- **BF-only 不受编码方式影响** (T29a ≈ T27a)
- 最终编码选择取决于论文表述需求
