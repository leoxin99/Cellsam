# T28: Plan B Three-Channel (Legacy Encoding)

## 1. Metadata
- ID: T28
- Status: ✅ Completed (both seeds)
- Owner: A2
- Priority: P0
- Related config: `src/config/t28_planb_3ch.yaml`
- Related script: `scripts/train_t28_l4.sh`
- Related output dir: `checkpoints/T28_PlanB_3ch_seed{42,123}_*`

## 2. Background

T27a 基础上引入三通道: BF + DAPI + Actn2。
使用 `SemanticChannelMapper` 映射为 [R=BF, G=Actn2, B=DAPI] (旧编码)。
与 T27a 配置完全一致, 仅改通道输入。

## 3. Question / Hypothesis

三通道 (BF+DAPI+Actn2) 是否优于 BF-only?

## 4. Fixed Conditions

- 与 T27a 完全相同 (encoder冻结, PE冻结, lr=1e-4, posw=10, etc.)
- checkpoint: null
- use_semantic_mapping: true

## 5. Variables

- Seed: 42, 123
- 通道: R=BF, G=Actn2, B=DAPI (旧编码, 与 CellSAM 官方不一致)

## 6. Execution Plan

- L4 × 2 seeds → 完成

## 7. Expected Risks

- 通道编码与 CellSAM 官方不一致, 可能影响 encoder 特征提取效率
- (后续 T29 消融证明: 影响不大)

## 8. Results

| Seed | Val PQ | Val Dice | Best Epoch | Runtime |
|:----:|:------:|:--------:|:----------:|:-------:|
| 42 | 0.6863 | 0.8185 | E36 | ~6h |
| 123 | 0.6810 | 0.8147 | — | 11h30m |
| **Mean** | **0.6837** | **0.8166** | | |

## 9. Interpretation

- 三通道显著优于 BF-only (+4.1pp PQ, T28 mean=0.684 vs T27a mean=0.643)
- DAPI + Actn2 提供了额外的细胞核/结构信息
- 与 T29 消融对比: 旧编码 [BF,Actn2,DAPI] ≈ 官方编码+Actn2 [Actn2,DAPI,BF]

## 10. Decision
- **Keep as current best 3ch result**
- 通道编码选择需结合 T29 消融结论
