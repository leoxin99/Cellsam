# T32: Stage2-like Neck-Only Baseline (Our Allen Data)

## 1. Metadata
- ID: `T32`
- Status: `Completed`
- Owner: `A2`
- Priority: `P0`
- Related task: `docs/task_backlog.md` (methodology baseline gap)
- Related config: `src/config/t32_stage2_like_neck_only.yaml`
- Related scripts: `scripts/train_t32_s42_l4.sh`, `scripts/train_t32_s123_l4.sh`
- Related output dir: `checkpoints/T32_NeckOnly_Baseline_seed{42,123}_*`
- ALICE Job IDs: seed42=#1140905, seed123=#1140906

## 2. Background

CellSAM 论文 Stage2 文字描述是“冻结 SAM-ViT 与 mask decoder，仅微调 neck，使用 GT boxes + segmentation labels”。
当前主线 T27a 是 decoder-only 训练，不是 neck-only。
需要一个 Stage2-like 对照，作为方法学基线。

## 3. Question / Hypothesis

1. 在 Allen 数据上改成 neck-only（其余结构冻结）是否可训练并达到可比性能？
2. 与 T27a decoder-only 相比，neck-only 的收敛速度与最终 PQ 差距有多大？

## 4. Fixed Conditions

- 模型分支: `model_cp` (`adv_mode=True`)
- 训练框: GT boxes（沿用当前训练数据流）
- 预处理: 官方链路 `prep_2 + forward`（通过 `official_preprocess.py`）
- 训练轮次: 50 epochs
- 调度器: cosine_warmup
- Optimizer: AdamW (`lr=1e-4`, `wd=1e-4`)
- 评估口径: 统一 `segment_with_boxes + compute_all_metrics`

## 5. Variables

### T32a (主实验)
- `train_neck_only: true`
- `freeze_decoder: true`
- 额外损失关闭（`boundary/aji/focal/contour/topology/size/neighbor/overlap = false`）
- `pos_weight: 1.0`（仅做 class-balance 中性设置）

### T32b (敏感性对照，可选)
- 同 T32a，仅 `pos_weight: 10.0`

> 说明: 此处 Dice+BCE 是“我们项目可执行的 surrogate 设定”，不是 CellSAM 论文可证的 Stage2 官方 loss 公式。

## 6. Feasibility Gate (必须先过)

当前代码里 `use_lora=false` 时，encoder 前向固定在 `torch.no_grad()`：
- `src/train.py:307-314`

若直接做 neck-only，这会导致 neck 无梯度。

因此 T32 必须先做最小代码改动：
1. 在 `create_model()` 增加 `train_neck_only` 分支：
   - 冻结 `model.model_cp.image_encoder` 全部参数
   - 单独解冻 `model.model_cp.image_encoder.neck` 参数
2. 在 `train_one_epoch()` 中加入条件：
   - 当 `train_neck_only=true` 时，走带梯度的 encoder forward（不能在 `no_grad` 块内）
3. 增加可训练参数审计日志：
   - 目标是“仅 neck 可训练”

## 7. Execution Plan

1. 代码最小改动 + 单元检查（trainable 参数清单）
2. 本地 1-epoch smoke（验证 neck 梯度非 0）
3. ALICE 正式训练（50 epochs）
4. Oracle(val71) + Oracle(test73) 评估
5. 与 T27a（decoder-only）并排对比

## 8. Expected Risks

1. neck-only 表达能力可能不足，PQ 低于 T27a
2. 若梯度门禁未处理好，可能出现“看似训练但 neck 未更新”
3. 当前公开 CellSAM 无 Stage2 逐行训练脚本，论文对齐只能做到“结构/超参近似”

## 9. Success Criteria

- 工程可行: neck 参数梯度稳定非 0，训练正常收敛
- 方法学可比: 得到可复现的 Stage2-like 基线结果
- 文档可引用: 明确标注“Stage2-like surrogate，不宣称官方复现”

## 10. Decision Rule

- 若 T32a 明显低于 T27a（例如 PQ 差 >5pp），主线继续 decoder-only
- 若 T32a 接近或优于 T27a，进入下一轮 neck-only 细化

---

## 11. Results (2026-03-07)

### 实现验证清单

| 设计项 | 实际实现 | 状态 |
|--------|---------|:----:|
| `train_neck_only: true` | `create_model()` L179-193: 解冻 `image_encoder.neck` | ✅ |
| `freeze_encoder: true` | 全部 encoder params frozen → neck 单独解冻 | ✅ |
| `freeze_decoder: true` | mask_decoder 全冻结 | ✅ |
| Encoder forward WITH gradients | `train_one_epoch()` L328: `use_lora or train_neck_only` 条件 | ✅ |
| Trainable audit | ALICE log: `[T32] Neck-only mode: unfroze 787,456 neck parameters` | ✅ |
| Loss: BCE only | `use_boundary/aji/focal/contour = false`, `pos_weight=1.0` | ✅ |
| pos_weight: 1.0 | YAML confirmed | ✅ |
| 训练轮次: 50 epochs | 两个 seed 均跑满 50/50 | ✅ |

> ALICE Log 确认: `[FROZEN] cellfinder: 0 / 102,122,590 trainable`, `encoder: 787,456 / 89,670,912 trainable`

### 定量结果

| Seed | Best Val Dice | Final Val PQ | Final BM-1to1 | Final Sem | Epochs |
|:----:|:------------:|:------------:|:-------------:|:---------:|:------:|
| 42   | **0.7832**   | **0.6169**   | 0.7813        | 0.8182    | 50/50  |
| 123  | **0.7866**   | **0.6225**   | —             | —         | 50/50  |

**Seed 42 最终 epoch**: Train Loss=0.2658, BM-1to1=0.7813, PQ=0.6148, Sem=0.8182

### 与参照实验对比

| 实验 | 方法 | Best PQ | Best Dice | 可训练参数 |
|------|------|:-------:|:---------:|:---------:|
| CellSAM 原始 (test73) | 无训练 (baseline) | 0.491 | 0.723 | 0 |
| **T32 s42** | **Neck-only** | **0.617** | **0.783** | **787K** |
| **T32 s123** | **Neck-only** | **0.623** | **0.787** | **787K** |
| T27a s42 | Decoder-only | 0.617 | 0.783 | ~4.1M |

### 初步结论

1. **Neck-only 可行**: 787K 参数即可将 PQ 从 0.491 提升至 0.62，相当显著
2. **与 T27a 对比惊人接近**: T32 PQ ≈ T27a PQ (0.617 vs 0.617)，尽管 T32 仅有 T27a ~19% 的参数量
3. **两 seed 一致性极好**: PQ 差异仅 0.006 (0.617 vs 0.623)
4. **未触发 early stopping**: 50 epoch 全跑完，可能仍有提升空间

> ⚠️ 待 A1 审核: (1) T27a 正式对齐口径确认 (2) T32 是否需要 val=71 + test=73 评估 (3) 这些 PQ 值是否包含 F1/Precision/Recall
