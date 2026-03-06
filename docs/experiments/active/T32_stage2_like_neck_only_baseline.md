# T32: Stage2-like Neck-Only Baseline (Our Allen Data)

## 1. Metadata
- ID: `T32`
- Status: `Planned`
- Owner: `A2`
- Priority: `P0`
- Related task: `docs/task_backlog.md` (methodology baseline gap)
- Related config: `src/config/t32_stage2_like_neck_only.yaml` (to create)
- Related script: `scripts/train_t32_stage2_like.sh` (to create)
- Related output dir: `checkpoints/T32_Stage2Like_NeckOnly_*`

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
