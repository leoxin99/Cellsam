# CellSAM 更新总览（Q&A 汇总 + 对齐方案）
> 更新时间: 2026-02-23  
> 整理人: A1 (Codex)  
> 目的: 汇总此前关于 CellSAM 论文/源码/本项目实现的关键问答，并给出“更贴近论文 Stage2（只训 neck）”的最小改造清单。

---

## 1. 先说结论（避免混淆）

1. CellSAM 论文是“两阶段训练”：Stage1 重点训练检测（CellFinder），Stage2 重点修正分割侧特征对齐。  
2. 公开 `cellSAM_source` 快照以推理为主，**缺完整 stage-2 训练脚本**，因此部分 loss 公式和权重无法代码级逐行复现。  
3. 你们当前项目训练路径与论文 Stage2 不同：当前是“冻结 image encoder、训练 prompt encoder + mask decoder（和可选 adapter）”。  
4. SAM/CellSAM 的 box 是提示（prompt），不是硬裁剪；若不加 clipping，模型可在框外给出高响应。  
5. 若要“贴近论文 Stage2（只训 neck）”，核心是三点：  
   - 只解冻 `image_encoder.neck`  
   - 冻结 `prompt_encoder` 与 `mask_decoder`  
   - 训练时不要用 `torch.no_grad()` 包住 image encoder 前向（否则 neck 无梯度）

---

## 2. CellSAM 论文中的结构与训练

## 2.1 模型结构（论文口径）

CellSAM 可以理解为“共享编码器 + 检测分支 + 分割分支”：

1. 共享 SAM ViT 编码器（论文中作为共享 backbone）  
2. 检测分支 CellFinder（AnchorDETR 风格）输出候选框  
3. 分割分支 SAM（prompt encoder + mask decoder）根据框生成实例 mask  
4. neck 负责将 ViT 特征映射到分割端可用的表示空间（论文描述为 2D conv 的适配层）

## 2.2 论文 Stage1 / Stage2

1. Stage1（检测主导）  
   - 训练 CellFinder（并联合共享编码器）学习从图像直接产出细胞框。  
   - 论文给出较大训练预算（包含 2800 epochs 的训练设置）。  

2. Stage2（分割对齐）  
   - 论文描述重点是修复 Stage1 后分布不匹配问题。  
   - 口径为冻结主干大部分模块，仅微调分割侧关键适配部分（论文强调 neck）。  

## 2.3 “解冻 mask decoder 最后一层”说明（更正）

此前讨论中出现过“解冻 mask decoder 最后一层再训一轮”的说法。  
基于当前可核查的 Nature 正式论文 + 公开仓库快照，该说法**没有可复现证据链**。  
本仓库后续文档应以“论文明确项”和“源码可证项”分开标注，避免把推测写成事实。

---

## 3. CellSAM 的 loss：哪里确定、哪里不确定

## 3.1 检测侧（CellFinder）loss：公开代码可证

在 `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py` 可见：

1. `loss_ce`（分类项，focal 形式实现）  
2. `loss_bbox`（L1 框回归）  
3. `loss_giou`（GIoU）

对应位置：

- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:191`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:223`
- `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py:231`

## 3.2 分割侧（SAM 分支）loss：论文有描述，公开代码不完整

1. 论文描述了 Stage2 的 prompt-conditioned 分割训练思想。  
2. 但在当前公开快照中，找不到完整 stage-2 训练脚本与可直接复现的完整 loss 组合权重。  
3. 因此可写结论是：  
   - 分割侧确实受监督训练（论文事实）  
   - 具体损失公式/权重在公开快照里不可完全复刻（代码事实）

---

## 4. SAM / CellSAM 的 prompt、框外预测与冲突像素

## 4.1 prompt 是什么

原始 SAM 即“promptable segmentation”：  
prompt 可为 points / boxes / mask 等，经 prompt encoder 编码后与图像特征一起送入 mask decoder。

## 4.2 为什么会有框外预测

在 CellSAM 推理实现中，box 作为 prompt 输入 decoder，但输出 mask 上采样到整图；若不强制裁剪，框外仍可能保留高概率像素。  
你们项目里已经显式用 clipping 约束该行为（训练与推理都有）。

## 4.3 冲突像素归属（多实例重叠）

你们统一推理核心 `src/inference/core.py` 当前支持三种裁决策略：

1. `argmax_prob`（默认）  
2. `first_write`  
3. `last_write`

对应：
- `src/inference/core.py:29`
- `src/inference/core.py:258`

这属于“规则裁决”，不是全局最优化分配。

---

## 5. 你们项目方案 vs CellSAM 原方案

## 5.1 检测方案

1. CellSAM 原链路：CellFinder 自动出框。  
2. 你们链路：DAPI / Adaptive（Z-line）生成框，替代 CellFinder。  

原因：心肌细胞域上 CellFinder 早期效果不理想，DAPI 路径可控且可解释性更高。

## 5.2 训练策略

1. 论文 Stage2 思路：对齐分割特征（强调 neck）。  
2. 你们当前主线：  
   - `freeze_encoder: true`  
   - `freeze_decoder: false`  
   - 未冻结 prompt encoder  
   - 因此实际在训 prompt encoder + mask decoder（encoder 不训）

证据：
- `src/config/phase1_rebalance.yaml:17`
- `src/config/phase1_rebalance.yaml:18`
- `src/train.py:141`
- `src/train.py:146`
- `src/train.py:272`
- `src/train.py:275`

## 5.3 损失体系

你们项目 `CombinedLoss` 已扩展为：

1. 基础：Dice + BCE  
2. 结构项：Boundary / AJI / Topology / Size / Contour  
3. 粘连约束：NeighborIntrusionLoss / OverlapMutexLoss  

对应：
- `src/losses/combined.py:449`（`CombinedLoss`）
- `src/losses/combined.py:398`（Neighbor）
- `src/losses/combined.py:414`（Overlap）

---

## 6. 常见误解澄清（本轮统一）

1. “CellSAM 只会在框内分割”  
   - 不严谨。box 是 prompt，不是硬边界；不做 clipping 时可有框外响应。

2. “你们只训练 mask decoder”  
   - 当前实现不是。prompt encoder 也参与更新（默认未冻结）。

3. “论文 Stage2 与你们当前训练完全一致”  
   - 不是。当前更接近“冻结 encoder、训练 prompt+decoder”的工程策略，而非“只训 neck”。

4. “ContourLoss 之前一直有效”  
   - 历史上有过不可微实现问题，后续已重写为纯 PyTorch 可微版本（当前代码为可微实现）。

---

## 7. 如何改成“更贴近论文 Stage2（只训 neck）”

下面是最小改动清单，按“先配置、后代码、再验证”执行。

## 7.1 配置新增（建议）

在目标配置（建议新建 `src/config/phase2_stage2_neck_only.yaml`）加入：

1. `model.train_neck_only: true`  
2. `model.freeze_prompt_encoder: true`  
3. `model.freeze_decoder: true`  
4. 保留 `model.freeze_encoder: true`（但代码里会对 neck 单独放开）

## 7.2 `src/train.py` 改造点（精确到函数）

1. `create_model()`（`src/train.py:125`）  
   - 读取 `train_neck_only`。  
   - 若为 true：  
     - 冻结 `model.model.image_encoder` 全参数  
     - 仅解冻 `model.model.image_encoder.neck`  
     - 冻结 `model.model.prompt_encoder`  
     - 冻结 `model.model.mask_decoder`

2. `train_one_epoch()`（`src/train.py:201`）  
   - 新增参数 `train_neck_only: bool = False`。  
   - 当前 image encoder 前向在 `torch.no_grad()` 内（`src/train.py:229`），会阻断 neck 梯度。  
   - 需改为：当 `train_neck_only=true` 时，image encoder 前向走有梯度路径；否则维持现有 no_grad。

3. `main()` 训练调用（`src/train.py:592`）  
   - 调用 `train_one_epoch(..., train_neck_only=config['model'].get('train_neck_only', False))`

4. 可训练参数审计（建议加在 `create_optimizer()`）  
   - 打印 trainable 参数名，确认只含 `image_encoder.neck.*`。  
   - 若出现 `prompt_encoder.*` 或 `mask_decoder.*`，直接 warning/raise。

## 7.3 预期行为变化

1. 参数更新量大幅降低，训练更稳定。  
2. 与论文 Stage2 口径更一致，实验叙事更强。  
3. 若指标下降，说明你们数据域可能更依赖 prompt/decoder 的任务特化，可据此再做对照实验。

## 7.4 最小验证协议

1. 跑 1 epoch smoke（确保梯度仅在 neck）  
2. 跑 5 epoch 小样本（看 PQ/BM-1to1 方向）  
3. 再决定是否全量 50 epoch

---

## 8. 建议后续实验矩阵（短版）

1. A 组：当前主线（prompt+decoder 训练）  
2. B 组：stage2-like（neck-only）  
3. C 组：stage2-like + 你们现有 neighbor/overlap 约束  

固定同一数据划分、同一评估口径（Oracle 与 E2E 分开报告），避免再次出现“训练口径/推理口径混淆”。

---

## 9. 参考索引（本地代码）

1. 训练入口：`src/train.py`  
2. 统一推理：`src/inference/core.py`  
3. 损失实现：`src/losses/combined.py`  
4. Phase2A 配置：`src/config/phase2a_neighbor_overlap.yaml`  
5. CellSAM 推理源码：`cellSAM_source/cellSAM/sam_inference.py`  
6. CellFinder loss 代码：`cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py`

---

## 10. 证据口径说明

1. “论文事实”与“公开代码可证事实”已分开写。  
2. 对公开快照缺失内容（尤其 stage-2 完整训练细节）保持不确定性标注。  
3. 后续如拿到作者私有训练脚本或补充材料原始实现，可在本文件追加"代码级补证"章节，不覆盖历史结论。

---

## 11. R1 审核: Neck-Only 方案评估 + Encoder 微调分析 (2026-02-24)

> **审核人**: R1 (Reviewer)
> **审核对象**: 本文档 §7 "只训 neck" 方案 + encoder 是否需要微调
> **依据**: SAM / CellSAM / MedSAM 论文 + 最新文献 (SAMed, FSAM, S-SAM, Conv-LoRA) + 代码实测参数计数

### 11.1 SAM 各组件参数量 (实测)

| 组件 | 参数量 | 占比 |
|------|:------:|:----:|
| Image Encoder (ViT-B) | 89,670,912 | 95.7% |
| ┗ 其中 Neck (2 层 Conv) | 787,456 | 0.84% |
| Prompt Encoder | 6,220 | ~0% |
| Mask Decoder | 4,058,340 | 4.3% |
| **Total** | **93,735,472** | 100% |

Neck 结构: `Conv2d(768→256, 1×1)` + `LayerNorm` + `Conv2d(256→256, 3×3)` + `LayerNorm`

### 11.2 训练策略对比

| 策略 | 训练参数 | 论文先例 | 适用场景 |
|------|:--------:|---------|---------|
| **① 当前 (prompt+decoder)** | **4.06M (4.3%)** | MedSAM 近似 | ✅ **当前方案** |
| ② Neck-only (本文 §7) | 787K (0.84%) | CellSAM Stage2 | 特征空间对齐 |
| ③ Decoder-only | 4.06M (4.3%) | MedSAM (1M+ data) | 大数据微调 |
| ④ LoRA encoder + decoder | ~0.5M+4M ≈ 4.5M | SAMed, Conv-LoRA | **小数据最优** |
| ⑤ Full fine-tune | 93.7M (100%) | FSAM | 大数据+大计算 |

### 11.3 Neck-Only 评估: ❌ 不推荐

**CellSAM Stage2 使用 neck-only 的原因:**

CellSAM 的 Stage1 联合训练 CellFinder + ViT encoder 做检测 → **改变了 ViT 的特征分布** → Stage2 用 neck 做"特征空间重新对齐"（freeze ViT + decoder，只训 neck）。

**我们的情况不同:**

- 我们**从未动过 ViT encoder**（一直冻结）
- CellSAM 预训练时的 neck 已处于对齐状态
- 再训 neck = 对已对齐的空间引入扰动，可能**适得其反**
- 787K 参数太少，不足以学习心肌细胞的域特异性

**结论**: Neck-only 的动机源于 CellSAM 特殊的两阶段训练流程，不适用于我们的"冻结 encoder + 训 decoder"路线。作为论文对照实验可以跑（预期效果差），但不应作为主力方案。

### 11.4 Encoder 是否需要微调: ✅ 建议用 LoRA

**文献共识 (2024-2025):**

| 来源 | 结论 |
|------|------|
| MedSAM (Nature, 2024) | Decoder-only + 1M+ 数据可行 |
| FSAM (IEEE, 2024) | Encoder+Decoder 微调 > Decoder-only |
| SAMed (ICLR, 2024) | LoRA on encoder + full decoder → 小数据最优 |
| S-SAM (MICCAI, 2024) | SVD tuning encoder (0.4% params) 即可超越 decoder-only |
| Conv-LoRA (arxiv, 2024) | LoRA + Conv 注入 → 补充 ViT 缺失的空间感应偏置 |

**我们是小数据场景**: 334 张训练图 × ~10 cell/image ≈ 3300 样本

- Decoder-only 在小数据上天花板明显（MedSAM 靠 1M+ 数据才能 decoder-only 成功）
- LoRA 微调 encoder 是文献验证的小数据最佳选择（~0.5% 参数即可让 ViT 学到域特异特征）
- 当前 PQ=0.494 (Best Config) vs MedSAM PQ=0.576 的差距 (~8pp)，可能源于 encoder 特征不够适应心肌细胞

### 11.5 建议实验矩阵

| 优先级 | 实验 | 训练参数 | 预期 |
|:------:|------|:--------:|------|
| **P0** | Best Config (posw=10, contour=off) | 4.06M | PQ ≈ 0.50 (正在跑) |
| **P1** | Best Config + LoRA encoder (r=4) | ~4.5M | PQ ≈ 0.52-0.55 |
| P2 | LoRA encoder only (freeze decoder) | ~0.5M | 隔离 encoder 贡献 |
| P3 | Neck-only (论文对照) | 787K | 预期效果差，但可作对比行 |

### 11.6 论文叙事价值

如果 LoRA encoder 跑出比 Best Config 更好的结果:
- *"Adding LoRA to the frozen encoder allows domain-specific feature adaptation with minimal parameter overhead, bridging the gap with MedSAM's large-scale pretraining"*

如果效果不明显:
- 说明 decoder 微调 + loss 工程已经足够，encoder 特征足够通用
- 也是有价值的负结论

### 11.7 实现建议

| 任务 | 推荐角色 | 理由 |
|------|---------|------|
| LoRA 代码实现 (train.py) | **A2 (Claude)** | 需修改训练核心代码，A2 熟悉 train.py |
| Neck-only 代码实现 | **A1 (Codex)** | A1 已写好方案 (§7) |
| 实验配置 + SLURM | **A2** | A2 有 ALICE 经验 |
| 结果分析 + 论文 | **R1** | R1 做消融审核 |

> **时序**: 先等 Best Config (P0) 完成并验证 → 再跑 LoRA (P1)
