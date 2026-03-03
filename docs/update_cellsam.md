# CellSAM 更新总览 (Q&A 汇总 + 技术分析)

> **更新**: 2026-02-24  
> **整理人**: A1 (Codex) 原始调研 + R1 (Reviewer) 审核分析  
> **目的**: 汇总 CellSAM 论文/源码/本项目实现的关键问答、R1 审核结论、以及技术决策依据

---

## 目录

- **Part I: CellSAM 论文事实** — §1 ~ §3 (A1 整理)
- **Part II: 我们的方案 vs CellSAM** — §4 ~ §6
- **Part III: 关键技术分析** — §7 ~ §11 (R1 审核)
- **Part IV: 实验规划** — §12 ~ §13
- **Appendix** — 参考索引 / 证据口径 / 角色分工

---

# Part I: CellSAM 论文事实

> 以下内容由 A1 (Codex) 整理，基于 CellSAM Nature 论文 + 公开源码快照

## 1. 先说结论

1. CellSAM 论文是"两阶段训练"：Stage1 重点训练检测 (CellFinder)，Stage2 重点修正分割侧特征对齐。
2. 公开 `cellSAM_source` 快照以推理为主，**缺完整 Stage2 训练脚本**，因此部分 loss 公式和权重无法代码级逐行复现。
3. 我们当前训练路径与论文 Stage2 不同：当前是"冻结 image encoder、训练 prompt encoder + mask decoder (和可选 adapter)"。
4. SAM/CellSAM 的 box 是提示 (prompt)，不是硬裁剪；若不加 clipping，模型可在框外给出高响应。

## 2. CellSAM 的结构与训练

### 2.1 模型结构 (论文口径)

CellSAM = "共享编码器 + 检测分支 + 分割分支"：

1. 共享 SAM ViT 编码器 (论文中作为共享 backbone)
2. 检测分支 CellFinder (AnchorDETR 风格) 输出候选框
3. 分割分支 SAM (prompt encoder + mask decoder) 根据框生成实例 mask
4. neck 负责将 ViT 特征映射到分割端可用的表示空间 (论文描述为 2D conv 的适配层)

### 2.2 Stage1 / Stage2

1. **Stage1 (检测主导)**
   - 训练 CellFinder (并联合共享编码器) 学习从图像直接产出细胞框
   - 论文给出较大训练预算 (包含 2800 epochs)

2. **Stage2 (分割对齐)**
   - 修复 Stage1 后 ViT↔Decoder 分布不匹配问题
   - 论文文本描述为冻结 ViT encoder 与 mask decoder，仅训练 neck 做特征对齐；但公开仓库缺 Stage2 训练脚本，无法代码级逐行复验。

### 2.3 "解冻 mask decoder 最后一层" (已澄清)

此前讨论中出现过"解冻 mask decoder 最后一层再训一轮"的说法。基于当前可核查的论文 + 公开仓库快照，该说法**没有可复现证据链**。后续应以"论文明确项" vs "源码可证项"分开标注。

## 3. CellSAM 的 Loss

### 3.1 检测侧 (CellFinder) loss：公开代码可证

在 `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py` 可见：

1. `loss_ce` (分类项，focal 形式, L191)
2. `loss_bbox` (L1 框回归, L223)
3. `loss_giou` (GIoU, L231)

### 3.2 分割侧 (SAM 分支) loss：论文有描述，公开代码不完整

1. 论文描述了 Stage2 的 prompt-conditioned 分割训练思想
2. 公开快照中找不到完整 Stage2 训练脚本与可复现的完整 loss 权重
3. 结论：分割侧确实受监督训练 (论文事实)，但具体损失公式/权重不可完全复刻 (代码事实)
4. 写作约束：除非拿到作者补充训练代码/附录公式证据，不应把 CellSAM Stage2 loss 写死为 "Dice+BCE"。

---

# Part II: 我们的方案 vs CellSAM

## 4. 检测方案对比

| | CellSAM 原方案 | 我们的方案 |
|---|---|---|
| **检测器** | CellFinder (Anchor-DETR) | DAPI 核检测 + Adaptive 框扩展 |
| **优势** | 端到端，无需手工特征 | 可控、可解释、参数已优化 (E34b) |
| **劣势** | 未适配心肌细胞 (E2E PQ=0.180) | 依赖 DAPI 信号质量 |

原因：心肌细胞域上 CellFinder 早期效果不理想 (PQ=0.180)，DAPI 路径可控且可解释性更高。

## 5. 训练策略对比

### 5.1 当前策略

- `freeze_encoder: true` — ViT encoder 完全冻结
- `freeze_decoder: false` — mask decoder 可训练
- prompt encoder 默认未冻结 → 实际训练 prompt encoder + mask decoder (~4.06M 参数)

证据: `src/train.py:141-146`, `src/config/phase1_rebalance.yaml:17-18`

### 5.2 损失体系

`CombinedLoss` (`src/losses/combined.py:449`) 包含：

1. **基础**: Dice + BCE
2. **结构项**: Boundary / AJI / Topology / Size / ~~Contour~~ ⚠️ T12 消融证明 ContourLoss 有害 (+2.3pp PQ when removed, 与 SAM prompt-conditioned 范式冲突)
3. **粘连约束**: NeighborIntrusionLoss / OverlapMutexLoss

**Best Config (2026-02-24 验证)**: posw=10, contour=off, boundary=1.5, AJI=0.2 → PQ=**0.484** (4 runs mean)

### 5.3 IoU Head 解释 + 建议 (P2)

**SAM 的 IoU Head**: SAM mask decoder 输出两个东西：
1. **Mask logits** (H×W) — 分割预测
2. **IoU prediction** (标量) — 模型对自身输出质量的置信度估计

IoU Head 是 mask decoder 最后的全连接层 (`iou_prediction_head`)，预测当前 mask 与 GT 的 IoU 值：

```
mask_decoder 输出:
  ├── mask_tokens → 转置卷积 → mask logits (H×W)
  └── iou_token → MLP → iou_prediction (标量)
```

**当前状态**: 我们的 `CombinedLoss` **没有对 IoU prediction 做损失监督**。它只用了 mask logits 做 Dice+BCE。

**影响分析**:
- IoU prediction 在**推理时用于选择最优 mask** (SAM 输出 3 个 mask 候选，取 IoU 预测最高的)
- 我们只用 1 个 box prompt → 1 个 mask → IoU prediction **不参与选择**
- 因此 IoU Head 对当前 pipeline **无实际影响**

**建议**:
| 方案 | 优先级 | 说明 |
|------|:------:|------|
| 保持现状 (不训 IoU Head) | ✅ 推荐 | 单 prompt → 单 mask, IoU Head 无用 |
| 加 IoU supervision loss | P3 | 仅在多模板选择时有意义 |
| 用 IoU prediction 做 post-filtering | P3 | 可用于置信度阈值过滤 |

### 5.4 Focal Loss 对比 + 建议 (P2)

**Focal Loss** (Lin et al. 2017): 解决类别不平衡的经典方案

$$FL(p) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

- $\gamma$ (focusing parameter): 降低 easy example 的 loss 权重
- 当 $\gamma=0$ 时退化为标准 CE/BCE

**与我们的当前方案对比**:

| 方案 | 机制 | 我们的实现 |
|------|------|-----------|
| **pos_weight (当前)** | 正类整体权重 ×10 | `nn.BCEWithLogitsLoss(pos_weight=10)` |
| **Focal Loss** | 按 sample 难度动态加权 | ❌ 未实现 |
| **Dice Loss (当前)** | 区域重叠, 天然处理不平衡 | ✅ `DiceLoss` |

**分析**:
- `pos_weight=10` 是**全局固定**权重，不区分 easy/hard 像素
- Focal Loss 是**逐像素自适应**权重: hard pixel (边界/小目标) 获得更高权重
- T12 消融显示 `pos_weight` 从 2→10 带来 **+4.1pp PQ** — 说明前景欠重是主要瓶颈
- Focal Loss 可能进一步改善边界像素的学习

**建议**:
| 方案 | 优先级 | 预期 |
|------|:------:|------|
| Focal Loss 替换 BCE | **P2** | 边界像素学习可能改善 |
| pos_weight + Focal 组合 | P3 | 可能 over-weighting |
| 保持现状 | ✅ 安全 | pos_weight=10 已经足够好 |

> 📎 Focal Loss 实验应在 Best Config 基础上做单因素消融 (类似 T12), 预设为 γ=2, α=0.25。优先级低于 T18/T20。

## 6. 常见误解澄清

1. **"CellSAM 只会在框内分割"** — 不严谨。box 是 prompt 不是硬边界。详见 §10
2. **"我们只训练 mask decoder"** — 不准确。prompt encoder 也参与更新 (默认未冻结)
3. **"论文 Stage2 与我们当前训练完全一致"** — 不是。我们是"冻结 encoder、训 prompt+decoder"，不是"只训 neck"。详见 §8
4. **"ContourLoss 有效"** — T12 消融已证明 ContourLoss 有害 (移除后 +2.3pp PQ)

---

# Part III: 关键技术分析

> 以下内容由 R1 (Reviewer) 审核整理 (2026-02-24)  
> 依据: SAM / CellSAM / MedSAM 论文 + 最新文献 (SAMed, FSAM, S-SAM, Conv-LoRA) + 代码实测

## 7. SAM 参数量与训练策略

### 7.1 各组件参数量 (实测)

| 组件 | 参数量 | 占比 |
|------|:------:|:----:|
| Image Encoder (ViT-B) | 89,670,912 | 95.7% |
| ┗ 其中 Neck (2 层 Conv) | 787,456 | 0.84% |
| Prompt Encoder | 6,220 | ~0% |
| Mask Decoder | 4,058,340 | 4.3% |
| **Total** | **93,735,472** | 100% |

Neck 结构: `Conv2d(768→256, 1×1)` + `LayerNorm` + `Conv2d(256→256, 3×3)` + `LayerNorm`

### 7.2 五种训练策略对比

| 策略 | 训练参数 | 论文先例 | 适用场景 |
|------|:--------:|---------|---------|
| **① 当前 (prompt+decoder)** | **4.06M (4.3%)** | MedSAM 近似 | ✅ **当前方案** |
| ② Neck-only | 787K (0.84%) | CellSAM Stage2 | 特征对齐 (⛔ 不适用) |
| ③ Decoder-only | 4.06M (4.3%) | MedSAM (1M+ data) | 大数据微调 |
| ④ LoRA encoder + decoder | ~4.5M | SAMed, Conv-LoRA | **小数据最优** |
| ⑤ Full fine-tune | 93.7M (100%) | FSAM | 大数据+大计算 |

## 8. Neck-Only 评估: ⛔ 不推荐

> **A1 原始提案** (原 §7): 只解冻 `image_encoder.neck`，冻结 prompt_encoder + mask_decoder。  
> **R1 审核结论**: ❌ 不推荐作为主力方案。

**CellSAM Stage2 使用 neck-only 的原因:**

CellSAM Stage1 联合训练 CellFinder + ViT encoder → **改变了 ViT 特征分布** → Stage2 用 neck 做"特征空间重新对齐" (freeze ViT + decoder，只训 neck)。

**我们的情况不同:**

- 我们**从未动过 ViT encoder** (一直冻结) → **不存在分布偏移**
- CellSAM 预训练时的 neck 已处于对齐状态 → 再训 = 引入扰动
- 787K 参数太少，不足以学习心肌细胞的域特异性

**结论**: Neck-only 的动机源于 CellSAM 特殊的两阶段训练流程，不适用于我们的"冻结 encoder + 训 decoder"路线。可作论文对照实验 (预期效果差)。

<details>
<summary>📋 A1 原始 neck-only 改造清单 (已否决，保留供参考)</summary>

**配置新增** (`src/config/phase2_stage2_neck_only.yaml`):
- `model.train_neck_only: true`
- `model.freeze_prompt_encoder: true`
- `model.freeze_decoder: true`
- 保留 `model.freeze_encoder: true` (代码里对 neck 单独放开)

**`src/train.py` 改造点**:
1. `create_model()` (L125): 读取 `train_neck_only`，冻结全部只留 neck
2. `train_one_epoch()` (L201): 新增 `train_neck_only` 参数，修改 `torch.no_grad()` 路径让 neck 有梯度
3. `main()` (L592): 传递 `train_neck_only` 配置
4. 可训练参数审计: 确认只含 `image_encoder.neck.*`

**验证协议**: 1 epoch smoke → 5 epoch 小样本 → 全量 50 epoch
</details>

## 9. Encoder 微调: ✅ 建议用 LoRA

### 9.1 文献共识 (2024-2025)

| 来源 | 结论 |
|------|------|
| MedSAM (Nature, 2024) | Decoder-only + 1M+ 数据可行 |
| FSAM (IEEE, 2024) | Encoder+Decoder 微调 > Decoder-only |
| SAMed (ICLR, 2024) | LoRA on encoder + full decoder → 小数据最优 |
| S-SAM (MICCAI, 2024) | SVD tuning encoder (0.4% params) 即可超越 decoder-only |
| Conv-LoRA (arxiv, 2024) | LoRA + Conv 注入 → 补充 ViT 缺失的空间感应偏置 |

### 9.2 为什么适合我们

**我们是小数据场景**: 334 张训练图 × ~10 cell/image ≈ 3300 样本

- Decoder-only 在小数据上天花板明显 (MedSAM 靠 1M+ 数据才能成功)
- LoRA 微调 encoder 是文献验证的小数据最佳选择 (~0.5% 参数即可让 ViT 学到域特异特征)
- 当前 PQ=0.484 (Best Config) vs MedSAM PQ=0.576 的差距 (~9pp)，可能源于 encoder 特征不够适应心肌细胞

## 10. 框外分割策略与 Box Clipping ★

### 10.1 SAM 的分割机制

SAM 的 box prompt **不是硬边界**，而是"注意力引导":

```
分割流程:
1. Image Encoder: 整张图 → 64×64×256 特征 (与 box 无关)
2. Prompt Encoder: box [x1,y1,x2,y2] → prompt token
3. Mask Decoder: 特征 + prompt → 整图 sigmoid mask (256×256)
   → 理论上 box 外的像素也可能获得高概率值
```

**关键**: Decoder 输出覆盖**整张图**。如果模型学到了"细胞从 box 内延伸到 box 外"，它完全可以在框外预测高概率。

### 10.2 Box Clipping 的工作原理

```
无 Clipping (理论上更自由):          有 Clipping (当前方案):
  ┌──────────────┐                    ┌──────────────┐
  │ image        │                    │ image        │
  │   ┌──────┐   │                    │  ┌────────┐  │
  │ ██│██████│██ │ ← 泄漏到框外      │  │████████│  │ ← 框外强制清零
  │ ██│██████│██ │   与邻居冲突       │  │████████│  │   不会与邻居冲突
  │   └──────┘   │                    │  └────────┘  │
  └──────────────┘                    └──────────────┘
```

**T19-abl 实验结果**: with_clip PQ=**0.466** > no_clip PQ=0.437 (-6.2%)

### 10.3 为什么去掉 Clipping 反而变差？

根本原因: **我们的模型没学会"框外抑制"**

MedSAM (百万级数据训练) 能自然地在 box 外给低概率。但我们只有 334 张训练图，模型没学好这个约束 → box 外有大量泄漏 → 与邻居细胞冲突 → PQ 下降。

```
MedSAM:   box 内 ████(高)  box 外 ░░░░(低)  → 不需要 clipping
我们:     box 内 ████(高)  box 外 ██░░(中高) → 需要 clipping 兜底
```

### 10.4 心肌细胞的特殊问题

1. DAPI 框基于核 (核是圆的) → 扩展后的框偏正方形
2. 心肌细胞是**长条形** → 框不够长，框住不了整个细胞
3. Box clipping 把框外的**正确预测**也截掉了 → mask 被截成框的形状 (偏圆)

**问题链**: DAPI 框不准 → clipping 截断 → mask 偏圆/偏小 → PQ 下降

### 10.5 优化方向

| 方案 | 难度 | 预期效果 | 优先级 |
|------|:----:|---------|:------:|
| **改善框质量** (Adaptive 检测优化) | 中 | 框更贴合长条细胞 → 截断更少 | **P0** |
| **LoRA 微调 encoder** | 中 | 模型学到心肌特征 → 减少泄漏 → 降低对 clipping 的依赖 | **P1** |
| **Soft clipping** (高斯衰减) | 低 | 框外不完全清零，保留部分框外预测 | P2 |
| **微调 CellFinder** | 高 | 端到端检测 → 框质量理论最优 | P3 |

> **最有希望的路径是 LoRA**: 如果 encoder 学到心肌细胞的长条形特征，即使框不完美，decoder 也能根据图像特征推断出框外的正确形状 — 此时框外泄漏变成"框外正确预测"，去掉 clipping 反而可能提升 PQ。

## 11. CellFinder 微调可行性 ★

### 11.1 CellFinder 架构

CellFinder = **Anchor-DETR** (transformer-based object detector):

| 属性 | 值 | 来源 |
|------|-----|------|
| 架构 | Anchor-DETR (6 enc + 6 dec layers) | `sam_inference.py:82-88` |
| Hidden dim | 256 | `modelconfig.yaml` |
| Query positions | 3500 (max detections/image) | `num_query_position=3500` |
| Classes | 2 (background + cell) | `num_classes=2` |
| Backbone | SAM ViT-B (shared, 768-dim) | `in_channels=768` |
| Attention | RCDA (Row-Column Decomposed) | `attention_type="RCDA"` |

### 11.2 "CellSAM 未公开训练脚本"是什么意思？

CellSAM GitHub 仓库只包含**推理代码** (用已训练好的模型做预测)。训练 CellFinder 的脚本 (Stage1 如何用 GT mask 生成 bbox → 如何训练 Anchor-DETR) **没有公开**。因此要微调 CellFinder 需**从零构建训练管道**。

### 11.3 LoRA 微调 encoder ≠ 微调 CellFinder

```
CellSAM 架构:
┌──────────────────────────────────────────────────┐
│              ViT Image Encoder (共享)              │
│                (89.7M params)                      │
├───────────────────┬──────────────────────────────┤
│   CellFinder       │    SAM Mask Decoder           │
│ (Anchor-DETR)      │  (Prompt Enc + Decoder)       │
│  输出: bbox        │   输出: segmentation mask     │
│  ~10M params       │   ~4M params                  │
└───────────────────┴──────────────────────────────┘
     检测模块                   分割模块
  (我们没在用)              (我们在训练的部分)
```

- **LoRA 微调 encoder** → 让 encoder 提取更适合心肌细胞的特征 → **提升分割质量**
- **微调 CellFinder** → 让 CellFinder 检测心肌细胞框 → **替代 DAPI 检测** (完全不同的目标)

### 11.4 微调步骤 (若决定执行)

1. **数据转换**: GT instance masks → COCO JSON (bbox + category) — 工具脚本 ~50 行
2. **构建训练 DataLoader**: 参考 Anchor-DETR 原始训练代码，适配 CellSAM 预处理
3. **训练策略**: 冻结 ViT encoder，只训 CellFinder decode_head
4. **Loss**: Anchor-DETR 原始 = Hungarian matching + L1 + GIoU + CE
5. **评估**: 对比 DAPI 检测 vs CellFinder 检测的 box AP / F1

**可行性**: ⚠️ 可行但**风险高、优先级低** (P3)

| 方面 | 评估 |
|------|------|
| 数据 | 334 train + 5173 cells → 标注足够 |
| 训练管道 | 需从零构建 (~1-2 周) |
| ViT 共享 | 可能需联合训练 ViT → 破坏冻结策略 |
| vs DAPI | DAPI 已可用且 E34b 已优化，收益重叠 |

---

# Part IV: 实验规划

## 12. 当前实验矩阵

> 更新于 2026-02-24，反映 Best Config + T18 三通道进展

| 优先级 | 实验 | 训练参数 | 状态 |
|:------:|------|:--------:|:----:|
| **P0** | Best Config (posw=10, contour=off) | 4.06M | ✅ PQ=0.484 |
| **P0** | T18: 三通道消融 (2ch/3ch/无adapter) | 4.06M+adapter | 🔄 训练中 |
| **P1** | LoRA encoder (r=4) + Best Config | ~4.5M | ⏳ 待 T18 后 |
| P2 | LoRA encoder only (freeze decoder) | ~0.5M | ⏳ 隔离 encoder 贡献 |
| P3 | Neck-only (论文对照) | 787K | ⏳ 预期效果差 |
| P3 | CellFinder 微调 | ~10M | ⏳ 高风险，待裁决 |

评估口径: 固定数据划分 (334/71/73)，Oracle (test) + E2E (test) 分开报告，避免"训练/推理口径混淆"

## 13. 论文叙事价值

**如果 LoRA encoder 跑出比 Best Config 更好的结果:**
- *"Adding LoRA to the frozen encoder allows domain-specific feature adaptation with minimal parameter overhead, bridging the gap with MedSAM's large-scale pretraining"*

**如果效果不明显:**
- 说明 decoder 微调 + loss 工程已足够，encoder 特征足够通用 — 也是有价值的负结论

**如果三通道 > BF-only:**
- *"Incorporating DAPI and α-Actinin channels through semantic channel mapping provides complementary structural cues, improving segmentation of cardiomyocyte boundaries"*

---

# Appendix

## A. 参考索引 (代码位置)

1. 训练入口: `src/train.py`
2. 统一推理: `src/inference/core.py`
3. 损失实现: `src/losses/combined.py`
4. 数据集: `src/augmented_dataset.py`
5. CellSAM 推理: `cellSAM_source/cellSAM/sam_inference.py`
6. CellFinder loss: `cellSAM_source/cellSAM/AnchorDETR/models/anchor_detr.py`

## B. 证据口径说明

1. "论文事实"与"公开代码可证事实"已分开标注
2. 对公开快照缺失内容 (尤其 Stage2 完整训练细节) 保持不确定性标注
3. 后续如拿到作者私有训练脚本，可追加"代码级补证"章节，不覆盖历史结论

## C. 实现角色分工

| 任务 | 推荐角色 | 理由 |
|------|---------|------|
| LoRA 代码实现 (train.py) | **A2 (Claude)** | 需修改训练核心代码，A2 熟悉 train.py |
| Neck-only 代码实现 | **A1 (Codex)** | A1 已写好方案 |
| CellFinder 数据转换 (GT→COCO) | **A1** | 纯数据处理 |
| CellFinder 训练 loop | **A2** | 需深度理解检测框架 |
| 实验配置 + SLURM | **A2** | A2 有 ALICE 经验 |
| 结果分析 + 论文 | **R1** | R1 做消融审核 |

> **时序**: Best Config ✅ → T18 三通道 (训练中) → LoRA (P1) → CellFinder (P3, 待裁决)
