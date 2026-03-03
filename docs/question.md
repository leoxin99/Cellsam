# CellSAM 项目技术问答汇总 (Question)

> 更新时间: 2026-02-25
> 范围: 项目中与训练/推理/评估/损失设计/模型架构相关的核心技术问题

---

## 10. CellSAM Image Encoder 如何微调？用什么 loss？(2026-02-24)

### Q1. CellSAM 公开源码有训练代码吗？

**❌ 没有。** `cellSAM_source/` 目录只有推理代码。GitHub 仓库 (vanvalenlab/cellSAM) 无 `train/` 目录。

| 模块 | 有 loss 代码？ | 说明 |
|------|:-----------:|------|
| `AnchorDETR/models/anchor_detr.py` | ✅ | CellFinder 检测 loss: focal CE + L1 + GIoU |
| `sam_inference.py` | ❌ | 纯推理: `predict()` 全在 `@torch.no_grad()` 下 |

### Q2. 论文怎么说的？

根据 Nature Methods 论文 + 补充材料 (通过 PMC 确认):

| 阶段 | 训练模块 | Loss | 冻结模块 |
|------|---------|------|---------|
| **Stage 1** | CellFinder + ViT Encoder | Focal CE + L1 + GIoU (检测 loss) | Mask Decoder |
| **Stage 2** | **仅 Neck** (2 层 Conv) | **Dice + BCE** (分割 loss) | ViT Encoder + Mask Decoder |

- Stage 1: ViT Encoder 和 CellFinder **联合训练**，loss 是检测 loss
- Stage 2: 用 Dice + BCE 对 neck 做分割微调 — 比 SAM 原版 (20×Focal + Dice + IoU MSE) 更简单
- 微调方法就是**标准监督学习** (loss → 反向传播 → 梯度更新)，没有蒸馏/对比学习等特殊方法

### Q3. `model` vs `model_cp` 的含义？

代码事实 (`sam_inference.py`):
- `__init__` (L127,137): `model = sam_vit_b()`, `model_cp = deepcopy(model)` → 初始时完全相同
- `load_state_dict` (L397-407): 如果 checkpoint 有 `model_cp.*` 键，两者各自加载不同权重；否则 `model_cp` 复制 `model`
- `forward` (L208): `adv_mode=True` 时用 `model_cp.image_encoder`
- `predict` (L327): `adv_mode=True` 时用 `model_cp` 的 prompt_encoder + mask_decoder

> ⚠️ **之前对话中的解读** ("model=原始, model_cp=微调") **是推测**。代码只能证明它们是两份独立的 SAM 权重副本，具体哪份代表什么取决于 checkpoint 内容。`adv` 前缀可能暗示 adversarial training，但代码中无对抗训练证据。

---

## 11. IoU Head 是什么？我们需要加入吗？(2026-02-24)

### Q1. 什么是 IoU Head？

SAM Mask Decoder 有两个输出:
```
SAM Mask Decoder 输出:
├── low_res_masks: 预测 mask (256×256)
└── iou_predictions: IoU head 输出 (标量 0~1) ← 预测"这个 mask 质量如何"
```

训练时用 MSE loss: `L_iou = MSE(predicted_iou, actual_iou)`

### Q2. CellSAM 怎么用 IoU Head 的？

推理时用于**质量过滤**: `if iou_predictions[0][0] < self.iou_threshold: 跳过此 mask`
(`sam_inference.py:350`)

### Q3. 我们需要加入吗？

**暂不需要。** 原因:
- IoU Head 主要用于推理时筛选低质量 mask，对训练本身提升不大
- 当前瓶颈不在 mask 质量评分，而在 encoder 特征适应性
- Best Config PQ=0.484 → T18 PQ=0.498 的提升来自通道信息而非评分机制

---

## 12. Focal Loss 有必要使用吗？(2026-02-24)

### Q1. Focal Loss 是什么？

```
标准 BCE:     L = -y·log(p) - (1-y)·log(1-p)
Focal Loss:   L = -α·(1-p)^γ · y·log(p) - (1-α)·p^γ · (1-y)·log(1-p)
```

- `γ=2` 时: 95% 确信的像素 loss 权重降低 ~400 倍
- 效果: 让模型**专注于边界等难分像素**

### Q2. 对比

| 方面 | Focal Loss | 我们的 BCE + pos_weight=10 |
|------|-----------|--------------------------|
| **解决什么** | 难/易样本不平衡 | 前景/背景数量不平衡 |
| **机制** | 降低易分样本权重 | 提升前景类权重 |
| **SAM 原版** | ✅ (weight=20) | — |
| **CellSAM Stage2** | ❌ (用 Dice+BCE) | — |

### Q3. 建议

**优先级低 (P2)**。原因:
1. CellSAM Stage2 自己也没用 Focal，用的 Dice+BCE — 和我们类似
2. pos_weight=10 已在处理不平衡问题
3. Focal 主要帮助边界像素，但我们已有 BoundaryLoss

---

## 13. Neck 微调需要 loss 吗？(2026-02-24)

**是的。** 任何基于梯度的微调都需要 loss:

```
输入 → 模型前向 → 预测 → 与 GT 计算 loss → 反向传播 → 更新参数
```

"微调 neck" = 只有 neck 参数能被更新 (其他层冻结)，但 loss 仍对最终 mask 输出计算 (Dice + BCE)。梯度从 loss → decoder → neck，只有 neck 参数被 optimizer 更新。

> 不存在"不用 loss 的微调" — 那叫"不训练"。

---

## 14. ALICE T18 训练结果 (2026-02-24)

### Q1. 结果总结

| 实验 | 通道 | PQ | BM-Dice | AJI | Sem.Dice | Best Ep |
|------|------|:--:|:-------:|:---:|:--------:|:-------:|
| T18-A (2ch, seed42) | BF+Actn2 | **0.496** | 0.724 | 0.573 | 0.799 | 27 (ES@22) |
| T18-B (3ch, seed42) | BF+DAPI+Actn2 | **0.498** | 0.725 | 0.574 | 0.801 | 37 |
| T18-C (3ch no-adapter) | BF+DAPI+Actn2 | **0.484** | 0.716 | 0.563 | 0.798 | 28 |

### Q2. 对比 Best Config

| 模型 | PQ | Δ PQ |
|------|:--:|:----:|
| Best Config (BF-only, 4runs mean) | 0.484 | — |
| T18-A (2ch: BF+Actn2) | **0.496** | **+1.2pp** |
| T18-B (3ch: BF+DAPI+Actn2) | **0.498** | **+1.4pp** |

### Q3. 关键发现

1. ✅ 三通道 > BF-only: T18-B (0.498) > Best Config (0.484) **+1.4pp**
2. ✅ 2ch ≈ 3ch: DAPI 通道贡献约 +0.2pp
3. ✅ PQ 首次接近 0.50 大关

> ⚠️ **经验教训**: 查询 ALICE 时曾使用虚构的 SSH 用户名 `s2688211`，正确信息存于 `docs/alice_quick_reference.md` (`s3890074@login.alice.universiteitleiden.nl`)。根因: AI 跳过了"先查文档"的步骤，凭空生成了合理格式的假用户名。所有涉及具体数值/账号/参数的信息必须从源文件核实。

---

## 1. `detach().cpu().numpy()` 和 `scipy.ndimage.label` 是什么？为什么会被用？

### Q1. `detach()` 做什么？
- 功能: 从 PyTorch 计算图中断开当前张量，后续操作不再参与 autograd 反向传播。
- 常见用途: 推理、日志、可视化、导出到 numpy。

### Q2. `.cpu()` 做什么？
- 功能: 把张量从 GPU 拷回 CPU。
- 常见用途: 仅 CPU 库（numpy/scipy/skimage）需要 CPU 内存。

### Q3. `.numpy()` 做什么？
- 功能: 把 CPU tensor 转成 `numpy.ndarray`。
- 结果: 后续 numpy/scipy 操作不在 PyTorch 计算图里。

### Q4. `scipy.ndimage.label` 做什么？
- 功能: 连通域标记（Connected Components）。
- 常用于:
  - 统计碎片数量（fragment）
  - 统计实例个数
  - 形态学后处理分析

### Q5. 当时为什么会用这些？
- 工程上最常见原因:
  1. 传统图像处理实现快（scipy/skimage API 直接可用）
  2. 调试和验证阶段先求“能算出指标/约束值”
  3. 早期方案偏分析脚本思路，后续才转训练可微实现
- 在本项目中，`ContourLoss/TopologyLoss` 正是这样写法导致了可微风险。

代码参考:
- `src/losses/combined.py:203`
- `src/losses/combined.py:331`

---

## 2. 为什么 `return torch.tensor(loss, device=device, dtype=dtype)` 会断图？

### Q1. 这个函数本身做什么？
- 功能: 用 Python 数值或 numpy 数值新建一个 PyTorch tensor。
- 特点: 它是“新叶子张量”，默认不带之前的计算历史。

### Q2. 什么叫“和 `pred` 没有图连接”？
- 训练时希望: `loss` 是 `pred` 的函数，能求 `d(loss)/d(pred)`。
- 若 `loss` 是新建常量张量，autograd 看不到它从 `pred` 怎么来。
- 结果: 该 loss 分支对参数梯度贡献为 0（或近似 0）。

### Q3. 图连接（计算图）意义是什么？
- 计算图是“值如何由参数算出来”的有向图。
- 反向传播通过这张图计算梯度，更新模型参数。
- 没有图连接就没有可用梯度，训练“看见了数值变化”，但“学不到这条约束”。

代码参考:
- `src/losses/combined.py:358`

---

## 3. 当前损失设计（实际生效口径）

`CombinedLoss` 当前结构:
- 基础项: Dice + BCE
- 可选项: Boundary / AJI / Topology / Size / Contour

关键代码:
- `src/losses/combined.py:361`
- `src/losses/combined.py:460`
- `src/losses/combined.py:472`

### Phase 1（当前已锁定）实际生效情况
- 配置:
  - `boundary_weight=1.5`
  - `aji_weight=0.2`
  - `use_contour=true, contour_weight=0.3`
  - `use_topology=false`
  - `use_size=false`
- 参考:
  - `src/config/phase1_rebalance_l4.yaml:33`
  - `src/config/phase1_rebalance_l4.yaml:36`
  - `src/config/phase1_rebalance_l4.yaml:44`
  - `src/config/phase1_rebalance_l4.yaml:45`

结合当前实现，训练中真正提供稳定梯度的主干是:
- Dice
- BCE
- BoundaryLoss
- AJILoss

`ContourLoss` 在当前实现中存在断图路径（`detach().numpy()` + 新建 tensor），对训练梯度贡献极弱/无效。

---

## 4. `base_weight` 触底是什么意思？影响是什么？

当前公式:
- `base_weight = max(0.3, 1.0 - total_extra_weight)`（`src/losses/combined.py:472`）

Phase 1 代入:
- `total_extra = boundary(1.5) + aji(0.2) + contour(0.3) = 2.0`
- `base_weight = max(0.3, -1.0) = 0.3`（触底）

影响:
1. 继续增加 extra 权重，不会再降低 base loss 占比（base 已到底）
2. 总 loss 标量会继续抬高（未归一化区间）
3. 不同配置之间 loss 数值不可直接横向对比
4. 若某 extra 分支不可微（如当前 contour/topology 风险），会放大“只变数值不变梯度”的问题

---

## 5. “相邻细胞像素互相侵占”当前是怎么处理的？

### 训练侧（当前）
- 还没有显式 `L_neighbor/L_overlap` 实现进训练代码。
- 当前主要靠:
  - box 区域裁剪训练
  - Dice/BCE/Boundary/AJI 的间接约束

证据:
- 设计文档写了计划，但代码未落地:
  - `docs/phase2_design.md:158`
  - `docs/codex_claude_seg.md:163`
- 训练调用仍是:
  - `loss = criterion(pred_clipped, target_clipped, box=...)`
  - `src/train.py:278`
  - `src/train.py:320`

### 推理/评估侧（当前）
- 推理冲突像素由规则裁决（统一口径后为 `argmax_prob` 等策略）。
- 侵占率有分析函数，但不是训练 loss:
  - `src/inference/core.py:362`

---

## 6. 验证集和测试集如何使用（项目约定）

推荐与当前实践:
1. 训练阶段/调参: 用 `val`
2. 阶段收官锁定: 用 `test`
3. test 不反向用于调参

当前脚本分工:
- `tools/smoke_test_e2e.py`: Oracle(val) 开发评估（`load_split_ids("val")`）
  - `tools/smoke_test_e2e.py:63`
- `tools/comprehensive_eval.py`: Oracle(test) 最终评估
  - `tools/comprehensive_eval.py:67`
- `tools/evaluate_e2e.py`: E2E(test) 部署路径评估（DAPI 检测框 -> SAM）
  - `tools/evaluate_e2e.py:59`
  - `tools/evaluate_e2e.py:77`
- `tools/test_unified_regression.py`: 统一推理/指标回归测试
  - `tools/test_unified_regression.py:2`

---

## 7. 为什么会出现 `Val > Train`？

常见且在本项目成立的原因:
1. 日志口径不同:
  - 打印的是 `Train Loss` + `Val BM/PQ`，不是同指标比较
  - `src/train.py:547`
2. 训练增强更强，验证更干净:
  - 训练侧有 distortion/noise/box 扰动
  - `src/augmented_dataset.py:107`
  - `src/augmented_dataset.py:120`
  - `src/augmented_dataset.py:349`
3. 优化目标和评估目标不同:
  - 优化的是组合 loss
  - 评估看 BM/PQ/AJI 等

---

## 8. Phase 2 当前建议（简版）

1. 先补齐 Oracle(test)+E2E(test) 的 SQ/RQ 输出
2. 先修 loss 可微性和权重归一化
3. 再上 `L_neighbor/L_overlap`（并加梯度门禁测试）
4. 消融实验坚持单变量（如 lr 和 epoch 拆开）

---

## 9. 关键文件导航

- 训练主入口: `src/train.py`
- 损失实现: `src/losses/combined.py`
- 统一推理核心: `src/inference/core.py`
- 指标实现: `src/metrics/instance_metrics.py`
- Phase 1 设计: `docs/phase1_design.md`
- Phase 2 设计: `docs/phase2_design.md`
- 联合过程文档: `docs/codex_claude_seg.md`
- 项目总览: `CLAUDE.md`

