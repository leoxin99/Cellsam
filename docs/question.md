# CellSAM 项目技术问答汇总（Question）

> 更新时间: 2026-02-12
> 范围: 本轮对话中与训练/推理/评估/损失设计相关的核心技术问题

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

