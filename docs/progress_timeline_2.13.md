# CellSAM 项目汇报与后续计划（2.13 时间线）

> 更新时间：2026-02-13  
> 目标：用于导师汇报的快速材料（当前进展 + 核心问题 + 下一阶段实验）

---

## 1. 当前在做什么（当前训练任务）

- 当前阶段：**Phase 2 Step 4（P2-A 训练）**
- 当前配置：`src/config/phase2a_neighbor_overlap.yaml`
- 训练入口：`scripts/train_phase2a.sh`（L4）与 `scripts/train_phase2a_a100.sh`（A100）
- 本轮核心改动：在 Phase 1 锁定配置上新增
  - `L_neighbor`（抑制当前细胞侵占邻居 GT 区域）
  - `L_overlap`（抑制多细胞对同一像素的过度重叠占有）
- 保持不变：主干训练框架、统一推理口径、PQ 早停
- 关闭项：`TopologyLoss` / `SizeLoss`（P2-A 暂不启用）

---

## 2. 为什么推理阶段会有“冲突像素”

在多框推理中，每个框都会单独产生一张概率图；当两个（或更多）框的前景区域在同一像素位置同时超过阈值，就会出现“冲突像素”。

- 对应实现：`src/inference/core.py:258` 的 `resolve_conflicts()`
- 输入：`pred_stack[N,H,W]`（N 个实例的预测）
- 冲突定义：`overlap_count >= 2`（`src/inference/core.py:281`-`src/inference/core.py:283`）
- 冲突归属策略：
  - `argmax_prob`：归给概率最高实例（默认）
  - `first_write`：先写入者保留
  - `last_write`：后写入者覆盖

这也是为什么即使是推理阶段，也必须显式做冲突裁决。

---

## 3. 已完成里程碑（可直接汇报）

### 3.1 Phase 1 已锁定（test 集）

来自 `CLAUDE.md` 锁定结果：

- Oracle(test)：
  - BM-1to1 Dice = **0.6954**
  - PQ = **0.4641**
  - AJI = **0.5195**
- E2E(test)：
  - BM-1to1 Dice = **0.5446**
  - PQ = **0.1719**
  - AJI = **0.3181**

### 3.2 当前瓶颈结论

- Oracle 显著高于 E2E，说明分割上限已较好，**部署链路仍有明显损失**。
- 当前主攻方向：减少相邻细胞互侵（实例冲突）并提升端到端稳定性。

---

## 4. 后续阶段与实验计划

### Phase 2（当前主线）

1. **Step 4（进行中）**：完成 P2-A 训练（L_neighbor + L_overlap）
2. **Step 5（下一步）**：统一口径评估（Oracle(test) + E2E(test)）并锁定结果
3. **是否进入 P2-B（条件触发）**：
   - 条件：P2-A 指标有稳定收益
   - 内容：评估“全局对称 overlap”版本（替代单趟近似）
4. **可选消融**：P2-D / P2-E（学习率与 epoch 拆分验证）
5. **若 E2E 仍受限**：单开检测链路优化分支（提升 RQ）

### Phase 3（规划中）

- 三通道 Adapter 对比实验（与当前主线统一口径横向对照）

### 阶段4（论文结果）

- 锁定最终模型与评估表，形成可复现实验包与论文结果表格

---

## 5. 导师汇报简版（30-60 秒）

“我们已经完成并锁定了 Phase 1，test 集上 Oracle 达到 BM-1to1 0.695、PQ 0.464；但 E2E 仍只有 BM-1to1 0.545、PQ 0.172，说明瓶颈主要在实例冲突和端到端链路。当前在做 Phase 2 的 P2-A 训练，在不改主框架的前提下新增 L_neighbor 和 L_overlap，专门抑制相邻细胞像素互侵。训练完成后会按统一口径做 Oracle/E2E 锁定评估，再决定是否进入 P2-B 的全局对称 overlap 版本。”  

