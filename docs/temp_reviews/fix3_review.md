# R1 审核报告: P2-A Fix3 训练结果

> **日期**: 2026-02-16  
> **审核对象**: A2(Claude) P2-A Fix3 延迟启用  
> **相关 inbox 条目**: [2026-02-16 03:00]

---

## 1. 审核结论

### ⚠️ 有条件通过 + P2-A 路线终止

Fix3 Best PQ=0.4657 超过止损线 (0.45)，但 best 发生在 **N/O 未激活的 epoch 3**。N/O 升温后 PQ 单调下降至 0.34。三轮 Fix 一致证明 N/O loss 在当前设计下对 PQ 有负面影响。

**决策: 终止 P2-A 路线。**

---

## 2. 全系列对比

| | P1 基线 | P2-A Initial | Fix1 | Fix2 | **Fix3** |
|-|---------|-------------|------|------|---------|
| N weight | — | 0.3 | 0.3 | 0.1 | 0.1 (delayed) |
| O weight | — | 0.1 | 0.1 | 0.05 | 0.05 (delayed) |
| Checkpoint | — | CellSAM | P1 | P1 | P1 |
| **Best PQ** | **0.4750** | 0.2206 | 0.2322 | 0.3929 | **0.4657** |
| **Best Dice** | 0.6927 | 0.6099 | 0.6382 | 0.6867 | **0.7117** |
| Best Epoch | 49 | 24 | — | — | **3** |
| N/O active @best? | — | ✅ | ✅ | ✅ | **❌ OFF** |

### 关键发现

1. **Fix3 的 best 发生在 N/O 关闭期间** — 本质上是 P1 的续训练 (从 P1 checkpoint + P1 losses)
2. **Fix3 ep1-9 Dice=0.711 > P1=0.693** — 说明 P1 checkpoint 仍有进步空间，但不需要 N/O
3. **每次 N/O 激活后，PQ 都下降** — 不论权重大小、是否延迟、是否加载 P1

---

## 3. 训练轨迹分析

### N/O 升温与 PQ 下降的因果关系

```
Epoch:    1    3    9   |10   12   15   18
N/O:    OFF  OFF  OFF  |ON↗  20%  50%  80%
PQ:     .34  .47  .46  |.40  .42  .38  .34
Loss:   .108 .106 .106 |.105 .121 .164 .210
```

N/O 升温的即刻效应: train loss 翻倍、PQ 退化。Conflict 从 50k 降到 29k，说明 N/O 过度抑制边界区域像素。

---

## 4. P2-A 路线终止理由

| 证据 | 结论 |
|------|------|
| 3 轮 Fix, 4 种权重配置, 全部 PQ 退化 | N/O loss 设计有根本缺陷 |
| Fix3 best@ep3 (N/O OFF) 优于 P1 | 续训练有价值，但 N/O 无价值 |
| Conflict 下降未转化为 PQ 提升 | 排斥约束过于激进，杀死 TP |

### 论文定位建议

> "We explored neighbor intrusion and overlap mutex losses to reduce inter-cell boundary confusion. However, across three configurations (full weight, reduced weight, delayed activation), the exclusion losses consistently degraded PQ by suppressing true positive predictions. The approach warrants further investigation with per-instance adaptive weighting."

---

## 5. 建议保留的资产

| 资产 | 用途 |
|------|------|
| Fix3 ep3 checkpoint | "P1+" 候选 (Dice 比 P1 高 2.7%) |
| `NeighborIntrusionLoss` / `OverlapMutexLoss` 代码 | 论文 Methods 描述 |
| Fix1-Fix3 全系列数据 | 论文消融表 |
