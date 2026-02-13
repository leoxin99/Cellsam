# Phase 1: Loss Weight Rebalancing — 设计与回顾

> **状态**: ✅ 已锁定 (2026-02-11)  
> **模型**: `checkpoints/E_phase1_rebalance_l4/best_model.pt` (epoch 49)

---

## 1. 目标

验证 loss 权重调整 + PQ 早停能否在 E29 baseline 基础上显著提升 PQ 和 BM-1to1 Dice。

## 2. 配置变更 (vs E29 baseline)

| 参数 | E29 | Phase 1 | 变更原因 |
|------|-----|---------|---------|
| boundary_weight | 0.5 | **1.5** | 增强边界学习信号 |
| contour_weight | OFF | **0.3** | 原设计为边界距离约束（事后发现零梯度）|
| pos_weight | 10.0 | **2.0** | 降低正样本过度补偿 |
| use_pq_early_stop | false | **true** | PQ 比 Dice 更反映实例分割质量 |
| use_topology | false | false | 两阶段均关闭 |
| use_size | false | false | 两阶段均关闭 |

**配置文件**: `src/config/phase1_rebalance_l4.yaml`

## 3. 训练结果

| | L4 (Job 974531) | A100 (Job 974530) |
|---|---|---|
| Epochs | 50/50 | ~47/50 (PQ early stop) |
| Best epoch | 49 | 32 |
| Val Dice | 0.6927 | 0.6828 |
| Val PQ | 0.4750 | 0.4533 |

## 4. Test 集锁定 (73 samples)

| 指标 | Oracle | E2E | BF_Baseline Oracle |
|------|--------|-----|-------------------|
| BM-1to1 Dice | **0.6954** | 0.5446 | 0.4695 |
| PQ | **0.4641** | 0.1719 | 0.0577 |
| AJI | **0.5195** | 0.3181 | 0.2853 |

## 5. 事后发现

> [!WARNING]
> ContourLoss (weight=0.3) 在 `src/losses/combined.py:331` 使用 `detach().numpy()`，**零梯度**。  
> Phase 1 实际有效 loss 仅：**Dice + BCE + BoundaryLoss(1.5) + AJILoss(0.2)**。  
> ContourLoss 只改变了 loss 标量显示，未提供训练信号。

### base_weight 触底

```
total_extra = boundary(1.5) + aji(0.2) + contour(0.3) = 2.0
base_weight = max(0.3, 1 - 2.0) = 0.3  ← 已触底
```

loss 尺度未归一化，Phase 2 需修复。

## 6. 结论

- PQ +704% vs BF_Baseline → 方向正确
- test/val 一致 → 当前未见明显过拟合迹象（val 为 n=30 抽样）
- E2E PQ 仅 0.172 → 瓶颈在 DAPI 检测框质量
- ContourLoss 零梯度 → Phase 2 修复
