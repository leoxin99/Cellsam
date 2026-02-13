# Phase 2: 分割质量提升 — 设计文档

> **状态**: 📋 规划中 (2026-02-12)  
> **前置依赖**: Phase 1 已锁定  
> **审核**: 经 Codex 三轮审核修正

---

## 1. Phase 1 瓶颈分析

### 1.1 PQ 分解 (Test, n=73)

| 指标 | Oracle(test) | E2E(test) | Delta |
|------|-------------|-----------|-------|
| **SQ** | 0.616 ± 0.027 | 0.544 ± 0.170 | -12% |
| **RQ** | 0.753 ± 0.162 | **0.290 ± 0.170** | **-61%** |
| TP | 7.5/img | 3.3/img | -56% |
| FP | 2.4/img | **8.5/img** | +254% |
| FN | 2.5/img | 6.7/img | +168% |

> [!IMPORTANT]
> **RQ 从 Oracle 到 E2E 降 61%**——主要瓶颈是 DAPI 检测质量，不是分割质量。
> Oracle SQ(test)=0.616 ≈ SQ(val)=0.623，一致性良好。
> Phase 2 应优先提升 Oracle SQ (分割边界)，同时考虑单独的 Phase 2b 解决检测 RQ。

### 1.2 Phase 1 有效 loss 清单

| Loss | 可微 | 权重 | 作用 |
|------|------|------|------|
| Dice | ✅ | 0.5×base(0.3) | 像素重叠 |
| BCE | ✅ | 0.5×base(0.3) | 分类 |
| BoundaryLoss | ✅ GPU | 1.5 | 边界 BCE+Dice |
| AJILoss | ✅ | 0.2 | soft IoU + FP/FN |
| ContourLoss | ❌ | 0.3 | **零梯度 (numpy)** |
| TopologyLoss | ❌ | OFF | **零梯度 (numpy)** |
| SizeLoss | ❌ | OFF | 有梯度问题待查 |

### 1.3 base_weight 问题

```
base_weight = max(0.3, 1.0 - total_extra_weight)
Phase 1: total_extra = 2.0 → base = 0.3 → 未归一化 → loss 尺度漂移
```

---

## 2. 训练循环数据流分析

**当前结构** (`train.py:218-278`):

```
for i in range(batch_size):        # 逐图
  for j, box in enumerate(boxes):  # 逐 box (逐细胞)
    pred_mask = model(image, box)       # 单细胞预测 (1024×1024)
    target = (mask == cell_id).float()  # 单细胞 GT
    loss = criterion(pred_clipped, target_clipped, box=box)
    batch_loss += loss
```

**对 L_neighbor / L_overlap 的影响**:

| Loss | 能在当前 per-box 循环中计算？ | 原因 |
|------|------|------|
| **L_neighbor** | ✅ 可以 | 只需当前 `pred` + 完整 `sample_mask` (已有) |
| **L_overlap** | ⚠️ 需两趟 | 需先积累所有 box 的 `sigmoid(pred)` 到全图，再算 `sum > 1` |

### L_overlap 实现方案

```python
# 单趟方案 (实用近似):
confidence_map = torch.zeros_like(pred_sigmoid)  # 全图置信度累加
for j, box in enumerate(boxes):
    pred_mask = model(image, box)
    pred_sigmoid = sigmoid(pred_mask)
    
    # L_overlap: 当前细胞的 overlap 惩罚
    # confidence_map 包含“其他”细胞的置信度，不包含当前细胞
    # 只对当前 pred 回传梯度
    local_sum = confidence_map.detach() + pred_sigmoid  # 他人(断梯度) + 自己(有梯度)
    overlap_loss = mean(ReLU(local_sum - 1 - margin)^2)
    
    # 正常计算其他 loss...
    
    # 累加当前 pred 到 confidence_map（供后续 box 使用）
    confidence_map = confidence_map + pred_sigmoid.detach()
```

> [!WARNING]
> 单趟方案是顺序近似——前面的 box 看不到后面的 box。
> 当前 box 顺序来自 `regionprops` 迭代（`src/augmented_dataset.py:333`），**未做每图内 shuffle**。
> 建议在 `train_one_epoch` 中对 `sample_boxes` 做随机打乱以降低近似偏差。

---

## 3. 执行步骤

### Step 1: 补全评估工具 SQ/RQ (~10min)

#### [MODIFY] `tools/comprehensive_eval.py` (line 143)

聚合列表添加 `'sq', 'rq', 'tp', 'fp', 'fn', 'n_gt_cells', 'n_pred_cells'`。

#### [MODIFY] `tools/evaluate_e2e.py` (summary section)

同样补全 SQ/RQ/TP/FP/FN 到摘要输出和 JSON 结果。

然后重跑**两个工具**：
- `comprehensive_eval.py` → Oracle(test) SQ/RQ
- `evaluate_e2e.py --checkpoint ...` → E2E(test) SQ/RQ

对比结果确定 Phase 2 主攻 SQ 还是 RQ。

---

### Step 2: 修复 Loss 基础设施

#### 2a. base_weight 归一化

#### [MODIFY] `src/losses/combined.py` (line 460-474)

```python
# 改为显式归一化:
raw_base = 0.3
total_weight = raw_base + sum(active_extra_weights)
total_loss = (raw_base / total_weight) * base_loss
for each active loss:
    total_loss += (wi / total_weight) * Li
```

#### 2b. ContourLoss → 可微版

#### [MODIFY] `src/losses/combined.py` (line 298-358)

用 BoundaryLoss 同款 `max_pool erosion` 提取边界 + 迭代 dilation 近似距离变换。全程 PyTorch。

#### 2c. TopologyLoss → 可微版

#### [MODIFY] `src/losses/combined.py` (line 167-230)

用形态学 opening (erosion→dilation) 检测碎片。`fragments = pred - opening(pred)`，全程 PyTorch。

#### 2d. 梯度验收门禁 (Codex 要求)

#### [NEW] `tools/test_loss_gradients.py`

```python
def test_loss_has_gradient(loss_fn, name):
    """验证 loss.backward() 后参数有非零梯度"""
    pred = torch.randn(256, 256, requires_grad=True)
    target = (torch.randn(256, 256) > 0).float()
    loss = loss_fn(torch.sigmoid(pred), target)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().max() > 1e-8, \
        f"{name}: 零梯度! Loss 不可微"
```

> [!IMPORTANT]
> **合入门禁**: 任何 loss 修改/新增必须通过此测试。

---

### Step 3: 实现 L_neighbor + L_overlap

> 设计来源: `codex_claude_seg.md` Ch2 (§2.1-§2.5)

#### [NEW] NeighborIntrusionLoss — 添加到 `src/losses/combined.py`

```python
class NeighborIntrusionLoss(nn.Module):
    """L_neighbor(k) = mean(n_k * p_k^gamma)"""
    def __init__(self, gamma=1.5):
        self.gamma = gamma
    
    def forward(self, pred, target, instance_mask):
        # pred: sigmoid 后单细胞预测
        # target: 当前细胞 binary GT
        # instance_mask: 完整多实例 GT (mask > 0 & mask != cell_id)
        cell_region = target > 0.5
        neighbor_region = (instance_mask > 0) & (~cell_region)
        neighbor_mask = neighbor_region.float()
        
        if neighbor_mask.sum() < 1:
            return torch.tensor(0.0, device=pred.device)
        
        intrusion = neighbor_mask * (pred ** self.gamma)
        return intrusion.sum() / (neighbor_mask.sum() + 1e-6)
```

**天然可微**: 纯张量运算，`pred ** gamma` 保持梯度。

#### [NEW] OverlapMutexLoss — 添加到 `src/losses/combined.py`

```python
class OverlapMutexLoss(nn.Module):
    """L_overlap = mean(ReLU(S - 1 - margin)^2)"""
    def __init__(self, margin=0.05):
        self.margin = margin
    
    def forward(self, pred, confidence_map):
        # confidence_map: 同图“其他”box 的 sigmoid(pred) 累加 (detached)
        # 不包含当前细胞——避免重复计入
        local_sum = confidence_map + pred  # 他人(detach) + 自己(有梯度)
        excess = F.relu(local_sum - 1.0 - self.margin)
        return (excess ** 2).mean()
```

#### [MODIFY] `src/train.py` (line 218-278) — 数据流改造

```python
# 主要改动:
# 1. CombinedLoss.forward 新增 instance_mask 参数 → L_neighbor
# 2. 第一趟循环积累 confidence_map → L_overlap
# 3. warmup: 前 5 epoch neighbor/overlap 权重线性升温
```

#### 3c. Warmup 机制

```python
warmup_factor = min(1.0, epoch / 5)
neighbor_w = config_neighbor_w * warmup_factor
overlap_w = config_overlap_w * warmup_factor
```

---

## 4. 实验计划

### P2-A (核心实验 — 先做)

| 参数 | Phase 1 | P2-A |
|------|---------|------|
| boundary_weight | 1.5 | 1.5 |
| aji_weight | 0.2 | 0.2 |
| contour_weight | 0.3 (零梯度) | **0** (关闭) |
| **neighbor_weight** | — | **0.3** |
| **overlap_weight** | — | **0.1** |
| **warmup_epochs** | — | **5** |
| loss 归一化 | ❌ | **✅** |
| 其余 | — | 同 Phase 1 |

### P2-B (可微边界 — P2-A 后评估决定)

在 P2-A 基础上启用可微版 ContourLoss(0.3) + TopologyLoss(0.1)。

前置: Step 2 梯度验收通过。

### P2-D / P2-E (学习率消融 — 拆分单变量)

| ID | 变更 | 测试假说 |
|----|------|---------|
| P2-D | lr=5e-5, epochs=50 | 纯 lr 影响 |
| P2-E | lr=1e-4, epochs=80 | 纯 epoch 影响 |

> 原 P2-C (lr+epochs 同时改) 按 Codex 建议拆分。

---

## 5. 成功标准

| 指标 | Phase 1 | Phase 2 目标 |
|------|---------|-------------|
| Oracle PQ | 0.464 | **≥ 0.50** |
| Oracle SQ | **0.616** (test) | **≥ 0.70** |
| Oracle BM-1to1 | 0.695 | **≥ 0.72** |


## 6. 执行顺序

```
Step 1: 补全 SQ/RQ → 确认 test SQ/RQ → 验证方向
  ↓
Step 2: 修复 loss 基础设施 (base_weight + ContourLoss + TopologyLoss)
  ↓ 梯度验收门禁通过
  ↓
Step 3: L_neighbor + L_overlap + train.py 改造
  ↓ 1-epoch smoke test
  ↓
P2-A: ALICE 训练 → Oracle(test) + E2E(test) 锁定
  ↓ 评估后决定是否继续 P2-B/D/E
```
