# Phase 2: 分割质量提升 — 设计文档

> **状态**: 🟡 P2-A 训练完成 / 退化分析中 (2026-02-14)  
> **前置依赖**: Phase 1 已锁定  
> **审核**: 经 Codex 三轮审核修正 + P2-A 退化分析 (2026-02-14)

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

---

## 7. P2-A 训练结果与退化分析 (2026-02-14)

### 7.1 训练概况

| Job | GPU | Dice | PQ | Early Stop | 耗时 | Checkpoint |
|-----|-----|------|----|------------|------|------------|
| 979158 | **L4** | **0.6099** | **0.2206** | ep39 (patience=15) | 2h45m | `E_phase2a_..._042102/` |
| 979159 | A100 | 0.5972 | 0.2078 | ep35 | 0h55m | `E_phase2a_..._074819/` ^1^ |
| 979161 | A100 | — | — | ep35 | 1h24m | 同上 (覆盖 979159) |

^1^ 979159 本地无独立日志；数值来源为 SLURM `sacct` 输出与 979161 日志交叉推断（`logs/p2a_a100_979161.log`）。

**Config**: `src/config/phase2a_neighbor_overlap.yaml`  
**Enabled losses**: Dice, BCE, Boundary(1.5), AJI(0.2), Contour(0.3), **Neighbor(0.3)**, **Overlap(0.1)**  
**Gradient gate**: 12/12 通过 (pre-flight)

### 7.2 与 Phase 1 对比

| 指标 | Phase 1 L4 | Phase 2A L4 | 变化 |
|------|-----------|-------------|------|
| **Best PQ** | 0.4750 (ep49) | 0.2206 (ep24) | **-53%** ❌ |
| **Best Dice** | 0.6927 | 0.6099 | -12% |
| Train Loss (final) | 0.2748 | 0.3418 | +24% 未收敛 |
| Conflict count (范围) | 40,000-50,000 | 10,000-24,000 | **-50%~-75%** |
| Sem Dice (best) | 0.758 | 0.658 | -13% |
| Epochs | 50 (完整) | 39 (early stop) | - |

### 7.3 退化根因分析

#### 🔴 嫌疑 1 (首要): 未加载 P1 微调权重

```yaml
# P1 config            # P2A config
model:                  model:
  checkpoint: null        checkpoint: null   # ← 未加载 P1 best!
```

> **澄清**: `checkpoint: null` 不等于随机初始化。`train.py:104` 调用 `get_model()` 会加载 CellSAM 预训练权重（`cellSAM_source/cellSAM/model.py:50`）。  
> 问题是：P2A 从 CellSAM 通用权重出发，**未继承 P1 已学习的心肌细胞特化能力**。模型需同时：

1. 重新学习心肌细胞域适应（P1 已完成但此处丢失）
2. 适应 Neighbor/Overlap 新 Loss

**结果**: Train loss 从未降到 P1 水平 (0.34 vs 0.27)，分割质量天花板被拉低。

> ⚠️ **尚需对照实验验证**: 要确认这是主因而非 loss 设计缺陷，需跑：`checkpoint=P1_best + 其余不变`。在此之前仅为**强怀疑**。

#### 🟡 根因 2: Loss 权重归一化稀释

**P1 权重分配** (`total_weight = 0.3 + 1.5 + 0.2 + 0.3 = 2.3`):

| Loss | raw_weight | 归一化占比 |
|------|------------|------------|
| base (Dice+BCE) | 0.3 | 13.0% |
| Boundary | 1.5 | 65.2% |
| AJI | 0.2 | 8.7% |
| Contour | 0.3 | 13.0% |

**P2A 权重分配** (`total_weight = 2.3 + 0.3 + 0.1 = 2.7`, neighbor/overlap computable 时):

| Loss | raw_weight | 归一化占比 | 相比 P1 |
|------|------------|------------|--------|
| base | 0.3 | 11.1% | ↓ -1.9pp |
| Boundary | 1.5 | 55.6% | ↓ -9.6pp |
| AJI | 0.2 | 7.4% | ↓ -1.3pp |
| Contour | 0.3 | 11.1% | ↓ -1.9pp |
| **Neighbor** | 0.3 | 11.1% | NEW |
| **Overlap** | 0.1 | 3.7% | NEW |

> ⚠️ Boundary 权重从 65% 降到 56%，分割核心梯度信号被削弱 ~15%。

> ⚠️ **Computability gating 问题**: Neighbor/Overlap 权重仅在 `instance_mask` 和 `confidence_map` 可用时计入分母。Pre-flight 打印 `base=15%, boundary=75%` 是 N/O 不可计算时的值，**实际训练中权重在两种模式间切换**。

#### 🟡 根因 3: 前 5 Epoch 冷启动

```
P2A: Epoch 1-5: BM-1to1=0.000, PQ=0.000, Sem=0.000, Conflict=0
P1:  Epoch 1:   BM-1to1=0.425, PQ=0.089, Sem=0.693, Conflict=183,126
```

P2A 前 5 个 warmup epoch 模型输出全部低于阈值，没有产生任何有效分割。P1 第 1 epoch 就有可评估输出。

#### 🟠 嫌疑 4: Conflict 数大幅下降（过于保守分割）

P1 稳定在 40-50k conflicts，P2A 范围 10k-24k（低谷 ep17=10,434，高峰 ep18=23,997）。Neighbor/Overlap loss 推动模型回避重叠区域，但代价是**吞掉过多 true positive**——Conflict 减少并未转化为 PQ 提升。

### 7.4 PQ 曲线对比

```
P1 L4:  ep4→0.30  ep11→0.43  ep20→0.43  ep27→0.45  ep39→0.46  ep49→0.475
        (稳步上升，最后一个 epoch 刷新 best)

P2A L4: ep6→0.12  ep18→0.21  ep24→0.22  ep25→0.08  ep31→0.11  ep36→0.05
        (ep24 后剧烈震荡 0.05-0.11 → early stop@39)
```

> P2A 的 PQ 在达到 0.22 后**无法再提升**，在 0.05-0.11 间震荡 15 个 epoch。疑似 **loss 冲突** 导致的训练不稳定，需通过对照实验（`checkpoint=P1_best`，其余不变）验证后确认。

### 7.5 修复方案

| 修复 | 内容 | 优先级 |
|------|------|--------|
| **P2A-fix1: 从 P1 微调** | config 设 `checkpoint: checkpoints/E_phase1_.../best_model.pt` | ⭐⭐⭐ |
| P2A-fix2: 降低新 loss 权重 | neighbor: 0.3→0.1, overlap: 0.1→0.05 | ⭐⭐ |
| P2A-fix3: 延迟启用 | 前 10 epoch 只用 P1 losses，之后线性升温 | ⭐ |
| P2A-fix4: 延长 warmup | warmup_epochs: 5→10 | ⭐ |

### 7.6 结论

训练过程本身正常（梯度门禁 12/12 pass，loss 在下降，early stop 逻辑正确）。  
性能退化的**首要嫌疑是未加载 P1 微调权重** (`checkpoint: null`)，但 L_neighbor / L_overlap 的设计缺陷**尚未排除**，需对照实验确认。  
**下一步**: 修复 config 后重新提交 P2A 训练（仅改 checkpoint，其余不变），**假设/目标** PQ ≥ 0.47 (至少 match P1)。若对照仍退化，则需追查 loss 设计。

---

## 8. P2-A Fix1 结果与 Fix2 计划 (2026-02-14)

### 8.1 Fix1 (从 P1 微调) 实验结果

**Job**: 980761 (L4), 980763 (A100)  
**Config**: checkpoint = P1 Best, losses unchanged (Neighbor 0.3, Overlap 0.1)

| 指标 | Phase 1 (基线) | P2-A Initial (从零) | P2-A Fix1 (从 P1) |
|------|---------------|-------------------|-------------------|
| **Best PQ** | **0.4750** | 0.2206 | **0.2322** (L4) |
| **Best Dice** | 0.6927 | 0.6099 | 0.6382 |
| **Conflict Count** | 40-50k | 12-15k | **11k-23k** (L4: 11035-22153, A100: 11272-23010) |
| **状态** | Locked | Failed | **Failed** |

**结论**:
1. **P1 权重未能挽救退化**: 仅在最初几个 epoch 略有提升，随后 PQ 迅速坍塌至 ~0.10，表现出与 Initial 相同的"loss 冲突"特征。
2. **冲突抑制波动大**: Conflict 在 11k-23k 间大幅波动 (P1 稳定在 40-50k), 整体偏低说明 exclusion 约束仍然过强, 但并非始终压制.
3. **首要嫌疑加强** (非根因确认): Fix1 排除了冷启动假设, 将 Loss 权重过大 从嫌疑提升为首要嫌疑. 但仅凭 Fix1 不能排除其他因素 (lr/warmup). 需 Fix2/Fix3/Fix4 对照才能最终定性.

### 8.2 Fix2 计划 (降低权重)

根据 §7.5 计划，进入 **P2A-fix2**。

**改动内容**:
1. **L_neighbor**: 0.3 → **0.1** (降低 3 倍)
2. **L_overlap**: 0.1 → **0.05** (降低 2 倍)
3. **保持**: 继续加载 P1 Checkpoint (作为好的起点)

**预期**:
减轻对分割主干的干扰，寻求 exclusion 与 segmentation 的平衡点。

**Experiment Name**: `E_phase2a_fix2_low_weight`

> **复现口径** (Codex review #4): `phase2a_neighbor_overlap.yaml` 已覆盖为 Fix2 参数。Fix1 完整配置存档于 Alice `checkpoints/E_phase2a_fix1_from_p1_*/config.yaml`。后续若需复现 Fix1 应从存档恢复。

### 8.3 Fix2 结果 (2026-02-15)

**Job**: 981146 (L4), 981147 (A100)  
**Config**: checkpoint = P1 Best, **Neighbor = 0.1** (↓3x), **Overlap = 0.05** (↓2x)

| 指标 | Phase 1 (基线) | Fix1 (N=0.3, O=0.1) | **Fix2 (N=0.1, O=0.05)** | Fix2 vs P1 |
|------|---------------|---------------------|--------------------------|-----------|
| **Best PQ** | **0.4750** | 0.2322 | **0.3929** | **-17%** ❌ |
| **Best Dice** | 0.6927 | 0.6382 | **0.6867** | **-0.9%** ✅ |
| Gradient gate | — | — | **12/12** ✅ | — |

**关键发现**:
1. **权重过大是退化主因**: Fix1→Fix2 唯一变量是降权，PQ 恢复 **+69%**。证实 §7.3 "嫌疑 2 (Loss 权重归一化稀释)" 为主因。
2. **PQ vs Dice 分离**: Dice 几乎恢复 (0.69→0.69) 但 PQ 差 17%。说明像素分割质量已恢复，**实例匹配退化** (IoU 阈值敏感)。
3. **可能原因**: N/O 仍在抑制边界区域，导致边界 IoU 刚好跌破匹配阈值。

**R1 审核**: ⚠️ 有条件通过 (见 `docs/temp_reviews/fix2_review.md`)

### 8.4 Fix3 计划 (延迟启用)

**思路**: 前 10 个 epoch 用纯 P1 losses (N/O weight = 0)，让模型先收敛到 P1 水平，然后线性升温 N/O。

**代码改动**:
- `CombinedLoss.set_epoch(epoch)`: 根据 `delay_epochs`/`ramp_epochs` 动态缩放 N/O 权重
- `train.py`: 每 epoch 开头调用 `criterion.set_epoch(epoch)`
- Config: `delay_epochs: 10`, `ramp_epochs: 10`

**权重调度表**:

| Epoch | N weight | O weight |
|-------|----------|----------|
| 0-9   | 0        | 0        |
| 10    | 0.01     | 0.005    |
| 15    | 0.05     | 0.025    |
| 20+   | 0.1      | 0.05     |

**Experiment Name**: `E_phase2a_fix3_delayed`

**止损线** (R1 review): Fix3 PQ < 0.45 → 终止 P2-A 路线，论文定位为 "Preliminary Exploration"

> **复现口径**: Fix2 配置存档于 Alice `checkpoints/E_phase2a_fix2_low_weight_*/config.yaml`。`phase2a_neighbor_overlap.yaml` 已更新为 Fix3 参数。

