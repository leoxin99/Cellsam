# 边界增强技术设计方案

> **目的**: 详细说明建议的 Loss 函数和数据增强方案
> **更新**: 2026-02-04
> **状态**: 设计阶段 (待审批)

---

## 一、建议添加的 Loss 函数

### 1.1 Topology Loss (拓扑损失)

**功能**: 确保预测 mask 的拓扑结构正确 (无空洞、保持连通性)

**原理**:
```
检测预测中的"洞"和"碎片"并惩罚

正常细胞: ████████   → 1个连通区域 ✓
有洞:     ███░███    → 1个区域但有洞 ✗
碎片:     ███  ███   → 2个碎片 ✗
```

**实现方案**:
```python
def topology_loss(pred_mask, gt_mask):
    """
    惩罚预测中的:
    1. 小碎片 (面积 < 阈值的连通区域)
    2. 孔洞 (内部非零区域)
    """
    from scipy import ndimage
    
    pred_binary = (pred_mask > 0.5).float()
    
    # 标记连通区域
    labeled, n_components = ndimage.label(pred_binary.cpu().numpy())
    
    # 惩罚碎片 (面积 < min_size)
    min_size = 16559  # E17 P1 (40836) scaled to 1024px (×0.4055)
    fragment_penalty = 0
    for i in range(1, n_components + 1):
        component_size = (labeled == i).sum()
        if component_size < min_size:
            fragment_penalty += 1.0  # 每个碎片惩罚
    
    # 归一化
    loss = fragment_penalty / max(n_components, 1)
    return loss
```

**预期效果**: 减少过分割产生的小碎片

---

### 1.2 Contour Loss (轮廓距离损失)

**功能**: 直接惩罚边界的距离误差

**原理**:
```
计算预测边界到GT边界的距离场

GT边界:      ░░████░░
Pred边界:    ░███░░░░
距离误差:      ↑↑↑↑
```

**实现方案**:
```python
def contour_loss(pred_mask, gt_mask):
    """
    使用距离变换计算边界误差
    """
    from scipy.ndimage import distance_transform_edt
    
    pred_binary = (pred_mask > 0.5).float().cpu().numpy()
    gt_binary = (gt_mask > 0).float().cpu().numpy()
    
    # 提取边界 (使用梯度)
    from scipy.ndimage import binary_erosion
    pred_boundary = pred_binary - binary_erosion(pred_binary)
    gt_boundary = gt_binary - binary_erosion(gt_binary)
    
    # GT边界的距离场
    gt_distance = distance_transform_edt(1 - gt_boundary)
    
    # 预测边界点到GT边界的平均距离
    if pred_boundary.sum() > 0:
        avg_distance = (gt_distance * pred_boundary).sum() / pred_boundary.sum()
    else:
        avg_distance = 0
    
    return avg_distance
```

**预期效果**: 边界更准确地对齐GT边界

---

### 1.3 Instance Dice Loss (实例级 Dice 损失)

**功能**: 每个实例单独计算 Dice，然后平均

**原理**:
```
当前: Dice = 所有细胞一起算 → 大细胞主导
改进: Dice = Σ(每个细胞的Dice) / N → 公平对待

小细胞分割差 → 当前Dice影响小
小细胞分割差 → Instance Dice影响大
```

**实现方案**:
```python
def instance_dice_loss(pred_probs, gt_mask, boxes):
    """
    对每个框/实例单独计算 Dice，然后平均
    
    Args:
        pred_probs: 预测概率图 (sigmoid后)
        gt_mask: 实例标签图
        boxes: 检测到的框列表
    """
    instance_dices = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # 提取框内区域
        pred_region = pred_probs[y1:y2, x1:x2]
        gt_region = gt_mask[y1:y2, x1:x2]
        
        # 该框对应的GT实例 (取众数)
        gt_labels_in_box = gt_region[gt_region > 0]
        if len(gt_labels_in_box) == 0:
            continue
        target_label = np.bincount(gt_labels_in_box.astype(int)).argmax()
        
        # 二值化
        pred_binary = (pred_region > 0.5).float()
        gt_binary = (gt_region == target_label).float()
        
        # Dice
        intersection = (pred_binary * gt_binary).sum()
        dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
        instance_dices.append(dice)
    
    # 平均
    if len(instance_dices) > 0:
        return 1 - torch.mean(torch.stack(instance_dices))
    return 0
```

**预期效果**: 小细胞得到更多关注

---

## 二、边界增强数据增强

### 2.1 当前已有增强

```python
# augmented_dataset.py
A.ElasticTransform(alpha=120, sigma=12, p=0.3)  # ✅ 已有
```

### 2.2 建议添加的增强

#### GridDistortion (网格扭曲)

```python
A.GridDistortion(
    num_steps=5,        # 网格点数
    distort_limit=0.3,  # 扭曲幅度
    p=0.3
)
```
**效果**: 局部区域独立扭曲，模拟细胞形态变化

#### OpticalDistortion (光学畸变)

```python
A.OpticalDistortion(
    distort_limit=0.2,
    shift_limit=0.1,
    p=0.2
)
```
**效果**: 整体畸变，模拟镜头畸变

#### CoarseDropout (局部遮挡)

```python
A.CoarseDropout(
    max_holes=8,        # 最多遮挡块数
    max_height=50,
    max_width=50,
    fill_value=0,
    p=0.2
)
```
**效果**: 随机遮挡边界区域，强制模型学习部分可见边界

---

## 三、消融实验计划

### 3.1 Loss 消融

| 实验ID | 配置 | 对比 |
|--------|------|------|
| L1 | Base (Dice + BCE + Boundary) | 当前基线 |
| L2 | L1 + Topology Loss | +拓扑约束 |
| L3 | L1 + Contour Loss | +轮廓距离 |
| L4 | L1 + Instance Dice | +实例级 |
| L5 | L1 + L2 + L3 + L4 | 全部 |

### 3.2 增强消融

| 实验ID | 配置 | 对比 |
|--------|------|------|
| A1 | 当前增强 (无新增) | 基线 |
| A2 | A1 + GridDistortion | +网格扭曲 |
| A3 | A1 + OpticalDistortion | +光学畸变 |
| A4 | A1 + CoarseDropout | +遮挡 |
| A5 | A1 + A2 + A3 + A4 | 全部新增 |

---

## 四、实施优先级

| 优先级 | 项目 | 难度 | 预期收益 |
|--------|------|------|----------|
| P0 | Instance Dice Loss | 低 | 高 |
| P1 | GridDistortion 增强 | 低 | 中 |
| P2 | Contour Loss | 中 | 中 |
| P3 | Topology Loss | 高 | 中 |

---

## 五、审批状态

- [ ] 用户审批 Loss 方案
- [ ] 用户审批增强方案
- [ ] 用户审批消融实验计划
