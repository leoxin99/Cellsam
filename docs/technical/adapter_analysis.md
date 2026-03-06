# 训练验证指标与 Adapter 设计分析

> **日期**: 2026-02-02
> **状态**: 技术分析报告

---

## 1. 为什么 BF Baseline 和 Semantic Adapter 结果相近？

### 训练结果对比

| 模型 | Best Val Dice | 训练样本 |
|------|---------------|----------|
| BF Baseline (全数据集) | **0.7349** | 334 |
| Semantic Adapter | **0.7353** | 334 |
| 差异 | **+0.04%** | - |

### 原因分析

#### A. 验证指标只有 Pixel Dice

当前 `train.py` 中的 `validate()` 函数 **只计算像素级 Dice**：

```python
# train.py L342-347
pred_binary = (combined_pred > 0.5).float()
target_binary = (sample_mask > 0).float()
intersection = (pred_binary * target_binary).sum()
dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
```

**问题**：
- Pixel Dice 无法区分实例级分割质量
- 两个模型可能 Dice 相同但 PQ/AJI 差异大
- 需要运行 `tools/comprehensive_eval.py` 获取完整指标

#### B. Adapter 设计限制

**当前 Adapter 参数**：

```
3 个 Conv2d(1, 1, 3×3) = 3 × (9 + 1) = 30 参数
```

每个卷积核仅能学习 **局部空间变换**，无法：
1. 跨通道融合信息 (Actn2+BF+DAPI 联合特征)
2. 学习全局对比度调整
3. 捕获长距离依赖

#### C. 恒等初始化 + 低学习率

```python
# 恒等初始化: 中心=1, 其他=0
conv.weight.data[:, :, center, center] = 1.0
```

加上 LR=1e-5，30 个 epoch 后参数几乎没有显著变化。

---

## 2. Val Dice 代表什么？为什么不全面？

### Val Dice 定义

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|}$$

- P = 预测 mask (所有细胞合并)
- G = Ground Truth mask

### 局限性

| 问题 | 说明 |
|------|------|
| **像素级 vs 实例级** | 无法区分"5个细胞预测正确"和"10个细胞都偏一点" |
| **不惩罚过分割** | 把 1 个细胞切成 2 块，Dice 可能不变 |
| **不惩罚欠分割** | 把 2 个细胞合成 1 块，Dice 可能不变 |
| **边界精度不敏感** | 边界偏移 10 像素，Dice 变化很小 |

### 应使用的完整指标

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **PQ@0.5** | Panoptic Quality (实例级) | ⭐⭐⭐⭐⭐ |
| **AJI** | Aggregated Jaccard Index | ⭐⭐⭐⭐ |
| **Boundary IoU** | 边界精度 | ⭐⭐⭐ |
| **Instance Dice** | 匹配实例的平均 Dice | ⭐⭐⭐ |
| Pixel Dice | 全局像素重叠 | ⭐⭐ |

---

## 3. Adapter 30 参数学的是什么？

### 当前架构

```python
class IndependentChannelAdapter:
    actn2_conv = Conv2d(1, 1, 3×3)  # 学习 Actn2 通道的局部纹理增强
    bf_conv = Conv2d(1, 1, 3×3)     # 学习 BF 通道的边缘锐化
    dapi_conv = Conv2d(1, 1, 3×3)   # 学习 DAPI 通道的降噪
```

### 30 参数能学什么？

每个 3×3 卷积核可以学习：
- **锐化** (Laplacian): 中心正，周围负
- **平滑** (Gaussian): 接近恒等
- **边缘检测** (Sobel): 方向梯度

但这些效果 **已经在 SemanticChannelMapper 预处理中实现了**：
- CLAHE 增强 BF
- 高斯平滑 DAPI
- 百分位截断 Actn2

**结论**：Adapter 学习的特征与预处理重复，贡献有限。

---

## 4. 如何改进 Adapter 学到有效变换？

### 方案 A: 增加跨通道融合 (推荐)

```python
class CrossChannelAdapter(nn.Module):
    """跨通道融合 Adapter - ~90 参数"""
    def __init__(self):
        super().__init__()
        # 1x1 卷积做通道融合
        self.channel_mix = nn.Conv2d(3, 3, kernel_size=1)
        # 3x3 卷积做空间变换
        self.spatial = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
        
    def forward(self, x):
        x = self.channel_mix(x)  # 跨通道融合
        x = self.spatial(x)       # 空间变换
        return x
```

**参数量**: 3×3 + 9×3 = 9 + 27 = 36 (1x1) + 27 (3x3) ≈ 63 参数

### 方案 B: 带注意力的 Adapter

```python
class AttentionAdapter(nn.Module):
    """带通道注意力的 Adapter - ~50 参数"""
    def __init__(self, reduction=4):
        super().__init__()
        # Squeeze-Excitation 风格的通道注意力
        self.fc1 = nn.Linear(3, 3 // reduction)  # 不适用 reduction<2
        self.fc2 = nn.Linear(3 // reduction, 3)
        self.spatial = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        
    def forward(self, x):
        # 全局池化 + 通道权重
        b, c, h, w = x.shape
        y = x.mean(dim=[2, 3])  # (B, 3)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # (B, 3)
        x = x * y.view(b, c, 1, 1)
        return self.spatial(x)
```

### 方案 C: LoRA 风格低秩分解

在 ViT 的 QKV 层插入低秩适配器：

```python
class LoRAAdapter(nn.Module):
    """LoRA 风格 Adapter for ViT - ~1000 参数"""
    def __init__(self, dim=1024, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, dim))
        
    def forward(self, x):
        return x + x @ self.lora_A @ self.lora_B
```

**优势**：可以在 ViT 内部学习特征变换，效果更强

---

## 5. 改进建议优先级

| 优先级 | 建议 | 预期效果 |
|--------|------|----------|
| **P0** | 运行全面评估 (`comprehensive_eval.py`) | 获取 PQ/AJI 真实差异 |
| **P1** | 实现跨通道融合 Adapter | 预期 +2-5% |
| **P2** | 在训练中加入 PQ/AJI 作为验证指标 | 更好的模型选择 |
| **P3** | 尝试 LoRA 风格 Adapter | 需要修改 ViT |

---

## 6. 下一步行动

1. **立即执行**：运行 `python tools/comprehensive_eval.py`
2. **短期**：实现 `CrossChannelAdapter` 并重新训练
3. **中期**：修改 `train.py` 加入多指标验证

需要我实现哪个方案？
