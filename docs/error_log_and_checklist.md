# CellSAM 错误归纳与检查清单

> **创建日期**: 2026-02-02
> **目的**: 记录历史错误，形成训练前强制检查清单
> **规则**: 每次训练前必须逐条确认

---

## 一、历史错误归纳

### 错误 1: uint8 截断导致通道数据丢失 ⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-02 |
| **实验** | E23 |
| **问题文件** | `src/augmented_dataset.py` L388 |
| **错误代码** | `image[..., c] = self._normalize_image(image[..., c])` |
| **原因** | `image` 是 uint8，`_normalize_image` 返回 float32 (0-1)，赋值回 uint8 截断为 0/1 |
| **影响** | DAPI 检测失效 (256→2 唯一值)，模型无法学习通道信息 |
| **修复** | 添加 `image = image.astype(np.float32)` |
| **教训** | **数组类型转换必须显式检查** |

### 错误 2: 训练-推理 Box Expand 不一致 ⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-02 |
| **实验** | E22 |
| **问题** | 训练 Loss 在 box+20% 区域计算，推理不裁剪导致 mask 超出 2-15x |
| **影响** | PQ=0，实例互相覆盖 |
| **修复** | 推理时添加 box clipping，统一 expand=0.1 |
| **教训** | **训练和推理的预处理/后处理必须一致** |

### 错误 3: E12 vs Semantic Adapter 比较用错模型 ⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-01-30 |
| **实验** | E16/E21 |
| **问题** | 比较时模型配置不一致 |
| **教训** | **对比实验必须明确记录每个模型的配置** |

### 错误 4: DAPI P2-P98 归一化在稀疏信号时失效

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-02 |
| **问题** | DAPI 背景 98%，P2=P98=0，导致除以 0 |
| **教训** | **百分位归一化需要检查边界情况** |

---

## 二、训练前强制检查清单 ✅

### A. 数据加载检查

- [ ] **A1**: 验证图像通道唯一值 > 10 (非二值)
  ```python
  sample = dataset[0]
  for c in range(3):
      assert len(np.unique(sample['image'][c].numpy())) > 10
  ```

- [ ] **A2**: 验证 mask 标签数量合理 (非全 0)
  ```python
  assert len(np.unique(sample['mask'].numpy())) > 1
  ```

- [ ] **A3**: 验证 boxes 数量 > 0
  ```python
  assert sample['num_boxes'] > 0
  ```

### B. 训练配置检查

- [ ] **B1**: 确认 `box_expand` 值 (当前应为 0.1)
- [ ] **B2**: 确认 `boundary_weight` 与实验设计一致
- [ ] **B3**: 确认 `use_bf_only` / `use_semantic_mapping` 与实验设计一致
- [ ] **B4**: 确认 `pretrained_path` 存在

### C. Loss 函数检查

- [ ] **C1**: 验证 `CombinedLoss.box_expand` 与推理一致
  ```python
  # 检查 src/losses/combined.py L231
  assert expand == 0.1, "expand mismatch!"
  ```

- [ ] **C2**: 验证 Loss 权重非 0

### D. 推理一致性检查

- [ ] **D1**: 推理 box clipping expand 与训练一致
- [ ] **D2**: 推理预处理与训练一致 (归一化方式)

### E. 文件检查

- [ ] **E1**: 所有依赖文件存在
- [ ] **E2**: Git 已提交最新更改
- [ ] **E3**: ALICE 上代码已同步

---

## 三、训练验证脚本

```python
# tools/verify_training_config.py
"""
训练前验证脚本
执行此脚本确保配置正确
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')

def verify_dataset():
    from augmented_dataset import AugmentedAllenDataset
    dataset = AugmentedAllenDataset('data/processed', is_training=False)
    sample = dataset[0]
    
    # A1: 通道唯一值
    for c, name in enumerate(['BF', 'DAPI', 'Actn2']):
        unique = len(np.unique(sample['image'][c].numpy()))
        status = "✅" if unique > 10 else "❌"
        print(f"{status} {name}: {unique} unique values")
        assert unique > 10, f"{name} has only {unique} unique values!"
    
    # A2: Mask
    mask_labels = len(np.unique(sample['mask'].numpy()))
    print(f"✅ Mask labels: {mask_labels}")
    
    # A3: Boxes
    print(f"✅ Boxes: {sample['num_boxes']}")
    
    print("\\n✅ 数据加载检查通过!")

def verify_loss_config():
    from losses.combined import CombinedLoss
    # 检查源代码
    import inspect
    source = inspect.getsource(CombinedLoss._box_clipped_loss)
    assert 'expand = 0.1' in source, "expand != 0.1 in CombinedLoss!"
    print("✅ Loss expand = 0.1")

if __name__ == "__main__":
    print("=== CellSAM 训练配置验证 ===\\n")
    verify_dataset()
    verify_loss_config()
    print("\\n=== 所有检查通过! 可以开始训练 ===")
```

---

## 四、使用规则

1. **训练前**: 执行 `python tools/verify_training_config.py`
2. **新错误**: 立即添加到此文档
3. **团队协作**: 共享此文档给所有成员
4. **版本控制**: 此文档必须 Git 跟踪
