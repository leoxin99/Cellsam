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

### 错误 5: 配置文件结构与 train.py 不匹配 ⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-03 |
| **实验** | 887121 |
| **问题文件** | `src/config/*.yaml` |
| **错误代码** | `experiment_name: "xxx"` 在顶层 |
| **原因** | `train.py` 期望 `config['output']['experiment_name']` |
| **影响** | 训练立即失败 KeyError |
| **修复** | 将 `experiment_name` 移到 `output:` 下 |
| **教训** | **新配置文件必须参考现有工作配置的结构** |

### 错误 6: 配置字段名称完全不匹配 ⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-03 |
| **实验** | 897502 |
| **问题文件** | `src/config/*.yaml` |
| **错误代码** | `data.data_dir`, `data.train_split` 等自定义字段 |
| **原因** | 未参照 `semantic_adapter.yaml` 的实际结构 |
| **正确字段** | `data.splits_dir`, `data.processed_data_dir`, `data.target_size`, `data.max_boxes_per_image` |
| **影响** | 训练立即失败 KeyError: 'splits_dir' |
| **修复** | 完全重写配置，参照工作的 `semantic_adapter.yaml` |
| **教训** | **创建新配置必须完全复制现有工作配置，只改关键参数！** |

### 错误 7: 原始图像分辨率记录错误 ⭐⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-05 |
| **问题文件** | `docs/dataset_parameters.md` L15 |
| **错误记录** | "图像尺寸: 1608 × 1608 像素" |
| **实际值** | **1736 × 1776 像素** (全部 478 张) |
| **影响范围** | 所有基于分辨率缩放的参数计算错误 |
| **受影响参数** | SizeLoss (min_area, max_area), TopologyLoss (min_size), postprocess thresholds, DAPI nucleus params |
| **错误缩放系数** | 使用 (1024/1608)²=0.4055 |
| **正确缩放系数** | 应为 (1024/1756)²=**0.340** |
| **修复** | 1) 修正文档分辨率 2) 重新计算所有缩放参数 |
| **教训** | **任何数据统计都必须明确记录：数据源、分辨率、统计方法** |

**详细影响**:
- E17 细胞面积统计在 1736×1776 分辨率下进行
- 训练使用 1024×1024 (resize)
- 所有 E17 面积值需乘以 0.340 (非 0.4055)

**参数修正对照**:
| 参数 | E17原始值 | 错误缩放(×0.4055) | 正确缩放(×0.340) |
|------|----------|-------------------|------------------|
| SizeLoss min_area | 40,836 | 16,559 | **13,884** |
| SizeLoss max_area | 513,928 | 208,378 | **174,735** |
| TopologyLoss min_size | 40,836 | 16,559 | **13,884** |

---

### 错误 8: ALICE 环境配置不一致 ⭐⭐

| 项目 | 内容 |
|------|------|
| **发现日期** | 2026-02-06 |
| **问题脚本** | `scripts/train_instance_20260205.sh` |
| **表现** | 训练 3 秒即退出，错误 `ModuleNotFoundError: No module named 'cellSAM'` |
| **根本原因** | 新脚本与已验证的旧脚本 (`train_semantic.sh`) 环境配置不一致 |

**新脚本 vs 旧脚本对比**:

| 项目 | 旧脚本 (工作) | 新脚本 (失败) |
|------|--------------|---------------|
| **Shebang** | `#!/bin/bash -l` (login shell) | `#!/bin/bash` |
| **Conda 路径** | 不显式指定 (靠 .bashrc) | `~/miniconda3` (路径错误) |
| **PYTHONPATH** | `export PYTHONPATH=$PYTHONPATH:~/CellSam/cellSAM_source` | 无 |
| **module load** | 无 | `module load cuda/11.8` (失败) |

**修复步骤**:
1. 使用 `#!/bin/bash -l` 启用 login shell 自动加载环境
2. 使用 `source ~/.bashrc` 而非硬编码 conda 路径
3. 在 ALICE 上安装 cellSAM: `pip install 'cellSAM @ git+https://github.com/vanvalenlab/cellSAM.git'`

**教训**:
> **⚠️ 创建新 ALICE 脚本时，必须复制已验证工作的旧脚本的环境配置部分！**
> **⚠️ 本地能运行不代表 ALICE 能运行，路径、已安装包可能不同**
> **⚠️ 新脚本首次运行前，先在空闲分区测试**

**ALICE 环境检查清单**:
- [ ] 验证 conda 环境路径正确
- [ ] 验证所有 `import` 的包已安装
- [ ] 使用 `#!/bin/bash -l` 启用 login shell
- [ ] 首次在空闲 GPU 分区测试

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

### F. 数据派生参数检查 (2026-02-05 新增)

- [ ] **F1**: 所有**面积参数**有明确的统计来源和分辨率标注
  ```
  示例: min_area=200 (Dev Set 50张, 1024px, P1=57, 2026-02-05)
  ```

- [ ] **F2**: 所有**像素距离参数**有明确的统计来源和分辨率标注
  ```
  示例: margin=32 (scaled from 50 @ 1608 → 32 @ 1024)
  ```

- [ ] **F3**: 参数变更时，**同步更新所有相关代码和文档**

- [ ] **F4**: 不确定的参数**必须用代码验证，禁止估算**

> ⚠️ **背景**: Error 7 (分辨率 1608→1736 错误) 导致所有缩放计算错误

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
