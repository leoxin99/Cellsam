# 项目交接与下一步工作指南 (Project Handoff & Next Steps)

> **致下一位 AI 助手**:
> 本文档定义了 CellSAM 项目在进入"论文完善阶段"的关键任务。
> **核心目标**: 确保实验的严谨性、可复现性，并与之前的开发思路（Opus 模型设定）保持对齐。
> 当前状态: 基本管线已跑通 (Detection F1=0.75, Segmentation Dice=0.82, PQ@0.5=0.087)。

---

## 📅 核心任务清单 (Priority Tasks)

### 1. 数据集标准化 (Dataset Standardization)
**现状**:之前实验可能采用了随机划分，导致不同实验间的可比性受限。
**任务**: 
- [ ] **固定划分**: 将所有 Annotated TIFF 样本显式划分为 `Train` (70%), `Val` (15%), `Test` (15%)。
- [ ] **固化文件**: 创建 `data/splits/train_ids.txt`, `val_ids.txt`, `test_ids.txt`。
- [ ] **强制执行**: 修改 `augmented_dataset.py` 和训练脚本，强制读取这些 ID 列表，而不是运行时随机划分。
- **目的**: 确保 Model A 和 Model B 的对比是"苹果对苹果"的，消除数据划分带来的随机性。

### 2. 训练规模与数据探索 (Scale Exploration)
**现状**: 当前使用 ~50 张图像，训练 20 Epochs。
**任务**:
- [ ] **数据量分析**: 绘制 `Training Loss` 和 `Val Dice` 曲线，判断是否过拟合或欠拟合。
- [ ] **轮次增加**: 尝试增加 Epochs (如 50, 100) 观察性能边界。
- [ ] **数据增强**: 评估当前增强策略是否足够，考虑增加 brightness/contrast 增强以适应不同批次数据。

### 3. 损失函数深度优化 (Loss Function Optimization)
**现状**: 刚引入 `Boundary Loss (0.3)`，PQ 提升显著。
**任务**:
- [ ] **理论分析**: 在 `methods_draft.md` 中详细阐述 Boundary Loss 对 CellSAM 的意义（解决 touching cells 分割难题）。
- [ ] **权重消融**: 尝试不同的 boundary_weight (0.1, 0.3, 0.5, 0.7) 寻找最优解。
- [ ] **新损失探索**: 考虑引入 `Hausdorff Loss` 或 `Lovasz-Softmax Loss` 进一步优化拓扑结构。

### 4. 生物学先验集成 (Biological Priors / SarcGraph)
**现状**: 目前仅利用 DAPI (核) 和 Brightfield (形态)。
**任务**:
- [ ] **Actn2 引导**: Actn2 通道 (Channel 1) 的高亮区域对应肌节。
- [ ] **边界假设**: 肌节的不连续处往往是细胞边界。
- [ ] **实现方案**: 将 Actn2 信号作为额外的 Prompt 输入 SAM，或作为后处理的约束条件。
- **参考**: SarcGraph 论文中的 z-disk 检测算法。

### 5. 基准模型对比 (Benchmarking Guidelines)
**现状**: 缺乏系统性的 SOTA 对比。
**任务**: 建立以下 Benchmark 表格：
| 模型 (Model) | 预训练 (Pretrain) | 微调 (Finetune) | 备注 |
|-------------|-------------------|----------------|------|
| **DeepLabV3+** | ImageNet | Allen Data | Allen 实验室原始方案 (Baseline) |
| **CellFinder** | - | - | 传统方法 (已证明失败, F1=0.01) |
| **CellSAM (Ours)** | SAM-B | Mask Decoder | 当前方案 |
| **CF + SAM** | SAM-B | CellFinder detection + SAM | 验证检测器影响 |
| **Fully Fine-tuned** | SAM-B | Image Encoder + Decoder | 验证全量微调是否更好 |
| **Cellpose / StarDist** | General | Allen Data | 强力竞品对比 |

### 6. 代码库优化 (Code Simplification via Claude)
**现状**: 经过多轮快速迭代，存在冗余代码 (如 `train_simple.py`, `train_expanded.py` 并存)。
**任务**:
- [ ] **Code Simplifier**: 调用 Claude 的代码优化能力，重构项目结构。
- [ ] **目标**: 
    - 统一为一个 `train.py` (支持 config 参数切换模式)。
    - 将散落在 `anti_test` 中的实用函数移入 `src/utils`。
    - 清理未使用的 import 和死代码。

### 7. 对比与消融实验设计 (Ablation Study)
**任务**: 设计严谨的消融实验矩阵：
- **Ablation 1 (Prompt)**: Box (DAPI) vs Box (CellFinder) vs Points.
- **Ablation 2 (Loss)**: CE vs Dice vs Boundary vs Combined.
- **Ablation 3 (Components)**: 验证后处理 (Closing/FillHoles) 的贡献。

---

## 🛠️ 下一步行动建议
1.  优先执行 **Task 1 (数据集标准化)**，这是所有后续对比的基石。
2.  执行 **Task 6 (代码清理)**，为大规模实验做准备。
3.  按顺序执行 **Task 5 (基准对比)**，补充论文所需的表格数据。
