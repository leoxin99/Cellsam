# CellSAM 项目方案 (Project Blueprint)

> **文档类型**: 项目总览 (AI 必读)
> **最后更新**: 2026-02-05
> **当前阶段**: 阶段3 - 实例级训练优化 (Instance-level Training)

---

## 项目状态仪表板

### 整体进度
```
阶段1 数据准备       [████████████████████] 100%  ✅ 完成
阶段2 检测优化       [████████████████████] 100%  ✅ 完成 (Hybrid DAPI+Actn2)
阶段2.5 三通道适配   [████████░░░░░░░░░░░░]  40%  🔄 Semantic Mapper + Adapter 已完成
阶段3 评估验证       [████░░░░░░░░░░░░░░░░]  20%  ⏳ 待全量评估
阶段4 论文结果       [░░░░░░░░░░░░░░░░░░░░]   0%  ⏳ 待开始
```

### 关键指标
| 指标 | 当前值 | 目标值 | 状态 |
|-----|-------|-------|------|
| **Detection (DAPI)** | F1=78% | - | ✅ E23 修复后 |
| **Semantic Dice** | 0.7595 | - | ⚠️ 无意义(见下) |
| **Instance Dice** | 0.03 (E25) | 0.50+ | ❌ 需要Instance训练 |
| **Instance PQ** | 0.00 (E25) | 0.30+ | ❌ 需要Instance训练 |

### ⚠️ 关键发现 (2026-02-05)

**之前所有实验的 Semantic Dice 无意义**:
- 训练用 `target = (mask > 0)` 合并所有细胞
- 导致模型学习预测大 blob 而非单细胞
- Instance Dice (每细胞) 仅 0.03

**已修复**: Instance-level training with box clipping

### 当前待训练实验
| 实验ID | 配置 | 阶段 | 方案 |
|--------|------|------|------|
| E29 | bf_instance_p1_20260205.yaml | Phase 1 | BF单通道 |
| E30 | adapter_instance_p1_20260205.yaml | Phase 1 | Semantic Adapter |
| E31 | bf_instance_p2_20260205.yaml | Phase 2 | BF + 全部Loss |
| E32 | adapter_instance_p2_20260205.yaml | Phase 2 | Adapter + 全部Loss |

---

## ⚠️ 环境配置 (CRITICAL)

**必须激活 conda 环境才能使用 GPU：**

```bash
conda activate cellsam
```

| 项目 | 值 |
|------|-----|
| **Conda 环境** | `cellsam` |
| **CUDA** | 12.4 |
| **PyTorch** | GPU 版本 |
| **训练位置** | 本地 (非 ALICE) |

> ⚠️ **重要**: 如果不激活环境，可能会使用系统 Python 导致 `CUDA not available` 错误。

### AI 工作规范

#### 🔴 审查制度 (二次验证)

**所有重大代码/方案变更必须经过三轮验证**:

| 轮次 | 执行者 | 内容 |
|------|--------|------|
| **第1轮** | AI 设计 | 提出方案，创建设计文档 |
| **第2轮** | AI 自检 | 检查方案完整性、代码正确性 |
| **第3轮** | 用户审批 | 用户确认后才能执行实施 |

**适用范围**:
- 新 Loss 函数
- 新数据增强
- 模型架构更改
- 评估流程更改

#### 🔴 实验记录要求

**每个新实验/训练都必须记录到 `docs/experiments_log.md`**:

| 必填项 | 示例 |
|--------|------|
| 实验ID | E25 |
| 日期 | 2026-02-04 |
| 实验名称 | 学习率消融 lr=5e-5 |
| 配置 | lr_5e-5.yaml |
| SLURM Job | 899581 |
| 关键变量 | lr=5e-5, patience=15 |
| 结果 | Val Dice=X.XX, Val PQ=X.XX |
| 结论 | 待完成 |

> **先方案后执行**: AI 在执行任何重大操作前，必须先提出方案供审批。

---

## 代码架构

```
src/
├── detection/           # ✅ 已完成
│   └── dapi.py          # Hybrid DAPI+Actn2 检测 (v3)
│                        # - detect_nuclei (min_area=3000)
│                        # - merge_close_nuclei (1.2x diameter)
│                        # - detect_with_adaptive_box (Z-线引导)
├── inference/           # ✅ 已完成
│   ├── postprocess.py   # 6步边界平滑
│   ├── visualize.py     # 图着色
│   └── pipeline.py      # run_sam_inference()
├── comparison/          # 参考实现
│   └── sarcgraph_pipeline/
├── losses/
│   └── combined.py      # 损失函数 (Updated 2026-02-05)
│                        # - DiceLoss, BCELoss (基础)
│                        # - BoundaryLoss, AJILoss (Phase 1)
│                        # - TopologyLoss, SizeLoss, ContourLoss (Phase 2)
└── train.py             # 主训练入口 (Updated 2026-02-05)
                         # - Instance-level target (cell_id)
                         # - Box clipping for pred/target
                         # - Instance Dice validation

tools/
├── visualize_detection_comparison.py  # Napari 检测对比
├── analyze_stats_final.py             # 参数统计脚本
└── run_inference.py                   # 推理脚本

docs/
├── claude_pipeline_analysis.md  # 三通道设计方案 ⭐
├── dataset_parameters.md        # 数据集参数
├── design_decisions.md          # 设计决策
├── troubleshooting.md           # 常见问题
├── error_log_and_checklist.md   # ⭐ 错误归纳 + 训练前检查清单
├── alice_quick_reference.md     # ALICE HPC 快速参考
├── code_inventory.md            # 代码清单和归档状态
└── archive/                     # 过时文档归档
```

---

## 阶段性任务清单

### 阶段1: 数据准备 ✅
- [x] 下载 Allen 数据集 (478 张 TIFF)
- [x] 验证通道映射 (Ch0=BF, Ch1=Actn2, Ch4=DAPI, Ch9=GT)
- [x] GT 统计分析 (5173 细胞, P1=40836, P99=513928)
- [x] 数据划分 (Dev=50, Train=400, Test=78)

### 阶段2: 检测优化 ✅
- [x] DAPI 核检测 (Otsu + 形态学)
- [x] 智能双核合并 (1.2x 直径, ratio<3.0)
- [x] 边缘过滤 (50px, 误删率 1.3%)
- [x] Z-线自适应框 (detect_with_adaptive_box)
- [x] Napari 可视化验证

### 阶段2.5: 三通道模型适配 🔄
- [x] **[P0]** 语义通道映射 `SemanticChannelMapper` (R=Actn2, G=BF, B=DAPI)
- [x] **[P0]** Channel Adapter 实现 `IndependentChannelAdapter` (3×3, ReLU)
- [ ] **[P1]** 训练 Mask Decoder (冻结 ViT) ← 下一步
- [ ] **[P1]** 验证三通道 vs 单通道效果

### 阶段3: 评估验证 ⏳
- [ ] 完整测试集评估 (78 张)
- [ ] 消融实验

### 待实现任务 (Future Tasks)
- [ ] **学习率消融实验**: 测试 lr=5e-5, 1e-4, 2e-4 对比 (配置已创建: `lr_5e-5.yaml`, `lr_2e-4.yaml`)
- [x] **Actn2 区域掩码 (方案B)**: ✅ 已实现 `use_actn2_mask` 参数
- [ ] **Actn2 训练时约束 (方案C)**: 在 loss 中添加 Actn2 边界惩罚
- [ ] **部分解冻 Encoder**: 只训练 ViT 最后 2-4 层
- [ ] **边界增强数据增强**: GridDistortion, 边界扰动 (当前已有 ElasticTransform)

### 最近更新
- **PQ 早停**: `train.py` 支持 `use_pq_early_stop: true` 配置
- **Actn2 掩码**: `comprehensive_eval.py` 支持 `use_actn2_mask=True`
- **A100 训练**: Job 899581 正在运行 (3ch_semantic_adapter, bf_adapter)

---

## 关键决策速查

| 决策 | 选择 | 理由 | 来源 |
|------|------|------|------|
| **检测方案** | Hybrid DAPI+Actn2 | 定位+形状 | `dapi.py` |
| **边缘过滤** | 50px | 误删 1.3% | `analyze_stats_final.py` |
| **双核合并** | 1.2x 直径 | 防止误合并邻居 | `dapi.py` |
| **核面积阈值** | ≥3000px | 过滤碎屑 | Dev Set 统计 |
| **三通道输入** | 语义映射+Adapter | 适配预训练 ViT | `claude_pipeline_analysis.md` |
| 训练框 | GT 框 | 解耦训练 | `design_decisions.md` |
| 冻结策略 | 仅训练 Decoder | 防过拟合 | `design_decisions.md` |

---

## 📚 新 AI 必读清单 (Required Reading)

**只需阅读本文档即可开始工作。** 如需深入了解，按需查阅：

| 优先级 | 文档 | 用途 |
|--------|------|------|
| **P0** | `CLAUDE.md` (本文件) | 项目总览、任务清单、关键决策 |
| **P0** | [error_log_and_checklist.md](docs/error_log_and_checklist.md) | ⚠️ **训练前必读** - 错误归纳 + 检查清单 |
| P1 | [claude_pipeline_analysis.md](docs/claude_pipeline_analysis.md) | 三通道设计详细方案 |
| P1 | [dataset_parameters.md](docs/dataset_parameters.md) | 数据集统计和参数 |
| P2 | [design_decisions.md](docs/design_decisions.md) | 设计决策的"为什么" |
| P2 | [experiments_log.md](docs/experiments_log.md) | 完整实验历史 |
| P2 | [alice_quick_reference.md](docs/alice_quick_reference.md) | ALICE 登录/训练快速参考 |

---

## 📊 关键实验历史 (Experiment Summary)

| 实验 | 日期 | 内容 | 结果 |
|------|------|------|------|
| **E01** | 01-08 | 类别不平衡修复 | Dice 0→0.52 ✅ |
| **E03** | 01-08 | DAPI 核检测方案 | F1=0.750 ✅ |
| **E12** | 01-11 | 边界损失微调 | **PQ↑265%, 当前最佳** ⭐ |
| E15b | 01-15 | 多通道 BF+DAPI+Actn2 | 劣于 E12 ❌ |
| **E18** | 01-23 | SarcGraph 检测对比 | F1↑7.4% ✅ |

> 完整记录: [anti_test/experiments_log.md](anti_test/experiments_log.md)

---

## 🚀 训练前必须执行 (CRITICAL)

**每次训练前必须运行验证脚本：**

```bash
conda activate cellsam
python tools/verify_training_config.py
```

验证通过后才能开始训练！详见 [错误归纳与检查清单](docs/error_log_and_checklist.md)

### 当前训练任务 (4 个消融实验)

| 实验 | 配置文件 | 目的 |
|------|----------|------|
| E1 | `bf_baseline_v2.yaml` | 修复后基线 |
| E2 | `boundary_enhanced.yaml` | 边界 Loss=0.5 |
| E3 | `3ch_no_adapter.yaml` | 3通道无Adapter |
| E4 | `3ch_semantic_adapter.yaml` | 3通道+Adapter |

```bash
# ALICE 上执行
sbatch scripts/train_ablation_v2.sh
```

---

## 更新日志

---

## 常见问题

| 问题 | 解决 | 详情 |
|------|------|------|
| Dice=0 | 边界框内损失 + pos_weight | [详情](docs/troubleshooting.md#q2) |
| 边界锯齿 | 6步平滑管道 | [详情](docs/troubleshooting.md#q5) |
| GPU OOM | batch_size=2, AMP | [详情](docs/troubleshooting.md#q3) |

完整 FAQ: [docs/troubleshooting.md](docs/troubleshooting.md)

---

*此文档由 AI 助手自动维护，每次重要进展后更新*
