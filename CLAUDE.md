# CellSAM 项目方案 (Project Blueprint)

> **文档类型**: 项目总览 (AI 必读)
> **最后更新**: 2026-02-13
> **当前阶段**: Phase 2 Step 3 完成 → Step 4 P2-A 训练

---

## 项目状态仪表板

### 整体进度
```
阶段1 数据准备       [████████████████████] 100%  ✅ 完成
阶段2 检测优化       [████████████████████] 100%  ✅ 完成 (Hybrid DAPI+Actn2)
阶段2.5 三通道适配   [████████░░░░░░░░░░░░]  40%  🔄 Semantic Mapper + Adapter 已完成
Phase 1 Loss优化     [████████████████████] 100%  ✅ 完成 + Test锁定
Phase 2 结构改进     [████████████░░░░░░░░]  60%  🔄 Step 3 完成, Step 4 训练中
阶段4 论文结果       [░░░░░░░░░░░░░░░░░░░░]   0%  ⏳ 待开始
```

### 关键指标 (Phase 1 Test 锁定, 2026-02-11)
| 指标 | Oracle(test) | E2E(test) | 状态 |
|-----|-------------|-----------|------|
| **BM-1to1 Dice** | **0.6954** | 0.5446 | ✅ Phase 1 锁定 |
| **PQ** | **0.4641** | 0.1719 | ✅ vs BF_Baseline +704% |
| **AJI** | **0.5195** | 0.3181 | ✅ vs BF_Baseline +82% |
| **Semantic Dice** | 0.7566 | 0.6006 | ✅ 稳定 |

> **历史检测里程碑**: DAPI 检测 F1≈78% (E23 修复后，非本轮 test 同步测得)

### 当前最优模型
| 项目 | 值 |
|------|----|
| Checkpoint | `checkpoints/E_phase1_rebalance_l4/best_model.pt` |
| 训练配置 | `src/config/phase1_rebalance_l4.yaml` |
| 训练平台 | ALICE L4 (Job 974531) |
| Best Epoch | 49/50 |

### 下一步: Phase 2 Step 4 — P2-A 训练
- **Step 3 已完成**: L_neighbor + L_overlap 实现 + Codex 审核通过
- **当前**: 在 Alice 提交 P2-A 训练 (`scripts/train_phase2a.sh`)
- **配置**: `src/config/phase2a_neighbor_overlap.yaml`
- **验证**: 梯度 12/12 + 回归 10/10 通过
- **Alice 操作**: `git pull && sbatch scripts/train_phase2a.sh`

---

## ⚠️ 环境配置 (CRITICAL)

**必须激活 conda 环境才能使用 GPU：**

```bash
conda activate cellsam
```

| 项目 | 值 |
|------|-----|
| **Conda 发行版** | Miniforge3 (系统级: `/easybuild/software/Miniforge3/`) |
| **Conda 环境** | `cellsam` (pytorch 2.1.2 + pytorch-cuda=12.1) |
| **CUDA Module** | `CUDA/12.1.1` ⚠️ 不再是 `cuda/11.8` |
| **训练位置** | ALICE HPC (L4/A100) + 本地 (评估) |

> ⚠️ **Alice 环境踩坑记录 (2026-02-13)**:
> Alice 集群会定期维护更新，可能导致：
> 1. **CUDA Module 名称变化**: `cuda/11.8` → `CUDA/12.1.1`（大小写也变了）
> 2. **Conda 路径变化**: `~/miniconda3` → 系统级 Miniforge3（用户 home 下不再有 conda）
> 3. **Conda activate 脚本冲突**: MKL 环境变量未定义 + `set -u` = 脚本静默退出
>
> **SLURM 脚本最佳实践**:
> - 不要 `source ~/miniconda3/...`，用 `eval "$(conda shell.bash hook)"`
> - `set -eo pipefail` 放在最前面，`set -u` 放在 `conda activate` **之后**
> - Login 节点能跑通的命令不代表 SLURM 脚本也能跑通（初始化路径不同）
### 核心文档状态 (2026-02-13)

| 文档 | 状态 | 用途 |
|------|------|------|
| `docs/inference_standard.md` | 🟢 Active | 推理口径 SSOT |
| `docs/code_inventory.md` | 🟢 Active | 代码入口速查 |
| `docs/experiments_log.md` | 🟢 Active | 实验流水账 |
| `docs/dataset_parameters.md` | 🟢 Active | 数据集参数 |
| `docs/naming_convention.md` | 🟢 Active | 命名规范 |
| `docs/error_log_and_checklist.md` | 🟢 Active | 错误归纳 + 检查清单 |
| `docs/alice_quick_reference.md` | 🟢 Active | Alice HPC 指南 |
| `docs/boundary_enhancement_design.md` | 🟡 Historical | 早期设计草案 |
| `docs/claude_pipeline_analysis.md` | 🟡 Historical | pipeline 分析 |

> 新对话仅需 `CLAUDE.md` + `inference_standard.md` + `code_inventory.md` 即可定位下一步。

---

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

#### 🔴 禁止估算原则 (2026-02-05 新增)

**AI 绝对不允许估算以下类型的数据**:
1. 图像分辨率、尺寸
2. 任何像素级阈值、面积参数
3. 数据集统计量

**正确做法**:
1. 如果文档无明确记录 → **用代码统计**并展示结果给用户
2. 如果代码无法运行 → **向用户询问**
3. 结果必须标注：**数据来源、统计方法、分辨率、日期**

**违反后果**:
- 参数计算全部错误 (如 Error 7: 分辨率 1608→1736 导致缩放系数错误)

---
---

## 关键文档链接 📚

| 文档 | 用途 | 更新频率 |
|------|------|----------|
| [codex_claude_seg.md](docs/codex_claude_seg.md) | **Codex+Claude 联合文档 (Ch1-16)** ⭐ | 每阶段 |
| [error_log_and_checklist.md](docs/error_log_and_checklist.md) | 历史错误归纳 + 训练前检查清单 | 每次发现错误 |
| [experiments_log.md](docs/experiments_log.md) | 实验记录 (E1-E30+ & Phase1) | 每次实验 |
| [dataset_parameters.md](docs/dataset_parameters.md) | 数据集统计参数 (分辨率、阈值) | 参数变化时 |
| [inference_standard.md](docs/inference_standard.md) | **推理标准** (Best-Match Dice) ⭐ | 推理方法变更时 |
| [naming_convention.md](docs/naming_convention.md) | **命名规范** (模型/实验/检测方案) ⭐ | 新方案时 |
| [boundary_enhancement_design.md](docs/boundary_enhancement_design.md) | Loss 函数设计文档 | 设计变更时 |
| [code_inventory.md](docs/code_inventory.md) | 代码文件清单 + 版本记录 | 新增/修改代码时 |

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
│   ├── core.py          # 统一推理核心 (segment_with_boxes)
│   ├── postprocess.py   # 6步边界平滑
│   ├── visualize.py     # 图着色
│   └── pipeline.py      # run_sam_inference()
├── metrics/
│   └── instance_metrics.py  # 统一指标 (BM-1to1, PQ, AJI, SQ, RQ)
├── losses/
│   └── combined.py      # 损失函数 (Updated 2026-02-13)
│                        # - DiceLoss, BCELoss (基础)
│                        # - BoundaryLoss, AJILoss, ContourLoss (Phase 1)
│                        # - NeighborIntrusionLoss, OverlapMutexLoss (Phase 2)
│                        # - TopologyLoss, SizeLoss (Phase 2 备用)
│                        # - Computability-gated normalization
├── config/              # 实验配置
│   ├── phase1_rebalance_l4.yaml        # ✅ Phase 1 锁定配置
│   ├── phase1_rebalance_a100.yaml
│   └── phase2a_neighbor_overlap.yaml   # 🔄 P2-A 当前配置
└── train.py             # 主训练入口 (Updated 2026-02-13)
                         # - Instance-level target, PQ early stop
                         # - Box shuffle + confidence_map accumulation
                         # - Neighbor/overlap loss data flow

tools/
├── smoke_test_e2e.py              # Oracle(val) 开发评估
├── comprehensive_eval.py          # Oracle(test) 最终评估
├── evaluate_e2e.py                # E2E(test) 部署效果评估
├── test_unified_regression.py     # 10-test 回归测试
├── test_loss_gradients.py         # 12-test 梯度门禁 (Phase 2)
└── run_inference.py               # [DEPRECATED] 旧推理入口，仅兼容提示

scripts/
├── train_phase1_full.sh   # ALICE A100 SLURM 脚本
├── train_phase1_l4.sh     # ALICE L4 SLURM 脚本
└── train_phase2a.sh       # 🔄 P2-A SLURM 脚本 (含 gradient gate)

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
- [x] GT 统计分析
- [x] 数据划分 (Train=334, Val=71, Test=73)

### 阶段2: 检测优化 ✅
- [x] DAPI 核检测 + 双核合并 + 边缘过滤
- [x] Z-线自适应框 (detect_with_adaptive_box)

### Phase 1: Loss 权重重平衡 + PQ 早停 ✅ (2026-02-10~11)
- [x] boundary_weight 0.5→1.5, contour_weight OFF→0.3, pos_weight 10→2
- [x] PQ 早停 (patience=15)
- [x] ALICE 双 GPU 训练 (L4 + A100)
- [x] Oracle(val,n=30) + Oracle(test,73) + E2E(test,73) 评估
- [x] **Phase 1 已锁定** — 不再调参

### Phase 2: 结构性改进 🔄
- [x] Step 1: SQ/RQ 评估工具补全
- [x] Step 2: Loss 基础设施修复 (归一化 + 可微 Contour/Topology)
- [x] Step 3: L_neighbor + L_overlap 实现 + Codex 审核通过
- [/] **Step 4: P2-A 训练** (`phase2a_neighbor_overlap.yaml`)
- [ ] Step 5: 评估 + 决定是否 P2-B
- [ ] 三通道 Adapter 对比实验 (Phase 3)

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
| **E12** | 01-11 | 边界损失微调 | PQ↑265% |
| **E29** | 02-05 | Instance-level 基线 | BM-1to1=0.593, PQ=0.326 |
| **Phase 1** | 02-10 | Loss 重平衡 + PQ 早停 | **Oracle PQ=0.464, BM=0.695** ⭐ |

> 完整记录: [experiments_log.md](docs/experiments_log.md), [codex_claude_seg.md](docs/codex_claude_seg.md)

---

## 🚀 训练前必须执行 (CRITICAL)

**每次训练前必须运行验证脚本：**

```bash
conda activate cellsam
python tools/verify_training_config.py
```

验证通过后才能开始训练！详见 [错误归纳与检查清单](docs/error_log_and_checklist.md)

### 评估工具分工

| 工具 | 数据集 | 用途 |
|------|--------|------|
| `smoke_test_e2e.py` | val (30 samples) | 开发阶段快速验证 |
| `comprehensive_eval.py` | **test** (73 samples) | Oracle 最终评估 |
| `evaluate_e2e.py` | **test** (73 samples) | E2E 最终评估 |
| `test_unified_regression.py` | - | 10-test 回归 |

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-02-13 | Fix: CUDA module cuda/11.8 → CUDA/12.1.1 (Alice 系统更新) |
| 2026-02-13 | P2-A 双 GPU 训练提交: L4 (979114) + A100 (979115) |
| 2026-02-13 | Phase 2 Step 3 完成: L_neighbor + L_overlap + computability gating |
| 2026-02-13 | P2-A SLURM 脚本 + 梯度门禁 12/12 + 回归 10/10 |
| 2026-02-11 | Phase 1 完成 + test 锁定评估 + 文档全量更新 |
| 2026-02-10 | ALICE 训练提交 (L4+A100)，统一推理核心 |
| 2026-02-05 | Instance-level 训练修复，E29 基线 |

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
