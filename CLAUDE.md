# CellSAM Project Blueprint

> **Doc type**: Project overview (AI must-read)
> **Last updated**: 2026-03-04
> **Current phase**: T27a PQ=0.643, T28 PQ=0.684, T29 ablation done, T30 LoRA running

---

## Project Status Dashboard

### Experiment Index (see docs/experiments/ for detail docs)

| Experiment | Description | Mean PQ | Status |
|------------|------------|:-------:|:------:|
| T27a | Plan B BF-only decoder-only | **0.643** | completed |
| T28 | Plan B 3ch [BF,Actn2,DAPI] | **0.684** | completed |
| T29a | Official BF [0,0,BF] | 0.642 (s42) | s123 running |
| T29b | Official 3ch [0,DAPI,BF] | 0.665 (s42) | s123 running |
| T29c | Official 3ch+Actn2 [Actn2,DAPI,BF] | 0.685 (s42) | s123 running |
| T30 | LoRA Q/V on encoder (BF-only) | -- | running |
| T31 | Cellpose paper-aligned baseline | -- | planned |

**T12 Loss ablation** (completed): posw=10 >> 2 (+4.1pp), Contour harmful

- **高置信结论**: pos_weight=10 >> 2 (+4.1pp), Contour Loss 有害 (+2.3pp)
- Best Config: posw=10 + contour=off → **PQ=0.484** (4-run mean, 验证完成)

**P2-A (N/O Loss) 已终止**: Fix1-3 均证明 N/O loss 导致 PQ 退化 (详见 `experiments_log.md`)

**T16 Baseline 对比 ✅ (2026-02-22 完成)**:

| Method | Type | PQ | BM-Dice | AJI |
|--------|------|----|---------|-----|
| Cellpose v4 | E2E | 0.000 | 0.053 | 0.025 |
| SAMCell | E2E | 0.000 | 0.008 | 0.004 |
| CellSAM (pretrained) | Oracle | 0.434 | 0.682 | 0.499 | ⚠️ T24 修正: 官方推理路径 |
| SAM ViT-B | Oracle | 0.286 | 0.631 | 0.440 |
| Ours (Best Config) | Oracle | 0.484 | 0.720 | 0.570 |
| MedSAM | Oracle | 0.576 | 0.771 | 0.634 |
| **Ours (T27a Plan B)** | **Oracle** | **0.638** | **0.791** | **—** | **🏆 超越 MedSAM** |
| Ours | E2E | 0.180 | 0.567 | 0.338 |

> ⚠️ MedSAM Oracle > Ours Oracle — 但 MedSAM 无检测能力，Ours 是唯一 E2E 方案

**Box Clipping 消融 (T19-abl)**: with_clip PQ=0.466 > no_clip PQ=0.437 (-6.2%) → clipping 有防御价值

### 📏 实验可视化标准 (强制)

每个实验完成后必须提供:
1. **5 个固定测试样本** napari 截图 (test set 前 5 张)
2. **三通道展示**: BF (灰度) + DAPI (蓝色) + Actn2 (绿色)
3. 预测分割 (Labels 层) + GT 分割 (对比)
4. 指标表: PQ, BM-Dice, Semantic Dice

> 原始 TIFF 通道索引: BF=Ch0, Actn2=Ch1, DAPI=Ch4

### 🔒 新实验 Pre-Flight Checklist

**每次新实验必须执行** `.agent/workflows/new-experiment-checklist.md`:
- Phase 1: 设计 — 找参照 YAML, 只改目标变量
- Phase 2: 本地 dry-run 验证
- Phase 3: ALICE `git add -A` → 确认文件 → 检查日志
- Phase 4: 训练后 napari 可视化 + 文档更新

**当前工作重点**:

| 任务 | 执行者 | 状态 |
|------|---------|------|
| T18 三通道消融 | A2 | ✅ 5/6 done: T18-C PQ=**0.500** (best!), seed123 补跑 Job 1036799 |
| T17 Training Curves | A2 | ✅ 工具完成, Phase1 图 ✅, 待下载 Best Config 日志 |
| T20 Attention 可视化 | A1 | 🔄 脚本就绪, 待 T18 完成后执行 |
| T12 消融 + Best Config | A2 | ✅ 完成 |
| T16 Baseline 对比 | A2 | ✅ 完成 |
| **T11 LoRA Encoder** | **A2** | **⏳ 设计完成, 待 R1 审核 → 实施** |
| 论文文档合并 | A1 | ✅ paper_writing_plan → paper_preparation §7 |

### ⚠️ 已修复: GT 框面积过滤 Bug (2026-02-13)

`_mask_to_boxes_with_ids` 曾用 `max_area_ratio=0.15` 过滤 GT regions，面积 >15% 图像的大细胞被静默丢弃。已删除该过滤，验证 5,173/5,173 GT regions 全部生成框。后续训练自动使用修复代码 (`git pull` 即可)。

2026-02-14 补充: 检测消融评估脚本 (`tools/ablation_dapi_val.py`, `tools/ablation_dapi_params.py`, `tools/ablation_adaptive_params.py`, `tools/ablation_adaptive_improved.py`) 已同步移除 GT `min_area=500` 过滤，避免评估分母被静默改变。

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
> **SLURM 脚本最佳实践**:
> - 不要 `source ~/miniconda3/...`，用 `eval "$(conda shell.bash hook)"`
> - `set -eo pipefail` 放在最前面，`set -u` 放在 `conda activate` **之后**
> - Login 节点能跑通的命令不代表 SLURM 脚本也能跑通（初始化路径不同）
### 核心文档状态 (2026-02-15)

| 文档 | 状态 | 用途 |
|------|------|------|
| `docs/inference_standard.md` | 🟢 Active | 推理口径 SSOT |
| `docs/dapi_detection_design.md` | 🟢 Active | 检测参数 SSOT (DAPI/Adaptive) |
| `docs/code_inventory.md` | 🟢 Active | 代码入口速查 |
| `docs/experiments_log.md` | 🟢 Active | 实验流水账 |
| `docs/dataset_parameters.md` | 🟢 Active | 数据集参数 |
| `docs/naming_convention.md` | 🟢 Active | 命名规范 |
| `docs/error_log_and_checklist.md` | 🟢 Active | 错误归纳 + 检查清单 |
| `docs/alice_quick_reference.md` | 🟢 Active | Alice HPC 指南 |
| `docs/task_backlog.md` | 🟢 Active | 短期/长期待办与完成标准 |
| `docs/progress_timeline_2.13.md` | 🟢 Active | 导师汇报时间线 + 后续计划 |
| `docs/phase2_design.md` | 🟢 Active | Phase 2 设计与执行计划 |
| [`docs/t11_lora_design.md`](docs/t11_lora_design.md) | 🟢 Active | **T11 LoRA Encoder 设计文档** (待 R1 审核) |
| `docs/agent_management.md` | 🟢 Active | **多 Agent 协作管理规范 SSOT** |
| `docs/agent_inbox.md` | 🟢 Active | **Agent 间异步通信信箱** |
| [`docs/technical_qa_2.27.md`](docs/technical_qa_2.27.md) | 🟢 Active | **技术细节 Q&A** (SAM 架构、训练策略、CellSAM 设计) |
| [`docs/paper_preparation.md`](docs/paper_preparation.md) §2.1b | 🟢 Active | **CellSAM 架构分析**: model vs model_cp 权重对比, Prompt Encoder 结构与微调价值 |
| `docs/temp_reviews/` | 🟠 Temp | 审核报告 (合并进 SSOT 后可删) |
| `docs/codex_claude_seg.md` | 🟡 Historical | A1/A2 联合工作台 (Phase 0-2 全链路) |
| `docs/codex_claude_arrange.md` | 🟡 Historical | 文件整理方案 (2026-02-10) |
| `docs/phase1_design.md` | 🟡 Historical | Phase 1 实施记录 |
| `docs/boundary_enhancement_design.md` | 🟡 Historical | 早期设计草案 |
| `docs/claude_pipeline_analysis.md` | 🟡 Historical | pipeline 分析 |

> 新 Agent 必读: `CLAUDE.md` → `docs/agent_management.md` → `docs/task_backlog.md` → 按角色读对应 SSOT 文档。

### 🔄 多 Agent 协作模式

> 详细规范见 [`docs/agent_management.md`](docs/agent_management.md)（Agent 清单、职能边界、通信协议、文件所有权、并发防护）。

| ID | 名称 | 角色 | 职责摘要 |
|----|------|------|----------|
| A1 | **Codex** | 实施 Agent | 代码实现、实验执行 |
| A2 | **Claude** | 实施 Agent | 代码实现、设计方案 |
| R1 | **Reviewer** | 审核 Agent | 第三方审核、SSOT 回填 |

**核心约束**: 审核 Agent 回填文档前须确认实施 Agent 无未 commit 修改 (A 模式)。

### 参数记录分工 (防重复)

| 参数类型 | SSOT 文档 | 辅助文档 | 备注 |
|------|------|------|------|
| 推理参数 (`InferenceConfig`) | `docs/inference_standard.md` | `docs/codex_claude_seg.md` | 以代码 `src/inference/core.py` 为最终真值 |
| 检测参数 (DAPI/Adaptive) | `docs/dapi_detection_design.md` | `docs/experiments_log.md` | 设计文档写“当前锁定值”，实验日志写“每次实验记录” |
| 数据统计阈值来源 (P1/P99 等) | `docs/dataset_parameters.md` | `docs/experiments_log.md` | 只在一个地方维护统计表 |
| 阶段进展/下一步 | `docs/task_backlog.md` | `CLAUDE.md`, `docs/progress_timeline_2.13.md` | Backlog 记录可执行待办，CLAUDE 仅保留摘要 |

> 约束: 参数表只在 SSOT 文档维护；其余文档仅引用“值 + 链接”，不复制整表。
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
| [codex_claude_seg.md](docs/codex_claude_seg.md) | **Codex+Claude 联合文档 (持续更新)** ⭐ | 每阶段 |
| [error_log_and_checklist.md](docs/error_log_and_checklist.md) | 历史错误归纳 + 训练前检查清单 | 每次发现错误 |
| [experiments_log.md](docs/experiments_log.md) | 实验记录 (E1-E30+ & Phase1) | 每次实验 |
| [dataset_parameters.md](docs/dataset_parameters.md) | 数据集统计参数 (分辨率、阈值) | 参数变化时 |
| [inference_standard.md](docs/inference_standard.md) | **推理标准** (Best-Match Dice) ⭐ | 推理方法变更时 |
| [naming_convention.md](docs/naming_convention.md) | **命名规范** (模型/实验/检测方案) ⭐ | 新方案时 |
| [progress_timeline_2.13.md](docs/progress_timeline_2.13.md) | 导师汇报材料 + 2.13 时间线 | 里程碑更新时 |
| [phase2_design.md](docs/phase2_design.md) | Phase 2 方案与实验路线 | Phase 2 变更时 |
| [phase1_design.md](docs/phase1_design.md) | Phase 1 实施记录 | 回溯 Phase 1 时 |
| [boundary_enhancement_design.md](docs/boundary_enhancement_design.md) | Loss 函数设计文档 | 设计变更时 |
| [code_inventory.md](docs/code_inventory.md) | 代码文件清单 + 版本记录 | 新增/修改代码时 |

---

## 代码架构

```
src/
├── detection/           # ✅ 已完成
│   └── dapi.py          # Hybrid DAPI+Actn2 检测 (v4)
│                        # - detect_nuclei (default: min/max=200/10000)
│                        # - merge_close_nuclei (1.2x diameter)
│                        # - detect_with_adaptive_box (default search_radius=256)
│                        # - DAPI/Adaptive 参数待 val→test 统一锁定
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
├── train_phase2a.sh       # 🔄 P2-A L4 SLURM 脚本 (含 gradient gate)
└── train_phase2a_a100.sh  # 🔄 P2-A A100 对照 SLURM 脚本

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

### Phase 2: 结构性改进 ⚠️ (P2-A 终止)
- [x] Step 1: SQ/RQ 评估工具补全
- [x] Step 2: Loss 基础设施修复 (归一化 + 可微 Contour/Topology)
- [x] Step 3: L_neighbor + L_overlap 实现 + Codex 审核通过
- [x] **Step 4: P2-A 训练 — ❗ 终止** (Fix1-3 均 PQ 退化)
- [x] **Step 4.5: 检测参数锁定** — DAPI F1=0.8033, Adaptive F1=0.7502 (test73 封板)
- [x] **Step 4.6: E34b 联合消融** — edge=20, ratio=2.5, merge=1.4, F1=0.8106
- [x] **T3b: Adaptive radius 重扫** — radius=160, F1=0.780 (val71)
- [ ] P2-D/E: LR+Epoch 消融 (论文需要)
- [ ] T7: Adapter Instance 评估
- [x] **T16 Baseline 对比实验 ✅** — 6/7 完成 (StarDist P3 暂缓)
- [x] **T19-abl Box Clipping 消融 ✅** — clipping PQ=0.466 > no-clip PQ=0.437

---

## 关键决策速查

| 决策 | 选择 | 理由 | 来源 |
|------|------|------|------|
| **检测方案** | Hybrid DAPI+Actn2 | 定位+形状 | `dapi.py` |
| **边缘过滤** | 50px | 误删 1.3% | `analyze_stats_final.py` |
| **双核合并** | 1.2x 直径 | 防止误合并邻居 | `dapi.py` |
| **核面积阈值 (DAPI)** | 默认 200/10000；评测锁定 1500/20000 + relative_1.2x | 默认用于运行；锁定参数用于统一评测（已封板） | `dapi.py`, `experiments/ablation_dapi_val/results.json`, `experiments/ablation_detection_lock/results.json` |
| **Adaptive 锁定参数** | radius=200, min_zlines=5, zline_threshold=0.01 | test73 对比参数（已封板） | `experiments/ablation_adaptive_val/results.json`, `experiments/ablation_detection_lock/results.json` |
| **检测参数最终锁定** | E34b(val71) + test73 单次封板 | 避免 test 泄漏，统一对比口径；当前 winner 为 DAPI | `experiments/ablation_detection_e34b/results.json`, `experiments/ablation_detection_lock/results.json` |
| **三通道输入** | 语义映射+Adapter | 适配预训练 ViT | `claude_pipeline_analysis.md` |
| 训练框 | GT 框 | 解耦训练 | `design_decisions.md` |
| 冻结策略 | 仅训练 Decoder | 防过拟合 | `design_decisions.md` |

---

## 📚 新 AI 必读清单 (Required Reading)

**新对话建议先读 3 份文档：`CLAUDE.md` + `inference_standard.md` + `code_inventory.md`。** 如需深入了解，按需查阅：

| 优先级 | 文档 | 用途 |
|--------|------|------|
| **P0** | `CLAUDE.md` (本文件) | 项目总览、任务清单、关键决策 |
| **P0** | [inference_standard.md](docs/inference_standard.md) | 推理与评估口径 SSOT |
| **P0** | [code_inventory.md](docs/code_inventory.md) | 当前活跃代码入口速查 |
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
| **E23** | 02-02 | uint8 截断 Bug 修复 | DAPI F1: 0→78% ✅ |
| **E29** | 02-05 | Instance-level 基线 | BM-1to1=0.593, PQ=0.326 |
| **Phase 1** | 02-10 | Loss 重平衡 + PQ 早停 | **Oracle PQ=0.464, BM=0.695** ⭐ |
| **E33** | 02-06 | 预训练 CellSAM Baseline | BM-Dice=0.111, PQ=0.000 ⚠️ 历史(旧 unified 路径); T24 修正后官方路径 PQ=0.434 |
| **E34** | 02-13~14 | 检测参数锁定 | DAPI F1=0.803 (test73 封板) |
| **P2-A** | 02-15~16 | N/O Loss Fix1-3 | ❌ **终止** (均退化) |
| **T3b** | 02-19 | Adaptive radius 重扫 | F1=0.780 (radius=160) |
| **T16** | 02-21~22 | Baseline 对比 (6 methods) | MedSAM PQ=0.576 最强 ⚠️ |
| **T19-abl** | 02-22 | Box Clipping 消融 | clip PQ=0.466 > no-clip 0.437 ✅ |
| **T12** | 02-23 | Loss 消融 (7组×2seed) | posw=10 (+4.1pp), contour有害 (+2.3pp) ⭐ |
| **BestCfg** | 02-24 | Best Config 验证 (4 runs) | **PQ=0.484** (+3.1pp vs Phase1) ⭐ |
| **T18** | 02-24 | 三通道消融 (2ch/3ch/no-adapter) | 🔄 训练中 (ALICE) |

> 完整记录: [experiments_log.md](docs/experiments_log.md)

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
| 2026-02-24 | **T12 消融完成** (posw=10+contour=off 高置信); **Best Config 验证** PQ=0.484; **T18 三通道部署** (ALICE训练中); 通道顺序改 R=BF/G=Actn2/B=DAPI; 文档同步 |
| 2026-02-22 | **T16 Baseline 完成** (6 methods); Box Clipping 消融; Cellpose d=200 补充 |
| 2026-02-19 | T3b Adaptive radius 重扫完成; 文档优化 (TOC + 早期实验归档) |
| 2026-02-16 | P2-A Fix3 审核完成, **P2-A 终止** |
| 2026-02-15 | P2-A Fix1-2 失败; Fix3 延迟启用方案 |
| 2026-02-14 | E34 检测参数锁定 (test73 封板) |
| 2026-02-13 | Phase 2 Step 3 完成; GT 框面积过滤 Bug 修复; CUDA 模块更新 |
| 2026-02-11 | Phase 1 完成 + test 锁定评估 |
| 2026-02-05 | Instance-level 训练修复; Semantic Dice 无意义发现 |

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
