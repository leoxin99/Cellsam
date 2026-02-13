# CellSAM 代码清单

> **状态**: 🟢 Active — 代码入口速查
> **最后更新**: 2026-02-13
> **事实来源**: 文件系统 (`src/`, `tools/`, `scripts/`)
> **规则**: 新增/修改代码后必须更新此文档

---

## 1. 训练入口

| 文件 | 功能 | 说明 |
|------|------|------|
| `src/train.py` | **训练主入口** | Instance-level training with box clipping |
| `src/config/phase2a_neighbor_overlap.yaml` | **当前配置** ⭐ | Phase 2-A (L_neighbor + L_overlap) |
| `src/config/phase1_rebalance_l4.yaml` | Phase 1 配置 | 已完成，产出 best_model.pt |

### 训练 SLURM 入口

| 脚本 | 状态 | 说明 |
|------|------|------|
| `scripts/train_phase2a.sh` | ⭐ Active | Phase 2-A L4 训练 |
| `scripts/train_phase2a_a100.sh` | ⭐ Active | Phase 2-A A100 对照 |
| `scripts/train_phase1_*.sh` | Legacy | Phase 1 已完成 |

---

## 2. 推理与评估

| 文件 | 功能 | 说明 |
|------|------|------|
| `src/inference/core.py` | **统一推理核心** ⭐ | InferenceConfig + segment_with_boxes |
| `src/inference/postprocess.py` | 后处理 | 面积过滤、形态学操作 |
| `tools/comprehensive_eval.py` | Oracle 评估 | GT boxes → 全指标 (Dice/PQ/SQ/RQ/AJI) |
| `tools/evaluate_e2e.py` | E2E 评估 | DAPI 检测 → 分割 → 评估 |
| `tools/test_unified_regression.py` | 回归测试 | 训练前必跑，防止退化 |
| `tools/smoke_test_e2e.py` | 冒烟测试 | 快速验证 (默认 30 样本) |

---

## 3. 训练前验证

| 文件 | 功能 | 说明 |
|------|------|------|
| `tools/verify_training_config.py` | 配置验证 | 文件/配置/数据/SLURM lint 检查 |
| `tools/test_loss_gradients.py` | 梯度检查 | 验证所有 loss 分支有梯度 |
| `tools/test_checkpoint_format.py` | Checkpoint 格式 | 验证模型保存/加载一致性 |

---

## 4. 核心模块

| 文件 | 功能 | 版本 |
|------|------|------|
| `src/losses/combined.py` | CombinedLoss (Phase 2 含 Neighbor + Overlap) | v4 |
| `src/augmented_dataset.py` | 数据加载 (Instance-level) | v2 |
| `src/detection/dapi.py` | DAPI 核检测 + 框生成 | v4 |
| `src/adapters/channel_adapter.py` | Semantic Channel Adapter | v1 |

---

## 5. 已归档代码 (tools/archive/)

| 目录 | 内容 |
|------|------|
| `tools/archive/legacy_eval/` | E24-E28 旧评估脚本 |
| `tools/archive/legacy_experiment/` | E29 早期推理测试 |
| `tools/archive/tests_deprecated/` | 旧版测试 |

---

## 6. 配置文件速查

### Active

| 文件 | 阶段 | 说明 |
|------|------|------|
| `phase2a_neighbor_overlap.yaml` | P2-A | ⭐ 当前训练 |
| `phase1_rebalance_l4.yaml` | P1 | L4 训练 (已完成) |
| `phase1_rebalance_a100.yaml` | P1 | A100 训练 (已完成) |

### Legacy

| 文件 | 说明 |
|------|------|
| `bf_instance_p1_20260205.yaml` | E29 旧配置 |
| `adapter_instance_p1_20260205.yaml` | E30 旧配置 |
| `bf_instance_p2_20260205.yaml` | E31 旧配置 |
| `adapter_instance_p2_20260205.yaml` | E32 旧配置 |
| `semantic_adapter.yaml` | E21 旧配置 |

---

## 更新日志

| 日期 | 更新 |
|------|------|
| **2026-02-13** | 重构为 Active/Legacy 分类；Phase 2-A 入口加入 |
| 2026-02-05 | 更新全部面积参数为 1024px 缩放 (×0.340) |
| 2026-01-23 | 初始创建 |
