# Training Scripts

> **Updated**: 2026-02-13
> **Main training entry**: `src/train.py`

## ⚠️ Alice 环境初始化规范

所有 SLURM 脚本必须使用以下环境初始化模板（不要使用 `~/miniconda3` 或 `cuda/11.8`）：

```bash
set -eo pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u   # conda 激活后再启用
```

详见 `docs/alice_quick_reference.md` 第七节踩坑记录。

---

## Script Status

### Active — Phase 2

| Script | Phase | GPU | Notes |
|--------|-------|-----|-------|
| `train_phase2a.sh` | **P2-A** | L4 | ⭐ 当前训练入口 (L_neighbor + L_overlap) |
| `train_phase2a_a100.sh` | **P2-A** | A100 | A100 对照组 |

### Legacy — Phase 1 (已完成，保留备用)

| Script | Phase | GPU | Notes |
|--------|-------|-----|-------|
| `train_phase1_full.sh` | P1 | A100 | Phase 1 正式训练 |
| `train_phase1_l4.sh` | P1 | L4 | Phase 1 L4 训练 |
| `train_phase1_smoke.sh` | P1 | A100 | Phase 1 冒烟测试 (1-epoch) |

### Legacy — Pre-Phase 1 (已过时)

| Script | Notes |
|--------|-------|
| `train_instance_alice.sh` | 早期 instance 训练，已被 Phase 1 替代 |
| `train_instance_20260205.sh` | 旧 instance 训练，带 CUDA fallback |
| `train_ablation_v2.sh` | 消融实验 |
| `train_lr_ablation.sh` | 学习率消融 |
| `train_ablation_l4.sh` | 旧 L4 消融 |
| `train_a100_pending.sh` | A100 待执行配置 |
| `train_semantic.sh` | 语义映射训练 (pre-Phase 0) |
| `train_bf_adapter.sh` | BF adapter 训练 |
| `train_bf_baseline_full.sh` | BF baseline 训练 |

### Utility

| Script | Notes |
|--------|-------|
| `monitor_alice.sh` | Alice 任务监控 |

## Status definitions

- **Active**: 当前阶段使用的脚本。P2-A 训练入口。
- **Legacy (Phase 1)**: Phase 1 已完成，脚本保留但环境已更新。
- **Legacy (Pre-Phase 1)**: 早期脚本，不再使用。考虑归档到 `scripts/archive/`。
