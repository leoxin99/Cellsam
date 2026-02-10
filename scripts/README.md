# Training Scripts

> **Updated**: 2026-02-10
> **Main training entry**: `src/train.py`

## Script Status

| Script | Status | Notes |
|--------|--------|-------|
| `train_instance_20260205.sh` | **active** | Latest instance segmentation training config |
| `train_ablation_v2.sh` | **active** | Current ablation study config |
| `train_lr_ablation.sh` | **active** | Learning rate ablation |
| `train_instance_alice.sh` | review | Alice cluster version, verify if still valid |
| `train_a100_pending.sh` | review | A100 config, pending execution |
| `train_semantic.sh` | legacy | Semantic mapping training (pre-Phase 0) |
| `train_bf_adapter.sh` | legacy | BF adapter training (superseded) |
| `train_bf_baseline_full.sh` | legacy | BF baseline full training (superseded) |
| `train_ablation_l4.sh` | legacy | Old L4 ablation (superseded by v2) |

## Status definitions

- **active**: Currently used or planned for use. Do not move.
- **review**: May still be useful. Verify with project lead before archiving.
- **legacy**: Superseded by newer configs. Will be moved to `scripts/archive/` upon confirmation.

## Archive directory

Legacy scripts confirmed by project lead will be moved to `scripts/archive/`.
