#!/bin/bash -l
#SBATCH --job-name=t32_s42_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t32_s42_l4_%j.log
#SBATCH --error=logs/t32_s42_l4_%j.err

# ============================================
# T32: Stage2-like Neck-Only Baseline (L4, seed=42)
#
# Purpose: Train ONLY encoder neck (freeze encoder+decoder)
#          Stage2-like surrogate for CellSAM paper comparison
# Config: src/config/t32_stage2_like_neck_only.yaml
# Result: checkpoints/T32_NeckOnly_Baseline_seed42_*/
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${HOME}/CellSam/cellSAM_source:${HOME}/CellSam/cellSAM_source/cellSAM:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T32: Neck-Only Baseline (L4, seed=42)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

# Snapshot checkpoint dirs BEFORE training
EXP_PREFIX="T32_NeckOnly_Baseline"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t32_stage2_like_neck_only.yaml --seed 42
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed"
  exit $TRAIN_EXIT
fi

# Find NEW checkpoint dir (before/after snapshot pattern)
AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)

if [ -z "$EXP_DIR" ]; then
  echo "WARNING: Could not identify new checkpoint dir, falling back to ls -td"
  EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)
fi

echo ""
echo "============================================"
echo "Training complete."
echo "Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
echo "============================================"
