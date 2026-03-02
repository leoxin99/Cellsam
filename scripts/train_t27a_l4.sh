#!/bin/bash -l
#SBATCH --job-name=t27a_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t27a_l4_%j.log
#SBATCH --error=logs/t27a_l4_%j.err

# ============================================
# T27a: Plan B Decoder-Only Fine-Tuning (L4)
#
# Background: Plan B migration to official CellSAM pipeline
# Purpose: Train model_cp.mask_decoder with Focal+IoU Head Loss
# Dependencies: src/train.py, src/config/t27a_planb_decoder.yaml
# Result: checkpoints/T27a_PlanB_DecoderOnly_*/
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T27a: Plan B Decoder-Only (L4)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

# Snapshot checkpoint dirs BEFORE training
EXP_PREFIX="T27a_PlanB_DecoderOnly"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t27a_planb_decoder.yaml
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
