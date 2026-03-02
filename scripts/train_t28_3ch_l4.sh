#!/bin/bash -l
#SBATCH --job-name=t28_3ch_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t28_3ch_l4_%j.log
#SBATCH --error=logs/t28_3ch_l4_%j.err

# ============================================
# T28: Three-Channel Decoder-Only (L4)
# Same as T27a but with BF+DAPI+Actn2 3-channel input
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T28: Three-Channel Decoder-Only (L4)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

EXP_PREFIX="T28_PlanB_3ch"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t28_planb_3ch.yaml
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed"
  exit $TRAIN_EXIT
fi

AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)

if [ -z "$EXP_DIR" ]; then
  echo "WARNING: Could not find new checkpoint dir"
  EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)
fi

echo ""
echo "============================================"
echo "Training complete. Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
echo "============================================"
