#!/bin/bash -l
#SBATCH --job-name=t29d_s123
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t29d_s123_%j.log
#SBATCH --error=logs/t29d_s123_%j.err

# T29d: 3ch Official + Actn2 + IndependentChannelAdapter (seed=123)
# Ablation: Does adapter help on model_cp (Phase 2)?
# Baseline: T29c (no adapter, PQ=0.682)

set -eo pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam/cellSAM_source:${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T29d: 3ch + Actn2 + Adapter (L4, seed=123)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "============================================"

EXP_PREFIX="T29d_Adapter_3ch_Actn2"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t29d_adapter_3ch_actn2.yaml --seed 123
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then echo "ERROR"; exit $TRAIN_EXIT; fi

AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)
[ -z "$EXP_DIR" ] && EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)

echo "Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
