#!/bin/bash -l
#SBATCH --job-name=t30_s123_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t30_s123_l4_%j.log
#SBATCH --error=logs/t30_s123_l4_%j.err

# T30: LoRA on Encoder Q/V (BF-only, seed=123)
# Checkpoint dir: T30_LoRA_QV_BF_seed123_YYYYMMDD_HHMMSS

set -o pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T30: LoRA Q/V BF-only (L4, seed=123)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

EXP_PREFIX="T30_LoRA_QV_BF_seed123"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t30_lora_qv_bf.yaml --seed 123
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then echo "ERROR"; exit $TRAIN_EXIT; fi

AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)
[ -z "$EXP_DIR" ] && EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)

echo "Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
