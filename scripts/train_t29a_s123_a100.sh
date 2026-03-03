#!/bin/bash -l
#SBATCH --job-name=t29a_s123_a100
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t29a_s123_a100_%j.log
#SBATCH --error=logs/t29a_s123_a100_%j.err

# T29a: BF-only Official Encoding, SEED=123
# Checkpoint dir: T29a_Official_BF_seed123_YYYYMMDD_HHMMSS

set -o pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T29a: Official BF-only (A100, seed=123)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

EXP_PREFIX="T29a_Official_BF_seed123"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t29a_official_bf.yaml --seed 123
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then echo "ERROR"; exit $TRAIN_EXIT; fi

AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)
[ -z "$EXP_DIR" ] && EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)

echo "Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
