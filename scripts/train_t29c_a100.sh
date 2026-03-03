#!/bin/bash -l
#SBATCH --job-name=t29c_a100
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t29c_a100_%j.log
#SBATCH --error=logs/t29c_a100_%j.err

# T29c: 3ch Official + Actn2 (R=Actn2, G=DAPI, B=BF)

set -o pipefail
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T29c: Official 3ch + Actn2 (A100, seed=42)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

EXP_PREFIX="T29c_Official_3ch_Actn2"
BEFORE_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)

python src/train.py --config src/config/t29c_official_3ch_actn2.yaml
TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then echo "ERROR"; exit $TRAIN_EXIT; fi

AFTER_DIRS=$(ls -d checkpoints/${EXP_PREFIX}_* 2>/dev/null | sort)
EXP_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -1)
[ -z "$EXP_DIR" ] && EXP_DIR=$(ls -td checkpoints/${EXP_PREFIX}_* 2>/dev/null | head -1)

echo "Checkpoint dir: $EXP_DIR"
echo "End: $(date)"
