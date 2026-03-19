#!/bin/bash -l
#SBATCH --job-name=t33g_dcm_q35_s123_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t33g_dcm_q35_s123_l4_%j.log
#SBATCH --error=logs/t33g_dcm_q35_s123_l4_%j.err

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam/src:${HOME}/CellSam:${HOME}/CellSam/cellSAM_source:${HOME}/CellSam/cellSAM_source/cellSAM:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs checkpoints

echo "============================================"
echo "T33g dapi_cm candidate-aware q35 (L4, seed=123)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

python tools/train_cellfinder_candidate_aware.py \
    --seed 123 \
    --epochs 150 \
    --patience 20 \
    --batch-size 4 \
    --num-workers 4 \
    --num-queries 35 \
    --candidate-mode dapi_cm \
    --profile-name locked_eval \
    --prior-mode strict \
    --early-stop-metric candidate_aligned_f1_0p3 \
    --output-dir "checkpoints/T33g_CandidateAware_dapicm_strict_q35_f1p03_seed123_$(date +%Y%m%d_%H%M%S)"

TRAIN_EXIT=$?
echo "Train exit: $TRAIN_EXIT, Time: $(date)"
exit $TRAIN_EXIT

