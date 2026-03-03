#!/bin/bash -l
#SBATCH --job-name=t18c_s123
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t18c_s123_%j.log
#SBATCH --error=logs/t18c_s123_%j.err

# ============================================
# T18-C Supplement: 3ch NO Adapter, seed=123
# Reason: T18-C seed=42 gave PQ=0.500 (best of all T18)
#         Need 2nd seed to confirm result is robust
# Estimated: ~4h train + ~2min eval
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs experiments/ablation_eval/t18

echo "============================================"
echo "T18-C Supplement: 3ch noAdapter seed=123"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

python src/train.py --config src/config/t18c_3ch_no_adapter.yaml --seed 123
train_exit=$?
echo "Train exit: $train_exit, Time: $(date)"

if [ $train_exit -ne 0 ]; then
  echo "ERROR: Training failed"
  exit $train_exit
fi

# Find latest T18C checkpoint
exp_dir=$(ls -td checkpoints/T18C_3ch_noAdapter_* 2>/dev/null | head -1)
eval_dir="experiments/ablation_eval/t18/t18c_seed123_a100"
mkdir -p "$eval_dir"

if [ -n "$exp_dir" ] && [ -f "$exp_dir/best_model.pt" ]; then
  echo "========== Eval: $exp_dir -> $eval_dir =========="
  python tools/eval_ablation.py --exp-dir "$exp_dir" --output "$eval_dir"
  echo "Eval exit: $?, Time: $(date)"
else
  echo "WARNING: No best_model.pt found in $exp_dir"
fi

echo ""
echo "============================================"
echo "T18-C seed=123 completed. End: $(date)"
echo "============================================"
