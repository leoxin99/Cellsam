#!/bin/bash -l
#SBATCH --job-name=t18_ctrl
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t18_ctrl_%j.log
#SBATCH --error=logs/t18_ctrl_%j.err

# ============================================
# T18-Control: BF×3 Continue-Training (Critical Control)
# Purpose: Rule out "extra training" as confound for T18 gains
# Setup: Identical to T18 except input stays BF×3
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
echo "T18-Control: BF Continue-Training"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

# Snapshot existing dirs
before_dirs=$(ls -d checkpoints/T18_Control_BF_ContinueTrain_* 2>/dev/null | sort)

python src/train.py --config src/config/t18_control_bf_continue.yaml --seed 42
train_exit=$?
echo "Train exit: $train_exit, Time: $(date)"

if [ $train_exit -ne 0 ]; then
  echo "ERROR: Training failed"
  exit $train_exit
fi

# Find NEW checkpoint dir
after_dirs=$(ls -d checkpoints/T18_Control_BF_ContinueTrain_* 2>/dev/null | sort)
exp_dir=$(comm -13 <(echo "$before_dirs") <(echo "$after_dirs") | head -1)

if [ -z "$exp_dir" ]; then
  exp_dir=$(ls -td checkpoints/T18_Control_BF_ContinueTrain_* 2>/dev/null | head -1)
fi

eval_dir="experiments/ablation_eval/t18/t18_control_bf_seed42_l4"
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
echo "T18-Control completed. $(date)"
echo "============================================"
cat "$eval_dir"/*.json 2>/dev/null
