#!/bin/bash -l
#SBATCH --job-name=best_cfg
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/best_cfg_%j.log
#SBATCH --error=logs/best_cfg_%j.err

# ============================================
# Best Config: posw=10 + contour=off
# 2 runs: seed=42, seed=123
# + auto Oracle eval after each run
# Estimated: 2 × (~8h train + ~2min eval) = ~16h
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs experiments/ablation_eval/best_config

echo "============================================"
echo "Best Config: posw=10 + contour=off"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

train_and_eval() {
  local seed=$1
  local label=$2

  echo ""
  echo "========== [$label] Training: best_config (seed=$seed) =========="
  echo "Start: $(date)"

  python src/train.py --config src/config/best_config.yaml --seed $seed
  local train_exit=$?
  echo "Train exit: $train_exit, Time: $(date)"

  if [ $train_exit -ne 0 ]; then
    echo "ERROR: Training failed for seed=$seed"
    return $train_exit
  fi

  # Find the latest experiment directory
  local exp_dir
  exp_dir=$(ls -td checkpoints/BestConfig_posw10_noCont_* 2>/dev/null | head -1)

  local eval_out="experiments/ablation_eval/best_config/seed${seed}"
  mkdir -p "$eval_out"

  if [ -n "$exp_dir" ] && [ -f "$exp_dir/best_model.pt" ]; then
    echo "========== [$label] Eval: $exp_dir -> $eval_out =========="
    python tools/eval_ablation.py --exp-dir "$exp_dir" --output "$eval_out"
    echo "Eval exit: $?, Time: $(date)"
  else
    echo "WARNING: No best_model.pt found"
  fi
}


# Run 1: seed=42
train_and_eval 42 "1/2 seed=42" || true

# Run 2: seed=123
train_and_eval 123 "2/2 seed=123" || true

echo ""
echo "============================================"
echo "All runs completed. End: $(date)"
echo "Results: experiments/ablation_eval/best_config/"
echo "============================================"
ls -la experiments/ablation_eval/best_config/ 2>/dev/null
