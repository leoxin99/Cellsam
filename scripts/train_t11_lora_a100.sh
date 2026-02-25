#!/bin/bash -l
#SBATCH --job-name=t11_lora_a100
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t11_lora_a100_%j.log
#SBATCH --error=logs/t11_lora_a100_%j.err

# ============================================
# T11 LoRA: rank=8 (A100)
# Runs: seed=42, seed=123
# Estimated: 2 × ~6h = ~12h (A100 faster)
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p logs experiments/ablation_eval/t11

echo "============================================"
echo "T11 LoRA rank=8 (A100)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

train_and_eval() {
  local config=$1
  local seed=$2
  local exp_prefix=$3
  local eval_dir=$4
  local label=$5

  echo ""
  echo "========== [$label] Training: $exp_prefix (seed=$seed) =========="
  echo "Start: $(date)"

  local before_dirs
  before_dirs=$(ls -d checkpoints/${exp_prefix}_* 2>/dev/null | sort)

  python src/train.py --config "$config" --seed $seed
  local train_exit=$?
  echo "Train exit: $train_exit, Time: $(date)"

  if [ $train_exit -ne 0 ]; then
    echo "ERROR: Training failed for $exp_prefix seed=$seed"
    return $train_exit
  fi

  local after_dirs
  after_dirs=$(ls -d checkpoints/${exp_prefix}_* 2>/dev/null | sort)
  local exp_dir
  exp_dir=$(comm -13 <(echo "$before_dirs") <(echo "$after_dirs") | head -1)

  if [ -z "$exp_dir" ]; then
    echo "WARNING: Could not identify new checkpoint dir, falling back to ls -td"
    exp_dir=$(ls -td checkpoints/${exp_prefix}_* 2>/dev/null | head -1)
  fi

  mkdir -p "$eval_dir"

  if [ -n "$exp_dir" ] && [ -f "$exp_dir/best_model.pt" ]; then
    echo "========== [$label] Eval: $exp_dir -> $eval_dir =========="
    python tools/eval_ablation.py --exp-dir "$exp_dir" --output "$eval_dir"
    echo "Eval exit: $?, Time: $(date)"
  else
    echo "WARNING: No best_model.pt found in $exp_dir"
  fi
}


# Run 1: rank=8, seed=42
train_and_eval \
  "src/config/t11_lora_r8.yaml" 42 "T11_LoRA_r8" \
  "experiments/ablation_eval/t11/r8_seed42_a100" \
  "1/2 LoRA-r8 seed=42" || true

# Run 2: rank=8, seed=123
train_and_eval \
  "src/config/t11_lora_r8.yaml" 123 "T11_LoRA_r8" \
  "experiments/ablation_eval/t11/r8_seed123_a100" \
  "2/2 LoRA-r8 seed=123" || true

echo ""
echo "============================================"
echo "T11 LoRA A100 runs completed. End: $(date)"
echo "Results: experiments/ablation_eval/t11/"
echo "============================================"
ls -la experiments/ablation_eval/t11/ 2>/dev/null
