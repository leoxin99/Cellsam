#!/bin/bash -l
#SBATCH --job-name=t18_a100
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/t18_a100_%j.log
#SBATCH --error=logs/t18_a100_%j.err

# ============================================
# T18: Three-Channel Experiments (A100)
# Runs: T18-A seed=123, T18-B seed=123
# Estimated: 2 × (~4h train + ~2min eval) = ~8h
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
echo "T18: Three-Channel Experiments (A100)"
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

  python src/train.py --config "$config" --seed $seed
  local train_exit=$?
  echo "Train exit: $train_exit, Time: $(date)"

  if [ $train_exit -ne 0 ]; then
    echo "ERROR: Training failed for $exp_prefix seed=$seed"
    return $train_exit
  fi

  # Find the latest experiment directory
  local exp_dir
  exp_dir=$(ls -td checkpoints/${exp_prefix}_* 2>/dev/null | head -1)

  mkdir -p "$eval_dir"

  if [ -n "$exp_dir" ] && [ -f "$exp_dir/best_model.pt" ]; then
    echo "========== [$label] Eval: $exp_dir -> $eval_dir =========="
    python tools/eval_ablation.py --exp-dir "$exp_dir" --output "$eval_dir"
    echo "Eval exit: $?, Time: $(date)"
  else
    echo "WARNING: No best_model.pt found in $exp_dir"
  fi
}


# Run 1: T18-A (2ch BF+Actn2) seed=123
train_and_eval \
  "src/config/t18a_2ch.yaml" 123 "T18A_2ch_BF_Actn2" \
  "experiments/ablation_eval/t18/t18a_seed123" \
  "1/2 T18-A seed=123" || true

# Run 2: T18-B (3ch BF+DAPI+Actn2) seed=123
train_and_eval \
  "src/config/t18b_3ch.yaml" 123 "T18B_3ch_BF_DAPI_Actn2" \
  "experiments/ablation_eval/t18/t18b_seed123" \
  "2/2 T18-B seed=123" || true

echo ""
echo "============================================"
echo "All A100 runs completed. End: $(date)"
echo "Results: experiments/ablation_eval/t18/"
echo "============================================"
ls -la experiments/ablation_eval/t18/ 2>/dev/null
