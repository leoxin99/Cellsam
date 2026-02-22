#!/bin/bash -l
#SBATCH --job-name=abl_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=167:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/abl_l4_%j.log
#SBATCH --error=logs/abl_l4_%j.err

# ============================================
# T12 Loss Ablation — L4 (Round 2, seed=123)
# 7 runs: Phase1 baseline + 6 ablations
# + auto Oracle eval after each run
# Estimated: 7 × (~20h train + ~1h eval) = ~147h
# ============================================

set -o pipefail  # no -e: allow individual runs to fail without killing the whole job

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

SEED=123

echo "============================================"
echo "T12 Loss Ablation — L4 (seed=$SEED)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

cd ~/CellSam
mkdir -p logs checkpoints experiments/ablation_eval

echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ---- Helper: train + eval ----
train_and_eval() {
  local config=$1
  local label=$2

  echo ""
  echo "========== [$label] Training: $config (seed=$SEED) =========="
  echo "Start: $(date)"

  python src/train.py --config src/config/${config}.yaml --seed $SEED
  local train_exit=$?
  echo "Train exit: $train_exit, Time: $(date)"

  if [ $train_exit -ne 0 ]; then
    echo "ERROR: Training failed for $config"
    return $train_exit
  fi

  # Find the latest experiment directory for this config
  local exp_prefix
  exp_prefix=$(python -c "import yaml; c=yaml.safe_load(open('src/config/${config}.yaml')); print(c['output']['experiment_name'])")
  local exp_dir
  exp_dir=$(ls -td checkpoints/${exp_prefix}_* 2>/dev/null | head -1)

  if [ -n "$exp_dir" ] && [ -f "$exp_dir/best_model.pt" ]; then
    echo "========== [$label] Eval: $exp_dir =========="
    python tools/eval_ablation.py --exp-dir "$exp_dir" --output experiments/ablation_eval
    echo "Eval exit: $?, Time: $(date)"
  else
    echo "WARNING: No best_model.pt found for $config"
  fi
}

# ---- Phase1 Baseline (seed=123, Round 2) ----
train_and_eval "phase1_rebalance_l4" "1/7 Phase1-Baseline" || true

# ---- 6 Ablation Groups ----
train_and_eval "ablation_bce_dice_only"    "2/7 Ab-0 BCE+Dice"    || true
train_and_eval "ablation_no_boundary"      "3/7 Ab-1 w/o Boundary" || true
train_and_eval "ablation_no_contour"       "4/7 Ab-2 w/o Contour"  || true
train_and_eval "ablation_no_aji"           "5/7 Ab-3 w/o AJI"      || true
train_and_eval "ablation_no_pq_earlystop"  "6/7 Ab-4 w/o PQ-ES"   || true
train_and_eval "ablation_posw10"           "7/7 Ab-5 posw=10"     || true

echo ""
echo "============================================"
echo "All 7 runs completed. End: $(date)"
echo "Results: experiments/ablation_eval/"
echo "============================================"
