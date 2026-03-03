#!/bin/bash -l
#SBATCH --job-name=t18_fix_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/t18_fix_eval_%j.log
#SBATCH --error=logs/t18_fix_eval_%j.err

# ============================================
# Fix: Re-evaluate T18-A and T18-B L4 (seed42) checkpoints
# Bug: Original eval used ls -td which picked A100 checkpoint
# Now: Explicitly specify the correct L4 checkpoint directories
# ============================================

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

export PYTHONPATH="${HOME}/CellSam:${PYTHONPATH:-}"
cd ~/CellSam
mkdir -p experiments/ablation_eval/t18

echo "============================================"
echo "T18 Fix: Re-evaluate L4 seed=42 checkpoints"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================"

# Fix 1: T18-A seed42 L4 (directory _171414)
echo ""
echo "=== T18-A seed42 L4: T18A_2ch_BF_Actn2_20260224_171414 ==="
# Backup old (wrong) eval
mv experiments/ablation_eval/t18/t18a_seed42_l4 experiments/ablation_eval/t18/t18a_seed42_l4_WRONG 2>/dev/null || true
mkdir -p experiments/ablation_eval/t18/t18a_seed42_l4
python tools/eval_ablation.py \
  --exp-dir checkpoints/T18A_2ch_BF_Actn2_20260224_171414 \
  --output experiments/ablation_eval/t18/t18a_seed42_l4
echo "T18-A fix exit: $?, Time: $(date)"

# Fix 2: T18-B seed42 L4 (directory _185215)
echo ""
echo "=== T18-B seed42 L4: T18B_3ch_BF_DAPI_Actn2_20260224_185215 ==="
# Backup old (wrong) eval
mv experiments/ablation_eval/t18/t18b_seed42_l4 experiments/ablation_eval/t18/t18b_seed42_l4_WRONG 2>/dev/null || true
mkdir -p experiments/ablation_eval/t18/t18b_seed42_l4
python tools/eval_ablation.py \
  --exp-dir checkpoints/T18B_3ch_BF_DAPI_Actn2_20260224_185215 \
  --output experiments/ablation_eval/t18/t18b_seed42_l4
echo "T18-B fix exit: $?, Time: $(date)"

echo ""
echo "============================================"
echo "Fix evals done. $(date)"
echo "============================================"
echo ""
echo "=== New T18-A seed42 result ==="
cat experiments/ablation_eval/t18/t18a_seed42_l4/*.json 2>/dev/null
echo ""
echo "=== New T18-B seed42 result ==="
cat experiments/ablation_eval/t18/t18b_seed42_l4/*.json 2>/dev/null
