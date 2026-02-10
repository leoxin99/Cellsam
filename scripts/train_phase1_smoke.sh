#!/bin/bash -l
#SBATCH --job-name=p1_smoke
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/p1_smoke_%j.log
#SBATCH --error=logs/p1_smoke_%j.err

# ============================================
# Phase 1 Smoke Train (1-epoch) — ALICE 环境验证
# 目的: 确认 ALICE 环境能跑通 Phase 1 训练流程
# 通过后再提交 50-epoch 正式任务
# ============================================

echo "============================================"
echo "Phase 1 Smoke Train — ALICE Environment Verification"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

# Setup environment
module load cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsam

cd ~/CellSam
mkdir -p logs checkpoints

# Verify environment
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Run 1-epoch smoke train
python src/train.py --config src/config/phase1_rebalance_smoke.yaml

# Verify output
CKPT_DIR=$(ls -td checkpoints/E_phase1_smoke_verify_* 2>/dev/null | head -1)
if [ -f "$CKPT_DIR/best_model.pt" ]; then
    echo "============================================"
    echo "✅ SMOKE PASSED: best_model.pt generated"
    echo "Checkpoint: $CKPT_DIR"
    ls -lh "$CKPT_DIR/"
    echo "============================================"
    echo ""
    echo ">>> Safe to submit full 50-epoch training:"
    echo ">>> sbatch scripts/train_phase1_full.sh"
else
    echo "============================================"
    echo "❌ SMOKE FAILED: no best_model.pt found"
    echo "Check logs for errors"
    echo "============================================"
fi
