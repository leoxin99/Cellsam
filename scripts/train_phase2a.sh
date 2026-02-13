#!/bin/bash -l
#SBATCH --job-name=p2a_neigh
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/p2a_%j.log
#SBATCH --error=logs/p2a_%j.err

set -euo pipefail

# ============================================
# Phase 2-A Training - L4 (24GB)
# Config: phase2a_neighbor_overlap.yaml
# Experiment: E_phase2a_neighbor_overlap
# Delta vs Phase 1: +L_neighbor(0.3) +L_overlap(0.1)
#   Topology/Size OFF (see codex_claude_seg.md Ch17)
# Time: 60h (same as Phase 1)
# ============================================

echo "============================================"
echo "Phase 2-A Training - L_neighbor + L_overlap"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

module load CUDA/12.1.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsam

cd "${SLURM_SUBMIT_DIR:-$HOME/CellSam}"
mkdir -p logs checkpoints

echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Pre-flight: verify gradient gate passes
echo ""
echo "=== Pre-flight: Gradient Gate ==="
python tools/test_loss_gradients.py
# set -e already ensures abort on failure

echo ""
echo "Disk before:"
df -h checkpoints/

# Temporarily allow non-zero exit for training (early stop exits 0, but capture code)
set +e
python src/train.py --config src/config/phase2a_neighbor_overlap.yaml
EXIT_CODE=$?
set -e

CKPT_DIR=$(ls -td checkpoints/E_phase2a_neighbor_overlap_* 2>/dev/null | head -1)
if [ -n "$CKPT_DIR" ] && [ -f "$CKPT_DIR/best_model.pt" ]; then
    echo ""
    echo "Best model metrics:"
    python -c "
import torch
ckpt = torch.load('$CKPT_DIR/best_model.pt', map_location='cpu', weights_only=False)
print(f\"  Epoch: {ckpt.get('epoch', '?')}\")
print(f\"  Best Dice: {ckpt.get('best_dice', 0):.4f}\")
print(f\"  Best PQ:   {ckpt.get('best_pq', 0):.4f}\")
"
    ls -lh "$CKPT_DIR/"
fi

echo "Disk after:"
df -h checkpoints/
echo "Exit: $EXIT_CODE, End: $(date)"
exit $EXIT_CODE
