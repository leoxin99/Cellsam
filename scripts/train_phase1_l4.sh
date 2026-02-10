#!/bin/bash -l
#SBATCH --job-name=p1_l4
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/p1_l4_%j.log
#SBATCH --error=logs/p1_l4_%j.err

# ============================================
# Phase 1 Training — L4 (24GB)
# Config: phase1_rebalance_l4.yaml
# Experiment: E_phase1_rebalance_l4
# Time: 60h (L4 slower than A100, needs headroom)
# ============================================

echo "============================================"
echo "Phase 1 Training — L4"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

module load cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsam

cd ~/CellSam
mkdir -p logs checkpoints

echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo "Disk before:"
df -h ~/CellSam/checkpoints/

python src/train.py --config src/config/phase1_rebalance_l4.yaml

EXIT_CODE=$?

CKPT_DIR=$(ls -td checkpoints/E_phase1_rebalance_l4_* 2>/dev/null | head -1)
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
df -h ~/CellSam/checkpoints/
echo "Exit: $EXIT_CODE, End: $(date)"
exit $EXIT_CODE
