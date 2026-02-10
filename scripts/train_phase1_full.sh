#!/bin/bash -l
#SBATCH --job-name=p1_rebalance
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/p1_rebalance_%j.log
#SBATCH --error=logs/p1_rebalance_%j.err

# ============================================
# Phase 1: Loss Weight Rebalance + PQ Early Stop
# 50 epochs, PQ early stop (patience=15)
# Changes from E29: boundary 0.5→1.5, contour 0.1→0.3, PQ early stop ON
# Topology/size OFF for clean ablation
#
# Checkpoint retention: keeps only last 3 periodic saves (save_every=5)
# Estimated disk: best_model.pt (1.1GB) + 3 periodic (3.3GB) = ~4.4GB
#
# Usage:
#   sbatch scripts/train_phase1_full.sh                         # gpu-a100-80g (default)
#   sbatch -p gpu-l4-24g scripts/train_phase1_full.sh            # L4 分区 (并行备选)
# ============================================

echo "============================================"
echo "Phase 1: Loss Weight Rebalance Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
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
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Check disk before starting
echo "Disk usage before training:"
df -h ~/CellSam/checkpoints/

# Run Phase 1 training
echo ""
echo "Starting Phase 1 training..."
echo "Config: src/config/phase1_rebalance.yaml"
echo "Key changes: boundary=1.5, contour=0.3, PQ early stop=ON"
echo ""

python src/train.py --config src/config/phase1_rebalance.yaml

EXIT_CODE=$?

# Post-training report
echo ""
echo "============================================"
echo "Training Exit Code: $EXIT_CODE"
echo "End Time: $(date)"

# Find the output directory
CKPT_DIR=$(ls -td checkpoints/E_phase1_rebalance_* 2>/dev/null | head -1)
if [ -n "$CKPT_DIR" ]; then
    echo "Checkpoint dir: $CKPT_DIR"
    echo "Contents:"
    ls -lh "$CKPT_DIR/"
    
    # Extract best metrics from checkpoint
    if [ -f "$CKPT_DIR/best_model.pt" ]; then
        echo ""
        echo "Best model metrics:"
        python -c "
import torch
ckpt = torch.load('$CKPT_DIR/best_model.pt', map_location='cpu', weights_only=False)
print(f\"  Epoch: {ckpt.get('epoch', '?')}\")
print(f\"  Best Dice: {ckpt.get('best_dice', 0):.4f}\")
print(f\"  Best PQ:   {ckpt.get('best_pq', 0):.4f}\")
"
    fi
fi

echo "Disk usage after training:"
df -h ~/CellSam/checkpoints/
echo "============================================"

exit $EXIT_CODE
