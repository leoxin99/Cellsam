#!/bin/bash -l
# =============================================================================
# Instance-level Training SLURM Script (Created 2026-02-05)
# =============================================================================
# Purpose: Submit Phase 1 and Phase 2 instance training jobs
# Experiments: E29-E32
# Key fixes: Instance-level target, box clipping, Instance Dice validation
# =============================================================================

#SBATCH --job-name=inst_train
#SBATCH --output=logs/inst_train_%A_%a.out
#SBATCH --error=logs/inst_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3

# Array job configuration (4 experiments)
CONFIGS=(
    "src/config/bf_instance_p1_20260205.yaml"
    "src/config/adapter_instance_p1_20260205.yaml"
    "src/config/bf_instance_p2_20260205.yaml"
    "src/config/adapter_instance_p2_20260205.yaml"
)

NAMES=(
    "E29_bf_instance_p1"
    "E30_adapter_instance_p1"
    "E31_bf_instance_p2"
    "E32_adapter_instance_p2"
)

# Select config based on array task ID
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Instance-level Training (2026-02-05)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $NAME"
echo "Config: $CONFIG"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Setup environment
# Note: conda is already initialized in .bashrc on ALICE
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "CUDA module not loaded"

# Activate conda environment
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate cellsam

# Navigate to project directory
cd $HOME/CellSam

# Create logs directory
mkdir -p logs

# Print enabled losses
echo ""
echo "Key improvements in this training:"
echo "  1. Instance-level target (cell_id instead of semantic mask)"
echo "  2. Box clipping for pred/target"
echo "  3. Instance Dice for validation"
echo ""

# Run training
python src/train.py --config $CONFIG

echo "============================================"
echo "Training completed: $NAME"
echo "End time: $(date)"
echo "============================================"
