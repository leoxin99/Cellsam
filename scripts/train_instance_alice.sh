#!/bin/bash
#SBATCH --job-name=instance_train
#SBATCH --output=logs/instance_train_%A_%a.out
#SBATCH --error=logs/instance_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1

# Instance-level Training Array Job
# Task 0: BF Instance v1
# Task 1: Semantic Adapter Instance v1

# Array job configuration
CONFIGS=(
    "src/config/bf_instance_v1.yaml"
    "src/config/semantic_adapter_instance_v1.yaml"
)

NAMES=(
    "E29_bf_instance_v1"
    "E30_semantic_adapter_instance_v1"
)

# Select config based on array task ID
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Running: $NAME"
echo "Config: $CONFIG"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Setup environment
module load cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellsam

# Navigate to project directory
cd $HOME/CellSam

# Create logs directory
mkdir -p logs

# Run training with instance-level improvements
# Key changes from previous training:
# - Instance-level target (specific cell, not entire mask)
# - Box clipping for both pred and target
# - Instance Dice for validation
python src/train.py --config $CONFIG

echo "============================================"
echo "Training completed: $NAME"
echo "End time: $(date)"
echo "============================================"
