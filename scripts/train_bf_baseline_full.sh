#!/bin/bash
#SBATCH --job-name=bf_baseline_full
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bf_baseline_%j.log
#SBATCH --error=logs/bf_baseline_%j.err

# BF Baseline Full Dataset Training
# For fair comparison with Semantic Adapter (both on 334 training samples)

echo "============================================"
echo "BF Baseline Full Dataset Training"
echo "Start Time: $(date)"
echo "============================================"

# Activate environment
source ~/.bashrc
conda activate cellsam

# Navigate to project
cd ~/CellSam

# Check GPU
nvidia-smi

echo ""
echo "Training BF Baseline on FULL dataset (334 samples)..."
echo "Config: src/config/bf_baseline_full.yaml"
echo ""

# Run training
python src/train.py --config src/config/bf_baseline_full.yaml

echo ""
echo "============================================"
echo "End Time: $(date)"
echo "============================================"
