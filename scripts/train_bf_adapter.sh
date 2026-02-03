#!/bin/bash -l
#SBATCH --job-name=cellsam_bf_adapter
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bf_adapter_%j.log
#SBATCH --error=logs/bf_adapter_%j.err

# ============================================
# BF + Adapter 消融实验
# ============================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start Time: $(date)"
echo "============================================"

# 激活 Conda 环境
conda activate cellsam

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:~/CellSam/cellSAM_source
cd ~/CellSam
mkdir -p logs checkpoints

# 验证环境
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo "============================================"
echo "Running Experiment: BF_Adapter"
echo "Config: bf_adapter.yaml"
echo "============================================"

# 运行训练
python src/train.py --config src/config/bf_adapter.yaml

echo "============================================"
echo "Experiment BF_Adapter Completed!"
echo "End Time: $(date)"
echo "============================================"
