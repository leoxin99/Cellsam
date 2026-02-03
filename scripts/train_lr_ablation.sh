#!/bin/bash -l
#SBATCH --job-name=cellsam_lr_ablation
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --output=logs/lr_ablation_%A_%a.log
#SBATCH --error=logs/lr_ablation_%A_%a.err

# ============================================
# Learning Rate 消融实验
# 2 个实验: lr_5e-5, lr_2e-4
# ============================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

# 实验配置数组
CONFIGS=(
  "lr_5e-5.yaml"
  "lr_2e-4.yaml"
)

NAMES=(
  "LR_5e-5"
  "LR_2e-4"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Running Experiment: $NAME"
echo "Config: $CONFIG"
echo "============================================"

# 运行训练
python src/train.py --config src/config/$CONFIG

echo "============================================"
echo "Experiment $NAME Completed!"
echo "End Time: $(date)"
echo "============================================"
