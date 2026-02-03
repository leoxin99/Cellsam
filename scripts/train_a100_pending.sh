#!/bin/bash -l
#SBATCH --job-name=cellsam_a100_pending
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --output=logs/a100_pending_%A_%a.log
#SBATCH --error=logs/a100_pending_%A_%a.err

# ============================================
# PENDING 任务加速 - 使用 A100 分区
# 2 个实验: 3ch_semantic_adapter, bf_adapter
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
  "3ch_semantic_adapter.yaml"
  "bf_adapter.yaml"
)

NAMES=(
  "3ch_Semantic_Adapter"
  "BF_Adapter"
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
