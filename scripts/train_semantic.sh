#!/bin/bash -l
#SBATCH --job-name=cellsam_semantic
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cellsam_%j.log
#SBATCH --error=logs/cellsam_%j.err

# CellSAM Semantic Adapter Training Script (Conda version)
# Usage: sbatch scripts/train_semantic.sh

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================"

# Activate conda environment (requires #!/bin/bash -l)
conda activate cellsam

# Navigate to project
cd ~/CellSam

# Add project to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/CellSam/cellSAM_source

# CellSAM authentication token
export DEEPCELL_ACCESS_TOKEN="X2Od0tJX.te0hEWOzZlRXoJzh5pkvw7l4S5GdpPxs"

# Create logs directory if not exists
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"

# Run training
python src/train.py --config src/config/semantic_adapter.yaml

echo "============================================"
echo "End Time: $(date)"
echo "============================================"
